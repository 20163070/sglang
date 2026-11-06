"""Frees of unified KV pages held until the forwards in flight are processed.

Under the overlap scheduler, the forward launched before a result is processed
may still read the pages of a request that result ends (its prepared page table
names them) and write the request's next slot. A static pool hands a freed page
out again only to a later forward, which the forward stream orders after the
running one. A unified pool also writes a page from the schedule stream when it
hands it out (zeroing) or compacts into it, and the schedule stream is not
ordered after the running forward.

So the pool itself holds every free that reaches a band while a forward is
launched and unprocessed: the free, with copies of its ids, waits in a queue
keyed by the newest such forward, and runs once every forward up to it has been
processed (`forward_retired`, called once per processed step, in launch order:
the host has synchronized the forward while processing its result). The point
is the same on every rank, so the pools stay identical across ranks; no stream
wait and no host sync is added. Callers free as they always did. Until the
release the pages stay bound: no take, zeroing or compaction destination reaches
them, and every capacity check counts them as occupied.

A capacity check that fails while the queue is not empty drains it (`drain`):
the host waits for the forwards launched so far, and everything held is
released before the check runs again. The trigger is logical, so every rank
drains at the same point; it costs that step's overlap, under pressure only.

Tree evictions free at once (`quiet_frees`): a node a forward in flight reads is
locked by its request, and the unlock of a request in a launched forward waits
here too (`defer`).
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Deque, Iterator, NamedTuple, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv.alloc.band import Band


class _Held(NamedTuple):
    # The newest step launched when the release was queued.
    step: int
    release: Callable[[], None]
    # A held band free: its band and its distinct virtual pages.
    band: Optional[Band] = None
    pages: Optional[torch.Tensor] = None


class ReleaseQueue:
    """One per lazy unified chain; every band of the chain consults it."""

    def __init__(self, bands, forward_stream=None) -> None:
        self._bands = tuple(bands)
        for band in self._bands:
            band.release_queue = self
            # Checked when ids are bound (`UnitIds.bind`, debug checks only).
            band.ids.held_pages = partial(self.held_pages, band)
        # The stream the forwards run on; `drain` waits for it on the host.
        self._forward_stream = forward_stream
        self._launched = 0
        self._retired = 0
        # Every step up to this one has finished on the device (`drain`).
        self._drained = 0
        self._queued: Deque[_Held] = deque()
        self._quiet = 0
        # Counts the releases that gave anything back, by retirement or drain:
        # capacity came back if it moved (`Scheduler._on_step_processed`).
        self.release_generation = 0

    # -- forwards --

    def forward_launched(self) -> None:
        """A step was launched; frees from now on wait for it."""
        self._launched += 1

    def forward_retired(self) -> None:
        """The oldest launched step was processed: run, in queue order, every
        held free that no unprocessed forward may still touch."""
        assert self._retired < self._launched, "a step retired that never launched"
        self._retired += 1
        self._release_through(self.quiet_through)

    @property
    def newest_launched(self) -> int:
        """The sequence number of the newest launched step."""
        return self._launched

    @property
    def quiet_through(self) -> int:
        """Every step up to this sequence number touches no page any more."""
        return max(self._retired, self._drained)

    def in_flight(self) -> bool:
        """Whether a launched step may still touch the pool's pages."""
        return self._launched > self.quiet_through

    def drain(self) -> bool:
        """Wait on the host for every step launched so far and release
        everything held; return whether anything was held. Frees made after
        it run at once until the next launch."""
        if not self._queued:
            return False
        if self._forward_stream is not None:
            self._forward_stream.synchronize()
        self._drained = self._launched
        self._release_through(self._drained)
        return True

    def drain_if_held(self, band: Band, pages: torch.Tensor) -> None:
        """Drain before ``pages`` of ``band`` are bound to other storage when
        any of them has a free held: that free, resolved through the new
        binding, would release the new storage. Rare paths only (it reads the
        device to decide)."""
        held = self.held_pages(band)
        if held is not None and bool(torch.isin(pages.to(torch.int64), held).any()):
            self.drain()

    def _release_through(self, step: int) -> None:
        released = False
        with self._quiet_scope():
            while self._queued and self._queued[0].step <= step:
                self._queued.popleft().release()
                released = True
        if released:
            self.release_generation += 1

    # -- frees --

    @contextmanager
    def _quiet_scope(self) -> Iterator[None]:
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    def quiet_frees(self):
        """Frees made in the block run at once: the caller has proven that no
        forward in flight reads or writes their pages."""
        return self._quiet_scope()

    def hold_free(
        self, band: Band, free_index: torch.Tensor, pages: Optional[torch.Tensor]
    ) -> bool:
        """Queue a band free while a forward is in flight; return whether it
        was queued. The ids are copied: a caller may pass a view of a row that
        is reused before the release. Its distinct pages are derived now, as
        the band derives them at the free, so the held count is exact."""
        if self._quiet or not self.in_flight():
            return False
        index = free_index.clone()
        if pages is not None:
            pages = pages.clone()
        elif band.page_size == 1:
            pages = index.to(torch.int64)
        else:
            pages = torch.unique(index.to(torch.int64) // band.page_size)
        self._queued.append(
            _Held(self._launched, partial(band.free, index, _pages=pages), band, pages)
        )
        return True

    def defer(self, step: Callable[[], None]) -> bool:
        """Queue ``step`` (a prefix-tree unlock) behind the frees made so far
        while a forward is in flight; return False, doing nothing, otherwise."""
        if self._quiet or not self.in_flight():
            return False
        self._queued.append(_Held(self._launched, step))
        return True

    # -- accounting --

    def held_tokens(self, band: Band) -> int:
        """Tokens of ``band`` that held frees still own."""
        pages = sum(
            int(held.pages.numel()) for held in self._queued if held.band is band
        )
        return pages * band.page_size

    def held_pages(self, band: Band) -> Optional[torch.Tensor]:
        """The virtual pages of ``band`` whose free is held, if any."""
        pages = [held.pages for held in self._queued if held.band is band]
        return torch.cat(pages) if pages else None

    def __len__(self) -> int:
        return len(self._queued)

    def clear(self) -> None:
        """Drop every held free (the pool is being reset, nothing in flight)."""
        self._queued.clear()


def release_queue_of(allocator) -> Optional[ReleaseQueue]:
    """The pool's release queue, if its frees wait for the forwards in flight."""
    queue = getattr(allocator, "release_queue", None)
    return queue if isinstance(queue, ReleaseQueue) else None


def drain_held(allocator) -> bool:
    """For a capacity check that failed: if the pool holds frees for the
    forwards in flight, wait for those forwards, release the frees and return
    True (the caller checks again); otherwise return False."""
    queue = release_queue_of(allocator)
    return queue is not None and queue.drain()
