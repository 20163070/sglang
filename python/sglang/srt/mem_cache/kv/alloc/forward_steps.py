"""The forwards a lazily compacting unified pool may still be read or written by.

Under the overlap scheduler, the forward launched before a result is processed
may still read the pages of a request that result ends (its prepared page table
names them) and write the request's next slot. Its shared metadata reads end
before the result is processed (the scheduler's write-after-read barrier), so a
free unbinds the ids and returns them at once, as a static pool does. Only the
physical pages wait: a unified pool also writes a freed page from the schedule
stream when it hands it out (zeroing) or compacts into it, and the schedule
stream is not ordered after that forward.

So each band keeps the physical pages freed while a forward is in flight
pending, keyed by the newest unprocessed forward that recorded its
`forward_done` event, until the scheduler has processed that forward's result
(`Band.release_pending_through`). The point is logical, the same on every rank.
Pending pages count as free in every capacity check; a take that needs them
before then makes its stream wait that event first, and takes exactly the pages
it needs: the rest stay pending until processed.

Every launched step gets a launch record (`ForwardLaunch`) that its result
carries to processing. A step without a forward (a prebuilt batch) records no
event: it is never a key and its processing advances nothing. No managed page
is freed or allocated between a step's launch and the recording of its event
(`ForwardSteps.forward_recorded` asserts it), so every pending page has an event
to wait.

Processing makes a forward's pages reusable because their dependencies are
established for the schedule stream by then: the scheduler releases them at the
end of the result's processing, after the processor synchronized a completion
that covers `forward_done` (`Scheduler._on_step_processed`), so every forward up
to that one has finished. Other consumers inherit the order: a HiCache copy
stream waits the schedule stream's work before it starts, and a disaggregation
peer gets a page id only after a host-observed completion on the schedule
stream.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch


class ForwardLaunch(NamedTuple):
    """A launched step's record, carried by its result to processing."""

    seq: int


class _Batch(NamedTuple):
    """The physical pages freed while ``step`` was the newest unprocessed
    recorded forward, as freed (one chunk per free; a split keeps a view into
    the chunk, never a copy), the event that forward records, and their
    host-known count."""

    step: int
    event: Optional[torch.cuda.Event]
    pages: Tuple[torch.Tensor, ...]
    count: int


class PendingPages:
    """A band's physical pages pending until their forward is processed,
    oldest forward first, one batch per forward. Immutable: every change
    returns a new value, so a band rebinds it (a capacity field, which bumps
    the capacity epoch) and an allocation rollback restores the old one.
    Shared by end and float bands."""

    __slots__ = ("_batches", "count")

    def __init__(self, batches: Tuple[_Batch, ...] = ()) -> None:
        self._batches = batches
        self.count = sum(batch.count for batch in batches)

    def __bool__(self) -> bool:
        return self.count > 0

    def __len__(self) -> int:
        """The number of batches."""
        return len(self._batches)

    def pages(self) -> List[torch.Tensor]:
        """Every pending chunk, in batch order."""
        return [pages for batch in self._batches for pages in batch.pages]

    def add(
        self,
        step: int,
        event: Optional[torch.cuda.Event],
        pages: torch.Tensor,
        count: int,
    ) -> PendingPages:
        """These pages were freed while ``step`` was the key."""
        batches = self._batches
        if batches and batches[-1].step == step:
            last = batches[-1]
            merged = _Batch(step, event, last.pages + (pages,), last.count + count)
            return PendingPages(batches[:-1] + (merged,))
        assert not batches or batches[-1].step < step, (
            f"pages pending for forward {step} after forward {batches[-1].step}"
        )
        return PendingPages(batches + (_Batch(step, event, (pages,), count),))

    def claim(
        self, count: int
    ) -> Tuple[List[Tuple[Optional[torch.cuda.Event], torch.Tensor]], PendingPages]:
        """Detach the oldest ``count`` pages, chunk by chunk, splitting one
        chunk as a view if needed (its rest stays pending under its forward):
        the chunks with the event each must wait, and what stays pending."""
        assert 0 <= count <= self.count, f"claiming {count} of {self.count} pages"
        taken: List[Tuple[Optional[torch.cuda.Event], torch.Tensor]] = []
        rest = list(self._batches)
        while count > 0:
            batch = rest.pop(0)
            if batch.count <= count:
                taken += [(batch.event, chunk) for chunk in batch.pages]
                count -= batch.count
                continue
            chunks, left = list(batch.pages), count
            while left > 0:
                chunk = chunks.pop(0)
                n = int(chunk.shape[0])
                if n > left:
                    chunks.insert(0, chunk[left:])
                    chunk, n = chunk[:left], left
                taken.append((batch.event, chunk))
                left -= n
            rest.insert(
                0, _Batch(batch.step, batch.event, tuple(chunks), batch.count - count)
            )
            count = 0
        return taken, PendingPages(tuple(rest))

    def processed_through(self, step: int) -> Tuple[List[torch.Tensor], PendingPages]:
        """Detach the batches of forwards up to ``step``, processed: their
        chunks, and what stays pending."""
        n = 0
        while n < len(self._batches) and self._batches[n].step <= step:
            n += 1
        if n == 0:
            return [], self
        out = [pages for batch in self._batches[:n] for pages in batch.pages]
        return out, PendingPages(self._batches[n:])


NO_PENDING_PAGES = PendingPages()


class ForwardSteps:
    """Launch records, and the `forward_done` event of each unprocessed
    forward that recorded one; one per lazy unified chain."""

    def __init__(self, bands) -> None:
        self._bands = tuple(bands)
        for band in self._bands:
            band.forward_steps = self
        self._launched = 0
        # The newest launch, until its forward records its event or its result
        # is processed.
        self._open: Optional[int] = None
        # Managed frees and allocations while a launch is open.
        self._open_activity = 0
        # Unprocessed recorded forwards, in launch order, with their events.
        self._events: Dict[int, Optional[torch.cuda.Event]] = {}
        # The newest recorded forward processed: the processed prefix.
        self._processed = 0

    def forward_launched(self) -> ForwardLaunch:
        """A step was launched; its forward, if it has one, is enqueued next."""
        self._launched += 1
        self._open = self._launched
        self._open_activity = 0
        return ForwardLaunch(self._launched)

    def forward_recorded(self, event: Optional[torch.cuda.Event]) -> None:
        """The newest step's forward is enqueued and ``event`` records its end:
        pages freed from now on stay pending under this forward."""
        seq = self._open
        assert seq is not None and seq == self._launched, (
            "a forward was recorded with no step launched"
        )
        assert self._open_activity == 0, (
            f"{self._open_activity} managed frees or allocations between step "
            f"{seq}'s launch and its forward's event"
        )
        # A fresh event per forward: pending pages wait the one they name.
        assert event is None or all(e is not event for e in self._events.values()), (
            "a forward_done event recorded again while a forward in flight holds it"
        )
        self._events[seq] = event
        self._open = None

    def note_activity(self) -> None:
        """A band freed or allocated managed pages."""
        if self._open is not None:
            self._open_activity += 1

    def pending_key(self) -> Optional[Tuple[int, Optional[torch.cuda.Event]]]:
        """The forward, and its event, that pages freed now stay pending under:
        the newest unprocessed recorded forward. None when none is in flight:
        the page is free at once."""
        if not self._events:
            return None
        seq = next(reversed(self._events))
        return seq, self._events[seq]

    def forward_processed(self, launch: Optional[ForwardLaunch]) -> None:
        """The result of the step launched as ``launch`` was processed, after
        a completion covering its forward was synchronized on the host. A
        recorded forward advances the processed prefix: the pages pending under
        it, and under every earlier forward, join their band's free list, the
        only point they return without a take. A step without a recorded
        forward advances nothing."""
        if launch is None:
            return
        if self._open == launch.seq:
            self._open = None
        if launch.seq not in self._events:
            return
        oldest = next(iter(self._events))
        assert launch.seq == oldest, (
            f"recorded forward {launch.seq} processed before forward {oldest}"
        )
        del self._events[oldest]
        self._processed = oldest
        for band in self._bands:
            band.release_pending_through(oldest)
        if not self._events:
            self.check_idle()

    def in_flight(self) -> bool:
        """Whether a recorded forward is unprocessed."""
        return bool(self._events)

    def check_idle(self) -> None:
        """With no recorded forward in flight, no page is pending."""
        for band in self._bands:
            assert not band._pending_until_retire, (
                f"{band.sub_pool_name}: {band._pending_until_retire_pages} pages "
                f"pending with no forward in flight"
            )


def forward_steps_of(allocator) -> Optional[ForwardSteps]:
    """The pool's forward steps, if its freed pages wait for forwards in flight."""
    steps = getattr(allocator, "forward_steps", None)
    return steps if isinstance(steps, ForwardSteps) else None
