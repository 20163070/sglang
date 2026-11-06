"""A unified pool frees a page logically at once, as a static pool does, and
hands its physical page out again only once the forwards that may still touch
it are done.

Under the overlap scheduler, the forward launched before a result is processed
may still read the pages of a request that result ends and write its next
slot. Its shared metadata reads end before the result is processed (the
scheduler's write-after-read barrier), so the ids go back at once. The physical
page is pending until that step is processed: it counts as free, a take
reaches it only after its stream waits for the forward, and non-urgent
compaction leaves it in place.
"""

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardSteps
from sglang.srt.mem_cache.kv.alloc.pricing import no_relief_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftBand,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows, kv_storage_of
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_mamba_cache,
    build_swa_cache,
    build_swa_pool,
    build_tri_pool,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _physical(band, token_ids):
    pages = token_ids.to(torch.int64)[:: band.page_size] // band.page_size
    return band.ids.virtual_to_physical[pages].clone()


def _pending(band):
    pages = band._pending_until_retire.pages()
    return set(torch.cat(pages).tolist()) if pages else set()


def _launch(steps, event=None):
    """Launch a step and record its forward's event; the launch record."""
    launch = steps.forward_launched()
    steps.forward_recorded(event)
    return launch


class TestFreedPagesPendUntilTheirStepIsProcessed(CustomTestCase):
    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator = build_swa_pool(page_size=4, lazy=True)
        self.steps = self.allocator.forward_steps
        self.owner = self.allocator.full_attn_allocator

    def test_only_a_lazy_chain_counts_steps(self):
        self.assertIsInstance(self.steps, ForwardSteps)
        self.assertIsNone(build_swa_pool(page_size=4, lazy=False).forward_steps)

    def test_a_freed_page_is_unbound_at_once_counted_free_and_pending(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(16)
        physical = set(_physical(owner, row).tolist())
        free_before = allocator.available_size()

        launch = _launch(steps)
        allocator.free_segment(row, start_pos=0)

        self.assertEqual(allocator.available_size(), free_before + 16)
        self.assertTrue(bool((owner.ids.virtual_to_physical[row[::4] // 4] < 0).all()))
        self.assertTrue(
            set((row[::4] // 4).tolist()) <= set(owner.free_virtual_ids.tolist())
        )
        self.assertEqual(_pending(owner), physical)
        self.assertFalse(physical & set(owner._free_phys_pages.tolist()))

        steps.forward_processed(launch)
        self.assertEqual(_pending(owner), set())
        self.assertTrue(physical <= set(owner._free_phys_pages.tolist()))
        self.assertEqual(allocator.available_size(), free_before + 16)

    def test_a_free_with_no_step_in_flight_is_a_hole_at_once(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(8)
        launch = _launch(steps)
        steps.forward_processed(launch)
        allocator.free_segment(row, start_pos=0)
        self.assertEqual(_pending(owner), set())

    def test_frees_of_one_step_share_one_batch_kept_as_freed(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(2)]
        _launch(steps)
        for row in rows:
            allocator.free_segment(row, start_pos=0)
        self.assertEqual(len(owner._pending_until_retire), 1)
        self.assertEqual(len(owner._pending_until_retire.pages()), 2)

    def test_a_step_without_a_forward_is_never_a_key(self):
        """A prebuilt step records no event: pages freed while it is the newest
        step stay pending under the forward before it, and with no forward in
        flight they are holes at once (nothing to wait on)."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(2)]
        event = object()
        launch = _launch(steps, event)
        prebuilt = steps.forward_launched()  # no forward
        allocator.free_segment(rows[0], start_pos=0)
        self.assertEqual(owner._pending_until_retire._batches[0].step, 1)
        self.assertIs(owner._pending_until_retire._batches[0].event, event)
        steps.forward_processed(launch)
        self.assertEqual(_pending(owner), set())
        allocator.free_segment(rows[1], start_pos=0)  # only the prebuilt step
        self.assertEqual(_pending(owner), set())
        steps.forward_processed(prebuilt)

    def test_no_free_or_allocation_between_launch_and_the_event(self):
        allocator, steps = self.allocator, self.steps
        steps.forward_launched()
        allocator.alloc(8)
        with self.assertRaises(AssertionError):
            steps.forward_recorded(None)

    def _spy_waits(self):
        band = type(self.owner)
        return mock.patch.object(
            band,
            "_take_pending_pages",
            autospec=True,
            side_effect=band._take_pending_pages,
        )

    def test_a_take_uses_pending_pages_before_it_grows(self):
        """No holes, some pending pages, room to grow: the take uses the
        pending pages (after a stream wait), the order capacity counts them
        in, and the frontier stays."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(8)
        pending = set(_physical(owner, row).tolist())
        _launch(steps)
        allocator.free_segment(row, start_pos=0)
        frontier = owner.watermark_physical
        with self._spy_waits() as waited:
            taken = allocator.alloc(8)
            waited.assert_called()
        self.assertEqual(set(_physical(owner, taken).tolist()), pending)
        self.assertEqual(owner.watermark_physical, frontier)
        self.assertEqual(_pending(owner), set())

    def test_an_owner_take_leaves_the_shared_gap_its_neighbour_was_priced_on(self):
        """BUG REGRESSION. The owner, with pending pages, took growth from the
        gap it shares with the window instead of its pending pages, and the
        window then lacked the room the capacity check had counted for it."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        window = allocator.swa_attn_allocator
        ps = owner.page_size
        row = owner.alloc(8)
        room = window.available_size() // ps
        self.assertIsNotNone(window.take_physical_pages(room - 4))
        left = window.available_size() // ps
        _launch(steps)
        owner.free(row)
        self.assertIsNotNone(owner.alloc(8))
        self.assertEqual(window.available_size() // ps, left)
        self.assertIsNotNone(window.take_physical_pages(left))
        self.assertEqual(owner._byte_accounting_violations(), [])
        self.assertEqual(window._byte_accounting_violations(), [])

    def test_a_take_without_room_uses_pending_pages(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(8)
        physical = set(_physical(owner, row).tolist())
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        _launch(steps)
        allocator.free_segment(row, start_pos=0)
        with self._spy_waits() as waited:
            taken = allocator.alloc(8)
            waited.assert_called()  # the owner and the window
        self.assertEqual(set(_physical(owner, taken).tolist()), physical)
        self.assertEqual(_pending(owner), set())

    def test_part_of_a_pending_batch_then_processed(self):
        """A take splits a pending batch; processing returns only the rest: no page
        is handed out or freed twice."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(16)
        physical = set(_physical(owner, row).tolist())
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        launch = _launch(steps)
        allocator.free_segment(row, start_pos=0)
        first = set(_physical(owner, allocator.alloc(8)).tolist())
        self.assertEqual(len(first), 2)
        self.assertEqual(_pending(owner), physical - first)
        steps.forward_processed(launch)
        self.assertEqual(_pending(owner), set())
        free = owner._free_phys_pages.tolist()
        self.assertEqual(sorted(free), sorted(physical - first))
        second = set(_physical(owner, allocator.alloc(8)).tolist())
        self.assertEqual(first | second, physical)
        self.assertFalse(first & second)
        self.assertEqual(allocator.available_size(), 0)

    def test_a_claim_of_a_later_step_leaves_its_rest_to_its_own_processing(self):
        """Claim all of S and part of S + 1 (waiting S + 1), then process S:
        the untouched pages of S + 1 stay pending until S + 1 is processed,
        and every page comes back once."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(2)]
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        physical = [set(_physical(owner, row).tolist()) for row in rows]
        launch = _launch(steps)
        allocator.free_segment(rows[0], start_pos=0)
        launch2 = _launch(steps)
        allocator.free_segment(rows[1], start_pos=0)
        owner._take_pending_pages(3)
        rest = _pending(owner)
        self.assertEqual(len(rest), 1)
        self.assertTrue(rest <= physical[1])
        steps.forward_processed(launch)  # S
        self.assertEqual(_pending(owner), rest)
        steps.forward_processed(launch2)  # S + 1
        free = owner._free_phys_pages.tolist()
        self.assertEqual(sorted(free), sorted(physical[0] | physical[1]))

    def test_a_normalized_step_then_processed_returns_nothing_twice(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(3)]
        launch = _launch(steps)
        allocator.free_segment(rows[1], start_pos=0)
        owner.flush_for_allocation()  # normalizes, then compacts
        self.assertEqual(_pending(owner), set())
        steps.forward_processed(launch)
        free = owner._free_phys_pages.tolist()
        self.assertEqual(len(free), len(set(free)))
        self.assertEqual(owner._byte_accounting_violations(), [])

    def test_a_free_after_a_claim_joins_the_same_step_once(self):
        """S is ordered by a claim's wait but unprocessed: a later free still
        stays pending under S, and processing S returns it once."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(2)]
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        launch = _launch(steps)
        allocator.free_segment(rows[0], start_pos=0)
        claimed = set(_physical(owner, allocator.alloc(8)).tolist())
        allocator.free_segment(rows[1], start_pos=0)
        later = _pending(owner)
        self.assertEqual(len(later), 2)
        self.assertFalse(later & claimed)
        steps.forward_processed(launch)
        free = owner._free_phys_pages.tolist()
        self.assertEqual(sorted(free), sorted(later))

    def test_a_rejected_take_detaches_nothing(self):
        """The window could serve a request from its one pending page, but the
        owner has no room and that page cannot hold both: the request is
        rejected before any pending page is detached, nothing is lost, and a
        later request succeeds."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        window = allocator.swa_attn_allocator
        ids = allocator.alloc(allocator.available_size())
        launch = _launch(steps)
        allocator.free_window(window.sub_pool_name, ids[:4])
        pending = window._pending_until_retire
        available = (owner.available_size(), window.available_size())
        self.assertIsNone(allocator.alloc(4))
        self.assertIs(window._pending_until_retire, pending)
        self.assertEqual((owner.available_size(), window.available_size()), available)
        self.assertEqual(window._byte_accounting_violations(), [])
        allocator.free_segment(ids[16:24], start_pos=16)
        self.assertIsNotNone(allocator.alloc(8))
        steps.forward_processed(launch)

    def test_a_drain_renews_the_pricing_snapshot(self):
        """Processing moves pending pages to the free list: the snapshot is
        rebuilt, and the next take uses exactly those pages."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(8)
        physical = set(_physical(owner, row).tolist())
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        launch = _launch(steps)
        allocator.free_segment(row, start_pos=0)
        before = allocator.pricing_snapshot()
        steps.forward_processed(launch)
        self.assertIsNot(allocator.pricing_snapshot(), before)
        self.assertEqual(set(_physical(owner, allocator.alloc(8)).tolist()), physical)

    def test_a_partial_claim_renews_the_pricing_snapshot(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        row = allocator.alloc(16)
        physical = set(_physical(owner, row).tolist())
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        _launch(steps)
        allocator.free_segment(row, start_pos=0)
        before = allocator.pricing_snapshot()
        first = set(_physical(owner, allocator.alloc(8)).tolist())
        after = allocator.pricing_snapshot()
        self.assertIsNot(after, before)
        self.assertEqual(after.bands[after.index(owner.sub_pool_name)].holes, 2)
        second = set(_physical(owner, allocator.alloc(8)).tolist())
        self.assertEqual(second, physical - first)

    def test_a_free_adds_no_host_sync(self):
        allocator, steps = self.allocator, self.steps
        row = allocator.alloc(16)
        _launch(steps)
        calls = []
        for name in ("item", "tolist", "cpu"):
            real = getattr(torch.Tensor, name)
            patcher = mock.patch.object(
                torch.Tensor,
                name,
                lambda t, *a, _real=real, _n=name, **k: (
                    calls.append(_n) or _real(t, *a, **k)
                ),
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            allocator.free_segment(row, start_pos=0)
        self.assertEqual(calls, [])
        synchronize.assert_not_called()

    def test_non_urgent_compaction_moves_nothing_while_pages_are_pending(self):
        """A band with pending pages only absorbs its frontier holes when not
        urgent: no survivor moves (into the interior hole or anywhere), and
        the frontier stops at the pending page."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(4)]
        allocator.free_segment(rows[0], start_pos=0)  # an interior hole
        allocator.free_segment(rows[3], start_pos=0)  # a frontier hole
        frontier = set(_physical(owner, rows[3]).tolist())
        _launch(steps)
        allocator.free_segment(rows[2], start_pos=0)  # pending, then frontier
        pending = _pending(owner)
        before = owner.ids.virtual_to_physical[rows[1][::4] // 4].clone()
        self.assertEqual(owner.flush_opportunistic(), 0)
        self.assertEqual(_pending(owner), pending)
        self.assertTrue(
            torch.equal(owner.ids.virtual_to_physical[rows[1][::4] // 4], before)
        )
        free = set(owner._free_phys_pages.tolist())
        self.assertFalse(frontier & free)  # absorbed into the gap
        self.assertFalse(pending & free)
        self.assertEqual(owner._byte_accounting_violations(), [])

    def test_urgent_compaction_settles_a_pending_frontier_page_first(self):
        """A pending page at the frontier is never a move source; an urgent
        pass settles it (stream wait) and only then walks, with no host read of
        the free list after that wait."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(3)]
        allocator.free_segment(rows[0], start_pos=0)  # interior hole
        _launch(steps)
        allocator.free_segment(rows[2], start_pos=0)  # pending, at the frontier
        events = []
        band = type(owner)
        real_take, real_tolist = band._take_pending_pages, torch.Tensor.tolist
        with (
            mock.patch.object(
                band,
                "_take_pending_pages",
                autospec=True,
                side_effect=lambda *a, **k: events.append("wait") or real_take(*a, **k),
            ),
            mock.patch.object(
                torch.Tensor,
                "tolist",
                lambda t: events.append("read") or real_tolist(t),
            ),
        ):
            owner.flush_for_allocation()
        self.assertIn("wait", events)
        self.assertNotIn("read", events[events.index("wait") :])
        self.assertEqual(_pending(owner), set())
        live = owner.ids.virtual_to_physical[rows[1][::4] // 4]
        self.assertTrue(bool((live >= 0).all()))
        self.assertEqual(owner._byte_accounting_violations(), [])

    def test_a_pending_frontier_page_is_never_a_non_urgent_source(self):
        allocator, owner, steps = self.allocator, self.owner, self.steps
        rows = [allocator.alloc(8) for _ in range(3)]
        allocator.free_segment(rows[0], start_pos=0)
        _launch(steps)
        allocator.free_segment(rows[2], start_pos=0)
        before = owner.ids.virtual_to_physical[rows[1][::4] // 4].clone()
        self.assertEqual(owner.flush_opportunistic(), 0)
        self.assertTrue(
            torch.equal(owner.ids.virtual_to_physical[rows[1][::4] // 4], before)
        )
        self.assertEqual(len(_pending(owner)), 2)

    def test_relief_credits_a_neighbours_pending_pages_and_realizes_them(self):
        """The owner, short of room, is credited the window's pending pages;
        relief settles them in the window (stream wait) before the window's
        frontier retreats, and the owner then grows into the bytes."""
        allocator, owner, steps = self.allocator, self.owner, self.steps
        window = allocator.swa_attn_allocator
        ids = allocator.alloc(allocator.available_size())
        _launch(steps)
        allocator.free_window(window.sub_pool_name, ids[:16])  # window rows
        self.assertEqual(owner.available_size(), 0)
        self.assertGreaterEqual(owner.schedulable_available_size(), 8)
        self.assertIsNotNone(owner.alloc(8))
        self.assertEqual(window._pending_until_retire_pages, 0)
        self.assertEqual(window._byte_accounting_violations(), [])

    def test_page_choice_does_not_depend_on_whether_the_forward_finished(self):
        """Every rank picks the same pages, whatever its local event state."""
        taken = []
        for fired in (True, False):
            allocator = build_swa_pool(page_size=4, lazy=True)
            owner = allocator.full_attn_allocator
            event = mock.Mock(spec=["query"])
            event.query.side_effect = AssertionError("decisions never query")
            owner._latest_forward_done_event = event
            rows = [allocator.alloc(8) for _ in range(3)]
            self.assertIsNotNone(allocator.alloc(allocator.available_size()))
            _launch(allocator.forward_steps, event)
            allocator.free_segment(rows[1], start_pos=0)
            out = allocator.alloc(allocator.available_size())  # needs them
            taken.append(_physical(owner, out).tolist())
        self.assertEqual(taken[0], taken[1])


class TestFloatPendingPages(CustomTestCase):
    """A float middle (here the window of a full + window + state chain)."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        _, self.allocator = build_tri_pool(page_size=1)
        self.steps = self.allocator.forward_steps
        self.float = next(
            m for m in self.allocator.members if m.grow_direction == "float"
        )

    def _take(self, n):
        """``n`` owner token ids whose window rows live in the float."""
        ids = self.allocator.alloc(n).to(torch.int64)
        self.assertTrue(bool((self.float.ids.virtual_to_physical[ids] > 0).all()))
        return ids

    def test_an_all_pending_float_stays_opaque_until_processed(self):
        flt, steps = self.float, self.steps
        v = self._take(4)
        span = flt._span_pages()
        launch = _launch(steps)
        flt.free(v, _pages=v)
        self.assertEqual(flt._live_pages(), 0)
        self.assertFalse(flt._is_frontier_transparent())
        self.assertEqual(flt._span_pages(), span)  # not parked
        steps.forward_processed(launch)
        self.assertTrue(flt._is_frontier_transparent())  # parked now

    def test_normalizing_an_all_pending_float_parks_it_for_its_neighbours(self):
        flt, steps = self.float, self.steps
        v = self._take(4)
        _launch(steps)
        flt.free(v, _pages=v)
        name = flt.sub_pool_name
        before = self.allocator.pricing_snapshot()
        self.assertFalse(before.bands[before.index(name)].transparent)
        flt.compact_holes(retreat_side="high")  # normalizes first
        self.assertEqual(flt._pending_until_retire_pages, 0)
        self.assertTrue(flt._is_frontier_transparent())
        self.assertEqual(flt._span_pages(), 0)
        after = self.allocator.pricing_snapshot()
        self.assertTrue(after.bands[after.index(name)].transparent)
        self.assertEqual(flt._byte_accounting_violations(), [])

    def test_a_planned_take_served_by_pending_pages_does_not_grow(self):
        """BUG REGRESSION. A plan met a float's demand from its pending pages
        (no fresh range); the take read that as an unplanned take and grew the
        float into room the plan had given another band."""
        flt, steps = self.float, self.steps
        v = self._take(8)
        _launch(steps)
        flt.free(v[2:7], _pages=v[2:7])  # 5 interior pages, pending
        pending = _pending(flt)
        plan = no_relief_plan(
            self.allocator.pricing_snapshot(), ((flt.sub_pool_name, 5),)
        )
        self.assertIsNotNone(plan)
        take = plan.takes[flt.sub_pool_name]
        self.assertIsNone(take.fresh)
        span = (flt.low_wm_page, flt.high_wm_page)
        taken = flt.take_planned_physical_pages(take)
        self.assertEqual(set(taken.tolist()), pending)
        self.assertEqual((flt.low_wm_page, flt.high_wm_page), span)

    def test_a_pending_only_float_is_opaque_in_pricing(self):
        flt, steps = self.float, self.steps
        v = self._take(4)
        _launch(steps)
        flt.free(v, _pages=v)
        band = self.allocator.pricing_snapshot().band(flt.sub_pool_name)
        self.assertEqual((band.live, band.pending), (0, 4))
        self.assertFalse(band.transparent)

    def test_a_float_counts_its_neighbours_pending_pages(self):
        """An end band's pending pages next to a float are room the float can
        reach after the end's urgent flush, as holes are."""
        flt, steps = self.float, self.steps
        full = self.allocator.full_attn_allocator
        ids = self._take(8)
        before = flt._peer_drainable_hole_bytes()
        _launch(steps)
        full.free(ids[:4])
        self.assertEqual(full._pending_until_retire_pages, 4)
        self.assertEqual(
            flt._peer_drainable_hole_bytes() - before, 4 * full.entry_bytes_per_page
        )

    def test_a_take_of_holes_alone_leaves_the_float_clean(self):
        """A take that needs no pending page claims none, and the next flush
        reads nothing on the host."""
        flt, steps = self.float, self.steps
        v = self._take(8)
        _launch(steps)
        flt.free(v[2:6], _pages=v[2:6])
        flt._normalize_pending()
        flt.flush_opportunistic()
        self.assertIsNotNone(flt.take_physical_pages(2))
        reads = []
        real_tolist = torch.Tensor.tolist
        with mock.patch.object(
            torch.Tensor, "tolist", lambda t: reads.append(1) or real_tolist(t)
        ):
            flt.flush_opportunistic()
        self.assertEqual(reads, [])

    def test_relocation_settles_pending_pages_and_keeps_live_ones_bound(self):
        flt, steps = self.float, self.steps
        v = self._take(8)
        _launch(steps)
        flt.free(v[2:4], _pages=v[2:4])  # interior, pending
        live = torch.cat([v[:2], v[4:]])
        flt.compact_holes(retreat_side="high")
        self.assertEqual(flt._pending_until_retire_pages, 0)
        physical = flt.ids.virtual_to_physical[live]
        self.assertTrue(bool((physical > 0).all()))
        self.assertEqual(len(set(physical.tolist())), live.numel())
        self.assertEqual(flt._byte_accounting_violations(), [])


def _two_windows(*, state=True):
    """[state | window float | draft window float | full] (without state:
    [window up | draft window float | full down]), window pages of 4 tokens,
    the draft window binding every owner page."""
    setup_allocator_context()
    placement = FusedDraftPlacement.from_counts(
        counts={"draft_swa": [1]},
        regions={
            "draft_swa": DenseDraftRegion(
                lane_num=1, head_num=1, head_dim=24, store_dtype=torch.bfloat16
            )
        },
        new_bands={"draft_swa": DraftBand(kind="window", window=12)},
    )
    return build_unified_bundle(
        UnifiedGeometry(
            chain="tri" if state else "swa",
            full_layer_ids=(0,),
            swa_layer_ids=(1,),
            state_layer_ids=(2,) if state else (),
            page_size=4,
            window=16,
            context_len=32,
            full_tokens=96,
            swa_tokens=48,
            state_slots=8,
            lazy=True,
            fused_draft=placement,
        )
    ).token_to_kv_pool_allocator


class TestTwoFloatPlans(CustomTestCase):
    def _with_pending(self, draft):
        allocator = _two_windows()
        window_name = (
            "draft_swa" if draft else allocator.swa_attn_allocator.sub_pool_name
        )
        self.assertIsNotNone(allocator.alloc_state_slots(1))
        resident = allocator.alloc(28)
        self.assertIsNotNone(resident)
        launch = _launch(allocator.forward_steps)
        allocator.free_window(window_name, resident[4:24])  # 5 interior pages
        window = allocator.window_allocators[window_name]
        self.assertEqual(window._pending_until_retire_pages, 5)
        return allocator, window, launch

    def test_recovery_settles_a_float_only_pending_pages_keep(self):
        """BUG REGRESSION. With both of its neighbours' gaps short, recovery
        never reached the relocation that would settle a float whose pages
        were all pending, and refused a demand that fits once it parks."""
        allocator = _two_windows(state=False)
        draft = allocator.window_allocators["draft_swa"]
        rows = []
        while (row := allocator.alloc(4)) is not None:
            rows.append(row)
        ids = torch.cat(rows)
        _launch(allocator.forward_steps)
        allocator.free_window("draft_swa", ids)
        self.assertEqual(draft._live_pages(), 0)
        self.assertGreater(draft._pending_until_retire_pages, 0)
        self.assertIsNotNone(allocator.alloc(8))
        self.assertEqual(draft._pending_until_retire_pages, 0)

    def test_a_planned_float_take_with_no_fresh_range_never_grows(self):
        allocator, window, _ = self._with_pending(draft=False)
        span = (window.low_wm_page, window.high_wm_page)
        self.assertIsNone(window.take_physical_pages(6, planned=True))
        self.assertEqual((window.low_wm_page, window.high_wm_page), span)
        self.assertEqual(window._pending_until_retire_pages, 5)

    def test_a_scoped_take_of_pending_and_fresh_pages_debits_its_reservation(self):
        """BUG REGRESSION. Pricing counted a float's pending pages as live, so
        a scoped host load that turned them live debited only its fresh pages;
        the phantom rest then failed a later take of the same scope."""
        allocator, draft, _ = self._with_pending(draft=True)
        take_host, _ = allocator.storage.host_load_alloc("draft_swa")
        full = allocator.full_attn_allocator
        live = allocator.pricing_snapshot().band("draft_swa").live
        with allocator.scoped_physical_takes(8, 0, auxiliary_swa_tokens=24):
            self.assertIsNotNone(take_host(24))  # 5 pending pages and 1 fresh
            self.assertEqual(draft._pending_until_retire_pages, 0)
            self.assertEqual(
                allocator.pricing_snapshot().band("draft_swa").live - live, 6
            )
            self.assertIsNotNone(full.take_physical_pages(2))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (stream waits)")
class TestATakeOfPendingPagesWaitsForTheForward(CustomTestCase):
    def test_the_take_stream_waits_the_step_event(self):
        setup_allocator_context()
        allocator = build_swa_pool(page_size=4, lazy=True)
        owner = allocator.full_attn_allocator
        event = object()
        row = allocator.alloc(allocator.available_size())
        _launch(allocator.forward_steps, event)
        allocator.free_segment(row[:8], start_pos=0)
        stream = mock.Mock()
        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(owner, "device", "cuda"),  # the wait's device check
        ):
            owner._take_pending_pages()
        stream.wait_event.assert_called_once_with(event)


def _own_table_cache(*, window):
    """A full + window cache over the pool's own request table."""
    bundle, allocator = build_tri_pool(page_size=1)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=bundle.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            sliding_window_size=window,
            tree_slots=tree_slots_of(
                (ComponentType.FULL, ComponentType.SWA), window=window
            ),
        )
    )
    return allocator, cache, allocator.forward_steps


def _request(cache, tokens, *, committed, allocated):
    pool = cache.req_to_token_pool
    req = Req(
        rid=f"r{tokens[0]}",
        origin_input_text="",
        origin_input_ids=array("q", tokens[: committed - 1]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(pool, getattr(pool, "state", None), [req])
    req.output_ids = array("q", tokens[committed - 1 : committed])
    req.full_untruncated_fill_ids = array("q", tokens[:committed])
    req.extend_end = committed
    slots = cache.token_to_kv_pool_allocator.alloc(allocated)
    pool.write((req.kv.req_pool_idx, slice(0, allocated)), slots)
    req.kv.kv_committed_len = committed
    req.kv.kv_allocated_len = allocated
    req.last_node = cache.root_node_handle()
    req.kv.cache_protected_len = 0
    req.lock = None
    req.extra_key = None
    return req


class TestEarlierReviewScenarios(CustomTestCase):
    """Scenarios that broke a logical hold; with the logical free at once they
    hold by construction."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()

    def test_rehome_onto_ids_whose_window_free_is_pending(self):
        allocator, cache, steps = _own_table_cache(window=8)
        window = allocator.swa_attn_allocator
        tokens = list(range(7000, 7032))
        prefix = _request(cache, tokens[:16], committed=16, allocated=16)
        long = _request(cache, tokens, committed=32, allocated=32)
        prefix.finished_reason = FINISH_LENGTH(length=1)
        launch = _launch(steps)
        checkpoint_kv_cache(long, cache)
        release_kv_cache(prefix, cache, checkpoint=True)
        steps.forward_processed(launch)
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens[:16])))
        ).last_device_node
        ids = cache.path_device_indices(node).to(torch.int64)
        self.assertTrue(bool((window.ids.virtual_to_physical[ids[8:16]] > 0).all()))

    def test_state_slots_with_pending_frees(self):
        """Copy-on-write in matching, a tracking slot and the state-slot gate
        all find slots whose free is pending."""
        _, allocator, cache = build_mamba_cache(page_size=1, lazy=True)
        steps, state = allocator.forward_steps, allocator.state_slots
        tokens = list(range(100, 116))
        req = _request(cache, tokens, committed=16, allocated=16)
        req.kv.mamba_pool_idx = state.alloc(1)[0]
        req.finished_reason = FINISH_LENGTH(length=1)
        release_kv_cache(req, cache, checkpoint=True)
        taken = []
        while (slot := state.alloc(1)) is not None:
            taken.append(slot)
        _launch(steps)
        state.free(torch.cat(taken))

        budget = allocator.create_prefill_budget(cache)
        self.assertTrue(budget.state_slots_fit(1))
        candidate = Req(
            rid="candidate",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens)), cow_mamba=True, req=candidate
            )
        )
        self.assertEqual(match.device_prefix_len, len(tokens))
        self.assertIsNotNone(candidate.kv.mamba_pool_idx)
        storage = kv_storage_of(cache.req_to_token_pool, allocator)
        self.assertIsNotNone(storage.take_state_slots(None, 1))

    def test_an_early_window_release_then_window_eviction_stays_pending(self):
        allocator, cache = build_swa_cache(occupancy=0, lazy=True, window=8)
        steps, window = allocator.forward_steps, allocator.swa_attn_allocator
        tokens = list(range(4000, 4016))
        ids = allocator.alloc(16)
        key = RadixKey(array("q", tokens))
        cache.insert(InsertParams(key=key, value=ids))
        lock = cache.lock(
            cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        )
        physical = set(window.ids.virtual_to_physical[ids.to(torch.int64)].tolist())

        launch = _launch(steps)  # decode W reads the window
        cache.release_swa(lock)  # preparing W + 1
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=1000))
        evicted = physical - {p for p in physical if p <= 0}
        self.assertTrue(evicted <= _pending(window))
        self.assertFalse(evicted & set(window._free_phys_pages.tolist()))
        steps.forward_processed(launch)
        self.assertEqual(_pending(window), set())
        cache.unlock(lock)


if __name__ == "__main__":
    unittest.main()
