"""A unified pool gives a freed page back only after the forwards that were in
flight at the free have been processed.

Under the overlap scheduler, the forward launched before a result is processed
still reads the pages of a request that result ends, and writes its next slot.
A unified pool hands a freed page out again from the schedule stream (zeroing
it, or copying into it while compacting), which is not ordered after that
forward. So the pool holds every free made while a step is launched and
unprocessed, whoever makes it, and runs it once that step is processed; a
request's tree unlock waits in the same queue. Every capacity check counts held
pages as occupied. A check that fails while frees are held drains the queue
(waits for the forwards on the host, releases everything) and checks again
before any retraction, abort or out-of-memory error.
"""

import types
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.mem_cache.allocation import retire_window
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.alloc.release_queue import ReleaseQueue
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import storage_tree_slots
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    arena_of,
    build_mamba_cache,
    build_swa_pool,
    build_test_unified_pools,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _physical_pages(band, token_ids):
    pages = token_ids.to(torch.int64)[:: band.page_size] // band.page_size
    return band.ids.virtual_to_physical[pages].clone()


class TestFreesWaitForTheForwardsInFlight(CustomTestCase):
    """A free made while a step is in flight keeps its pages bound."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator = build_swa_pool(page_size=4, lazy=True)
        self.queue = self.allocator.release_queue
        self.owner = self.allocator.full_attn_allocator

    def test_the_chain_holds_frees(self):
        self.assertIsInstance(self.queue, ReleaseQueue)

    def test_a_held_row_is_not_handed_out_before_its_forward_is_processed(self):
        allocator, owner, queue = self.allocator, self.owner, self.queue
        row = allocator.alloc(16)
        held_physical = set(_physical_pages(owner, row).tolist())
        free_before = allocator.available_size()

        queue.forward_launched()
        allocator.free_segment(row, start_pos=0)

        self.assertEqual(allocator.available_size(), free_before)
        self.assertEqual(queue.held_tokens(owner), 16)
        # Still bound: the forward's page table names these physical pages.
        self.assertEqual(set(_physical_pages(owner, row).tolist()), held_physical)
        # Fill the pool, compacting under pressure: no take reaches a held page.
        taken = allocator.alloc(free_before)
        self.assertIsNotNone(taken)
        self.assertFalse(held_physical & set(_physical_pages(owner, taken).tolist()))
        owner.flush_for_allocation()
        self.assertEqual(
            int((owner.ids.virtual_to_physical[row[::4] // 4] < 0).sum()), 0
        )

        queue.forward_retired()
        self.assertEqual(len(queue), 0)
        self.assertEqual(allocator.available_size(), 16)
        self.assertTrue(bool((owner.ids.virtual_to_physical[row[::4] // 4] < 0).all()))

    def test_a_free_waits_for_every_step_launched_before_it(self):
        allocator, queue = self.allocator, self.queue
        row = allocator.alloc(8)
        free_before = allocator.available_size()
        queue.forward_launched()
        queue.forward_launched()
        allocator.free_segment(row, start_pos=0)
        queue.forward_retired()
        self.assertEqual(allocator.available_size(), free_before)
        queue.forward_retired()
        self.assertEqual(allocator.available_size(), free_before + 8)

    def test_a_free_with_no_step_in_flight_runs_at_once(self):
        allocator, queue = self.allocator, self.queue
        row = allocator.alloc(8)
        free_before = allocator.available_size()
        queue.forward_launched()
        queue.forward_retired()
        allocator.free_segment(row, start_pos=0)
        self.assertEqual(allocator.available_size(), free_before + 8)

    def test_a_quiet_free_runs_at_once(self):
        allocator, queue = self.allocator, self.queue
        row = allocator.alloc(8)
        free_before = allocator.available_size()
        queue.forward_launched()
        with queue.quiet_frees():
            allocator.free_segment(row, start_pos=0)
        self.assertEqual(allocator.available_size(), free_before + 8)

    def test_any_free_path_waits_with_no_handling_of_its_own(self):
        """A free path nobody taught about forwards (here the plain token
        free, inside a free group as result processing makes them) is held
        all the same."""
        allocator, queue = self.allocator, self.queue
        row = allocator.alloc(8)
        free_before = allocator.available_size()
        queue.forward_launched()
        allocator.free_group_begin()
        allocator.free(row)
        allocator.free_group_end()
        self.assertEqual(allocator.available_size(), free_before)
        queue.forward_retired()
        self.assertEqual(allocator.available_size(), free_before + 8)


def _swa_cache(*, sessions=False, page_size=1, occupancy=4, window=128, max_num_reqs=4):
    """A full + window unified pool and its prefix tree over the pool's own
    request table, as the scheduler builds them, with ``occupancy`` cached
    single-page entries."""
    allocator = build_test_unified_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
        total_bytes=100 * page_size * 2 * 32,
        model_context_len=40 * page_size,
        max_num_reqs=max_num_reqs,
    ).token_to_kv_pool_allocator
    storage = allocator.storage
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            enable_session_radix_cache=sessions,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=window,
            tree_slots=storage_tree_slots(storage, window=window),
        )
    )
    for i in range(occupancy):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(i * 1000, i * 1000 + page_size))),
                value=allocator.alloc(page_size),
            )
        )
    return allocator, cache, allocator.release_queue
    return allocator, cache, allocator.release_queue


def _request(cache, tokens, *, committed, allocated, session=None):
    """A request whose row holds ``allocated`` token slots, ``committed`` of
    them computed (the rest over-allocated, as speculative decoding leaves)."""
    pool = cache.req_to_token_pool
    req = Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", tokens[: committed - 1]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        session=session,
    )
    assert cache.token_to_kv_pool_allocator.storage.acquire([req]) is not None
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


class TestReleaseAfterTheForward(CustomTestCase):
    """`release_kv_cache` and the prefix tree need nothing of their own."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator, self.cache, self.queue = _swa_cache()

    def _match(self, tokens):
        key = RadixKey(array("q", tokens))
        return self.cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len

    def test_insert_now_free_and_unlock_after_the_forward(self):
        tokens = list(range(500, 512))
        # The tree already holds the first four tokens: the insert frees the
        # request's own copies, which the running forward still reads.
        shared = self.allocator.alloc(4)
        self.cache.insert(
            InsertParams(key=RadixKey(array("q", tokens[:4])), value=shared)
        )
        req = _request(self.cache, tokens, committed=12, allocated=12)
        req.finished_reason = FINISH_LENGTH(length=1)
        free_before = self.allocator.available_size()
        evictable_before = self.cache.evictable_size()

        self.queue.forward_launched()
        release_kv_cache(req, self.cache, checkpoint=True)

        self.assertEqual(self._match(tokens), 12)  # inserted at once
        self.assertEqual(self.allocator.available_size(), free_before)
        # Still locked, the shared prefix included: an eviction cannot free
        # pages the forward reads.
        self.assertEqual(self.cache.evictable_size(), evictable_before - 4)
        self.cache.evict(EvictParams(num_tokens=1000))
        self.assertEqual(self._match(tokens), 12)

        self.queue.forward_retired()
        self.assertGreaterEqual(self.allocator.available_size(), free_before + 4)
        self.cache.sanity_check()

    def test_an_eviction_frees_at_once(self):
        """Unlocked nodes hold no page a forward in flight reads: an eviction
        while one runs gives its pages back at once."""
        free_before = self.allocator.available_size()
        self.queue.forward_launched()
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertGreater(self.allocator.available_size(), free_before)

    def test_a_running_request_checkpoint_holds_its_duplicate_pages(self):
        tokens = list(range(900, 908))
        shared = self.allocator.alloc(8)
        self.cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=shared))
        req = _request(self.cache, tokens, committed=8, allocated=8)
        free_before = self.allocator.available_size()

        self.queue.forward_launched()
        checkpoint_kv_cache(req, self.cache)

        self.assertEqual(self.allocator.available_size(), free_before)
        self.queue.forward_retired()
        self.assertEqual(self.allocator.available_size(), free_before + 8)


class TestStreamingSessionFrees(CustomTestCase):
    """A streaming session frees a turn's over-allocated tail when the turn
    finishes, and the duplicate pages of its prompt when it checkpoints: both
    while a forward that includes the turn may still run."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator, self.cache, self.queue = _swa_cache(sessions=True)
        self.session = SimpleNamespace(
            session_id="s",
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
        )

    def test_a_finished_turn_trims_its_overshoot_after_the_forward(self):
        tokens = list(range(300, 316))
        # Speculative decoding committed and allocated past the finished length.
        req = _request(
            self.cache, tokens, committed=12, allocated=16, session=self.session
        )
        self.cache.maybe_hand_to_session(req)
        req.finished_reason = FINISH_LENGTH(length=1)
        req.finished_len = 1
        free_before = self.allocator.available_size()

        self.queue.forward_launched()
        self.assertTrue(self.cache.session.try_cache_finished_req(req))

        self.assertEqual(self.allocator.available_size(), free_before)
        self.assertGreater(len(self.queue), 0)
        self.queue.forward_retired()
        self.assertGreater(self.allocator.available_size(), free_before)

    def test_a_first_turn_checkpoint_holds_its_duplicate_pages(self):
        tokens = list(range(400, 408))
        shared = self.allocator.alloc(8)
        self.cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=shared))
        req = _request(
            self.cache, tokens, committed=8, allocated=8, session=self.session
        )
        self.cache.maybe_hand_to_session(req)
        free_before = self.allocator.available_size()

        self.queue.forward_launched()
        self.assertTrue(self.cache.session.try_checkpoint(req, up_to=8))

        self.assertEqual(self.allocator.available_size(), free_before)
        self.queue.forward_retired()
        self.assertEqual(self.allocator.available_size(), free_before + 8)


class _ForwardStream:
    """Stands in for the forward stream: counts the host waits `drain` makes."""

    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


class TestDrainOnShortage(CustomTestCase):
    """A check that fails while frees are held waits for the forwards on the
    host and releases what is held; frees made after it run at once until the
    next launch."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator = build_swa_pool(page_size=4, lazy=True)
        self.queue = self.allocator.release_queue
        self.forward = _ForwardStream()
        self.queue._forward_stream = self.forward

    def test_drain_waits_for_the_forwards_and_releases_what_is_held(self):
        allocator, queue = self.allocator, self.queue
        row, later = allocator.alloc(8), allocator.alloc(8)
        free_before = allocator.available_size()
        queue.forward_launched()
        allocator.free_segment(row, start_pos=0)
        self.assertEqual(allocator.available_size(), free_before)

        self.assertTrue(queue.drain())
        self.assertEqual(self.forward.waits, 1)
        self.assertEqual(len(queue), 0)
        self.assertEqual(allocator.available_size(), free_before + 8)
        # The forward is done on the device: a free before its result is
        # processed runs at once, until the next launch.
        allocator.free_segment(later, start_pos=0)
        self.assertEqual(allocator.available_size(), free_before + 16)
        queue.forward_launched()
        self.assertTrue(queue.in_flight())
        queue.forward_retired()
        queue.forward_retired()
        self.assertFalse(queue.in_flight())

    def test_nothing_held_nothing_to_wait_for(self):
        self.queue.forward_launched()
        self.assertFalse(self.queue.drain())
        self.assertEqual(self.forward.waits, 0)


def _decode_batch(allocator, cache, reqs, tokens_per_req):
    """What `ScheduleBatch.check_decode_mem` reads of a decode batch."""
    batch = SimpleNamespace(
        reqs=reqs,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        spec_algorithm=None,
        new_tokens_required_next_decode=lambda selected=None: (
            tokens_per_req * (len(reqs) if selected is None else len(selected))
        ),
    )
    batch.check_decode_mem = types.MethodType(ScheduleBatch.check_decode_mem, batch)
    return batch


class _Retracted(Exception):
    pass


def _decode_scheduler():
    """What `Scheduler.update_running_batch` reads of the scheduler; reaching
    retraction raises `_Retracted`."""
    s = Scheduler.__new__(Scheduler)
    s.enable_skip_finishing_decode = False
    s.uocr_runtime = None
    s.decode_offload_manager = None
    s.forward_ct = 1
    s.new_token_ratio_tracker = SimpleNamespace(decay_step=lambda: None)
    s.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=mock.Mock(side_effect=_Retracted)
    )
    return s


def _running(batch):
    """Give a check-only decode batch what `update_running_batch` touches."""
    batch.batch_size = lambda: len(batch.reqs)
    batch.filter_batch = lambda **_: None
    batch.is_empty = lambda: not batch.reqs
    batch.batch_is_full = True
    batch.prepared = False

    def prepare_for_decode():
        batch.prepared = True

    batch.prepare_for_decode = prepare_for_decode
    return batch


class TestDecodeShortageDrains(CustomTestCase):
    """Decode short of memory because frees are held drains them and decodes
    now; it neither waits a step nor retracts."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()

    def test_decode_returns_its_batch_when_held_frees_cover_the_shortfall(self):
        """PD decode reads `update_running_batch`'s result as a batch, never
        None."""
        allocator, cache, queue = _swa_cache(occupancy=1)
        cache.evict(EvictParams(num_tokens=1000))
        rest = allocator.alloc(allocator.available_size())
        queue.forward_launched()
        allocator.free_segment(rest[:8], start_pos=0)
        self.assertEqual(allocator.available_size(), 0)

        batch = _running(_decode_batch(allocator, cache, [object()], 4))
        self.assertIs(Scheduler.update_running_batch(_decode_scheduler(), batch), batch)
        self.assertTrue(batch.prepared)
        self.assertEqual(len(queue), 0)

    def test_an_allocator_override_gets_only_the_decode_check_keywords(self):
        """DeepSeek V4 on NPU overrides `check_decode_capacity`; a shortage
        reaches retraction through it instead of a TypeError."""
        try:
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
                DSV4NPUTokenToKVPoolAllocator,
            )
        except ImportError as e:
            self.skipTest(f"NPU allocator unavailable: {e}")
        allocator = mock.create_autospec(DSV4NPUTokenToKVPoolAllocator, instance=True)
        allocator.check_decode_capacity.return_value = False
        batch = _running(_decode_batch(allocator, None, [object()], 1))
        with self.assertRaises(_Retracted):
            Scheduler.update_running_batch(_decode_scheduler(), batch)

    def test_a_cancelled_state_reservation_does_not_abort_the_last_request(self):
        """Unused state slots go back while a decode runs (held); the next
        decode check drains them instead of retracting or aborting."""
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="mamba",
                full_layer_ids=(0,),
                state_layer_ids=(1,),
                conv_shapes=((4, 8),),
                temporal_shape=(1, 6, 8),
                lazy=True,
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        queue, state = allocator.release_queue, allocator.state_slots
        before = allocator.available_size()
        state.alloc_group_begin(1)
        # One state slot takes eight token-equivalents of the shared arena.
        self.assertEqual(before - allocator.available_size(), 8)
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        self.assertEqual(allocator.available_size(), 0)

        queue.forward_launched()
        state.alloc_group_end()  # admission did not use the reserved slot
        self.assertEqual(allocator.available_size(), 0)

        batch = _decode_batch(allocator, None, [object()], 1)
        self.assertTrue(batch.check_decode_mem())
        self.assertEqual(len(queue), 0)

    def test_one_retraction_when_its_prefix_unlock_frees_enough(self):
        """Four requests on 1024-token cached prefixes: retracting one gives
        back its own page now (held) and its prefix once unlocked (deferred);
        the drain runs both, so decode evicts that prefix instead of
        retracting a second request."""
        page_size, prefix_len = 64, 1024
        allocator, cache, queue = _swa_cache(occupancy=1, page_size=page_size)
        cache.evict(EvictParams(num_tokens=10**6))
        reqs = []
        for i in range(4):
            prefix = list(range(100000 * (i + 1), 100000 * (i + 1) + prefix_len))
            ids = allocator.alloc(prefix_len)
            cache.insert(InsertParams(key=RadixKey(array("q", prefix)), value=ids))
            reqs.append(_request_on_prefix(cache, prefix, own=page_size))
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        self.assertEqual(cache.evictable_size(), 0)
        queue.forward_launched()

        retracted = []

        def release_req(idx, remaining, retirer, offload_kv=True):
            release_kv_cache(reqs[idx], cache, checkpoint=False)
            retracted.append(reqs[idx])
            return True

        batch = _decode_batch(allocator, cache, reqs, page_size)
        batch._get_decode_retraction_order = ScheduleBatch._get_decode_retraction_order
        batch.release_req = release_req
        batch.filter_batch = lambda **_: None
        _, _, aborted = ScheduleBatch.retract_decode(batch, None)
        self.assertEqual(len(retracted), 1)
        self.assertEqual(aborted, [])


def _request_on_prefix(cache, prefix, *, own):
    """A decoding request on the cached ``prefix`` (locked) plus ``own`` new
    token slots."""
    pool = cache.req_to_token_pool
    key = RadixKey(array("q", prefix))
    match = cache.match_prefix(MatchPrefixParams(key=key))
    assert match.device_prefix_len == len(prefix)
    req = Req(
        rid=f"p{prefix[0]}",
        origin_input_text="",
        origin_input_ids=array("q", prefix),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=64),
    )
    assert cache.token_to_kv_pool_allocator.storage.acquire([req]) is not None
    slots = cache.token_to_kv_pool_allocator.alloc(own)
    pool.write(
        (req.kv.req_pool_idx, slice(0, len(prefix) + own)),
        torch.cat(
            [cache.path_device_indices(match.last_device_node).to(slots.dtype), slots]
        ),
    )
    req.output_ids = array("q", [7] * own)
    req.kv.kv_committed_len = req.kv.kv_allocated_len = len(prefix) + own
    req.kv.cache_protected_len = len(prefix)
    req.last_node = match.last_device_node
    req.lock = cache.lock(match.last_device_node)
    req.extra_key = None
    return req


class TestWindowRetirementWaitsLikeAnyFree(CustomTestCase):
    """A chunk's checkpoint retires window rows up to the end of the chunk
    still running, whose earlier queries read them: they are held like any
    free."""

    def test_window_rows_retired_under_a_running_chunk_are_held(self):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(window=8, occupancy=0)
        window = allocator.swa_attn_allocator
        # A 32-token chunk: its checkpoint frontier is its last token, while
        # its earlier queries read the rows below.
        tokens = list(range(5000, 5032))
        req = _request(cache, tokens, committed=len(tokens), allocated=len(tokens))
        free_before = window.available_size()

        queue.forward_launched()
        retire_window(req, len(tokens) - 1, cache)

        held = queue.held_tokens(window)
        self.assertGreater(held, 0)
        self.assertEqual(window.available_size(), free_before)
        queue.forward_retired()
        self.assertEqual(window.available_size(), free_before + held)


class TestHeldWindowRowsAreReleased(CustomTestCase):
    """A window row whose retirement is held is released already: a later free
    that picks a row's live window rows by their bindings skips it."""

    def test_a_token_id_free_after_window_retirement_frees_each_row_once(self):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(window=8, occupancy=0)
        window = allocator.swa_attn_allocator
        tokens = list(range(900, 932))
        req = _request(cache, tokens, committed=32, allocated=32)
        row = cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :32].clone()
        window_before = window.available_size()

        queue.forward_launched()
        retire_window(req, len(tokens) - 1, cache)
        retired = queue.held_tokens(window)
        self.assertGreater(retired, 0)
        allocator.free(row)  # every owner id of the row, by token id
        # Each window row goes back once: the retired ones are not freed again.
        self.assertEqual(queue.held_tokens(window), 32)
        queue.forward_retired()
        free = window._free_phys_pages
        self.assertEqual(int(torch.unique(free).numel()), int(free.numel()))
        self.assertGreater(window.available_size(), window_before)


class TestRebindAfterAHeldFree(CustomTestCase):
    """A long running prompt and its shorter prefix in one prefill batch: the
    long one's checkpoint retires its early window rows (held), and the prefix
    recovers its window onto those ids. The held free must not release the
    recovered rows when it runs."""

    def _run(self, launched):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(window=8, occupancy=0)
        window = allocator.swa_attn_allocator
        long_tokens = list(range(7000, 7032))
        prefix = _request(cache, long_tokens[:16], committed=16, allocated=16)
        long = _request(cache, long_tokens, committed=32, allocated=32)
        prefix.finished_reason = FINISH_LENGTH(length=1)
        for _ in range(launched):  # 2: overlap, a later step already launched
            queue.forward_launched()
        checkpoint_kv_cache(long, cache)  # the long prompt keeps running
        release_kv_cache(prefix, cache, checkpoint=True)
        for _ in range(launched):
            queue.forward_retired()

        key = RadixKey(array("q", long_tokens[:16]))
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        ids = cache.path_device_indices(node).to(torch.int64)
        # The prefix's window (its last 8 tokens) is cached under the long
        # prompt's ids, bound to rows nobody freed.
        physical = window.ids.virtual_to_physical[ids[8:16]]
        self.assertTrue(bool((physical > 0).all()), physical.tolist())
        self.assertFalse(
            bool(torch.isin(physical, window._free_phys_pages).any()),
            "a recovered window row went back to the free list",
        )

    def test_non_overlap(self):
        self._run(launched=1)

    def test_overlap(self):
        self._run(launched=2)

    def test_binding_a_held_id_without_draining_fails_checks(self):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(window=8, occupancy=0)
        window = allocator.swa_attn_allocator
        owner = allocator.alloc(8)
        pages = owner.to(torch.int64)
        queue.forward_launched()
        allocator.free_window(window.sub_pool_name, owner)  # held
        spare = allocator.alloc(8)
        physical = window.ids.virtual_to_physical[spare.to(torch.int64)].clone()
        with envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.WARN)):
            with self.assertRaises(AssertionError):
                window.ids.bind(pages, physical)
            # The storage drains first.
            storage = kv_storage_of(cache.req_to_token_pool, allocator)
            storage.bind_window(window.sub_pool_name, owner, spare)
        self.assertEqual(len(queue), 0)


class TestStateSlotsHeldAtMatch(CustomTestCase):
    """Matching a cached prefix copies its recurrent state into a new slot
    (copy-on-write). With every free slot held for the forward in flight, the
    take drains them instead of failing."""

    def test_copy_on_write_while_state_slots_are_held(self):
        setup_allocator_context()
        _, allocator, cache = build_mamba_cache(page_size=1, lazy=True)
        queue, state = allocator.release_queue, allocator.state_slots
        tokens = list(range(100, 116))
        req = _request(cache, tokens, committed=16, allocated=16)
        req.kv.mamba_pool_idx = state.alloc(1)[0]
        req.finished_reason = FINISH_LENGTH(length=1)
        release_kv_cache(req, cache, checkpoint=True)
        taken = []
        while (slot := state.alloc(1)) is not None:
            taken.append(slot)
        queue.forward_launched()
        state.free(torch.cat(taken))  # held

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
        self.assertEqual(len(queue), 0)


class TestRetireReopensAdmission(CustomTestCase):
    """Admission that a held free rejected opens again when the free is
    released, as when a request finishes."""

    def test_a_state_slot_rejection_reopens_at_retire(self):
        setup_allocator_context()
        _, allocator, cache = build_mamba_cache(page_size=1, lazy=True)
        queue, state = allocator.release_queue, allocator.state_slots
        taken = []
        while (slot := state.alloc(1)) is not None:
            taken.append(slot)
        queue.forward_launched()
        state.free(torch.cat(taken))  # held
        self.assertFalse(allocator.create_prefill_budget(cache).state_slots_fit(1))

        scheduler = _processing_scheduler(allocator)
        Scheduler._on_step_processed(scheduler, mock.Mock())

        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertTrue(allocator.create_prefill_budget(cache).state_slots_fit(1))

    def test_a_decode_drain_reopens_admission_at_the_next_step(self):
        """A state-slot rejection closed admission; the same pass's decode
        drains the held frees; no request finishes before the next step."""
        setup_allocator_context()
        _, allocator, cache = build_mamba_cache(page_size=1, lazy=True)
        queue, state = allocator.release_queue, allocator.state_slots
        taken = []
        while (slot := state.alloc(1)) is not None:
            taken.append(slot)
        row = allocator.alloc(8)
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        queue.forward_launched()
        state.free(torch.cat(taken))  # held
        allocator.free(row)  # held
        scheduler = _processing_scheduler(allocator)  # admission closed

        decode = _decode_batch(allocator, None, [object()], 1)
        self.assertTrue(decode.check_decode_mem())  # drains
        self.assertEqual(len(queue), 0)
        Scheduler._on_step_processed(scheduler, mock.Mock())

        self.assertFalse(scheduler.running_batch.batch_is_full)


def _processing_scheduler(allocator):
    """What `Scheduler._on_step_processed` reads, with admission closed."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.running_batch = SimpleNamespace(batch_is_full=True)
    scheduler.release_generation_seen = 0
    scheduler.invariant_checker = mock.Mock()
    scheduler.running_reqs = mock.Mock()
    scheduler.prefill_launched_reqs = mock.Mock()
    return scheduler


class TestEarlyWindowReleaseWaitsForTheForward(CustomTestCase):
    """Decode W reads the last prompt page within its window while W + 1 is
    prepared and releases the prefix's window lock early. That release waits
    for W like the request's other releases: the window rows stay out of
    eviction, and so are neither handed out nor zeroed, until W retires."""

    def test_an_early_window_release_is_evictable_only_after_retire(self):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(window=8, occupancy=0)
        window = allocator.swa_attn_allocator
        tokens = list(range(4000, 4016))
        ids = allocator.alloc(16)
        key = RadixKey(array("q", tokens))
        cache.insert(InsertParams(key=key, value=ids))
        lock = cache.lock(
            cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        )
        # The lock holds the window's last 8 rows, which W still reads.
        pages = ids.to(torch.int64)[8:]
        physical = window.ids.virtual_to_physical[pages].clone()

        queue.forward_launched()  # decode W
        cache.release_swa(lock)  # preparing W + 1
        cache.evict(EvictParams(window_num_tokens={window.sub_pool_name: 1000}))
        # Still bound to the same storage: not freed, so not handed out or
        # zeroed.
        self.assertTrue(bool((window.ids.virtual_to_physical[pages] == physical).all()))

        queue.forward_retired()
        cache.evict(EvictParams(window_num_tokens={window.sub_pool_name: 1000}))
        self.assertTrue(bool((window.ids.virtual_to_physical[pages] < 0).all()))
        cache.unlock(lock)


class TestWindowReleaseUnderStrictChecks(CustomTestCase):
    """A request whose window retired part of its rows releases its live window
    rows, then its owner rows; with all of them held for the forward in
    flight, the owner release still passes the strict invariant that every
    window row of those ids was released."""

    def test_release_after_window_retirement_passes_strict_checks(self):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(window=8, occupancy=0)
        tokens = list(range(600, 632))
        req = _request(cache, tokens, committed=32, allocated=32)
        req.finished_reason = FINISH_LENGTH(length=1)
        free_before = allocator.available_size()

        queue.forward_launched()
        with envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.STRICT)):
            retire_window(req, len(tokens) - 1, cache)
            release_kv_cache(req, cache, checkpoint=False)
            self.assertEqual(allocator.available_size(), free_before)
            queue.forward_retired()
        self.assertEqual(len(queue), 0)
        self.assertEqual(allocator.available_size(), free_before + 32)


class TestHeldAccountingIsExact(CustomTestCase):
    """Held pages are counted from their ids, as the band frees them, so the
    busy memory check that adds them to what requests own stays exact."""

    def test_a_rehomed_owner_row_counts_in_the_busy_check(self):
        setup_allocator_context()
        allocator, cache, queue = _swa_cache(page_size=4)
        storage = kv_storage_of(cache.req_to_token_pool, allocator)
        window = allocator.swa_attn_allocator.sub_pool_name
        owner = allocator.full_attn_allocator
        kept, incoming = allocator.alloc(16), allocator.alloc(16)
        owner_before = owner.available_size()

        queue.forward_launched()
        # RecoverSWAWithLockedFull: the kept ids take the incoming window rows,
        # and the incoming owner rows go back by token id.
        storage.rehome_window(window, kept, incoming)
        storage.release_rehomed_owner_rows(incoming)

        self.assertEqual(queue.held_tokens(owner), 16)
        checker = SimpleNamespace(
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            _requests_owning_rows=lambda: [],
        )
        full, _ = SchedulerInvariantChecker._get_total_uncached_sizes(
            checker, has_window=True
        )
        self.assertEqual(full, 16)
        queue.forward_retired()
        self.assertEqual(queue.held_tokens(owner), 0)
        self.assertEqual(owner.available_size(), owner_before + 16)


class TestAdmissionCountsHeldPagesAsOccupied(CustomTestCase):
    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator, self.cache, self.queue = _swa_cache()

    @staticmethod
    def _request_fits(budget, total_tokens):
        return budget.check_prefill(
            extend_input_len=8,
            total_tokens=total_tokens,
            max_new_tokens=8,
            input_tokens=8,
            chunk_limit=None,
        )[0]

    def test_a_request_that_fits_now_is_admitted_while_frees_are_held(self):
        """Held pages more than the request's demand never make it negative."""
        allocator, cache, queue = self.allocator, self.cache, self.queue
        row = allocator.alloc(32)
        self.assertTrue(self._request_fits(allocator.create_prefill_budget(cache), 20))
        queue.forward_launched()
        allocator.free_segment(row, start_pos=0)
        self.assertTrue(self._request_fits(allocator.create_prefill_budget(cache), 20))
        self.assertGreater(len(queue), 0)  # it fit without them

    def test_a_request_that_needs_held_pages_drains_them(self):
        allocator, cache, queue = self.allocator, self.cache, self.queue
        cache.evict(EvictParams(num_tokens=1000))
        row = allocator.alloc(32)
        self.assertIsNotNone(allocator.alloc(allocator.available_size()))
        queue.forward_launched()
        allocator.free_segment(row, start_pos=0)
        self.assertTrue(self._request_fits(allocator.create_prefill_budget(cache), 20))
        self.assertEqual(len(queue), 0)


class TestAdmissionPinsReleaseAtOnce(CustomTestCase):
    """A request's tree unlock waits for its forwards like its frees; an
    admission's temporary pin on a candidate in no launched forward does not."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator, self.cache, self.queue = _swa_cache()
        self.tokens = list(range(800, 816))
        ids = self.allocator.alloc(16)
        key = RadixKey(array("q", self.tokens))
        self.cache.insert(InsertParams(key=key, value=ids))
        self.node = self.cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        self.candidate = Req(
            rid="candidate",
            origin_input_text="",
            origin_input_ids=array("q", self.tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )

    def test_a_rejected_candidates_pin_leaves_its_prefix_evictable(self):
        cache = self.cache
        evictable = cache.evictable_size()
        self.queue.forward_launched()  # an unrelated decode is in flight
        adder = SimpleNamespace(tree_cache=cache)
        with PrefillAdder._lock_node(adder, self.node, self.candidate):
            self.assertEqual(cache.evictable_size(), evictable - 16)
        # Admission rejected it: the pin is gone, and decode can evict the
        # prefix in the same iteration.
        self.assertEqual(cache.evictable_size(), evictable)
        free_before = self.allocator.available_size()
        cache.evict(EvictParams(num_tokens=1000))
        self.assertGreaterEqual(self.allocator.available_size(), free_before + 16)

    def test_a_running_requests_unlock_waits_for_its_forward(self):
        cache = self.cache
        evictable = cache.evictable_size()
        lock = cache.lock(self.node)
        self.queue.forward_launched()
        cache.unlock(lock)
        self.assertEqual(cache.evictable_size(), evictable - 16)
        self.queue.forward_retired()
        self.assertEqual(cache.evictable_size(), evictable)

    def test_an_unlock_bypassing_the_queue_for_a_running_request_warns(self):
        lock = self.cache.lock(self.node)
        self.candidate.inflight_steps = 1
        self.queue.forward_launched()
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
        ) as logs:
            self.cache.unlock_pin(lock, self.candidate)
        self.assertIn("has not retired", "\n".join(logs.output))

    def test_normal_unlocks_do_not_warn(self):
        self.queue.forward_launched()
        with self.assertNoLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
        ):
            self.cache.unlock_pin(self.cache.lock(self.node), self.candidate)
            self.cache.unlock(self.cache.lock(self.node))
            self.queue.forward_retired()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (zeroing kernel)")
class TestHeldPageBytesSurviveHandOut(CustomTestCase):
    """Zeroing on hand-out never reaches a page a forward in flight reads."""

    def test_held_page_is_not_zeroed_by_a_take(self):
        page_size = 4
        spec = MHASubPoolSpec(
            name="full",
            layer_num=1,
            grow_direction="up",
            head_num=1,
            head_dim=8,
            store_dtype=torch.bfloat16,
        )
        state_spec = MambaSubPoolSpec(
            name="mamba",
            layer_num=1,
            grow_direction="down",
            conv_state_shapes=((8, 3),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(2, 4, 4),
            temporal_dtype=torch.float32,
        )
        arena = arena_of(
            total_bytes=64 * page_size * spec.entry_bytes(),
            sub_pool_specs=[spec, state_spec],
            device="cuda",
            enable_memory_saver=False,
            page_size=page_size,
        )
        view = MHAView(
            arena=arena,
            spec=arena.spec("full"),
            page_size=page_size,
            model_layer_ids=(0,),
        )
        band = Band(
            arena=arena,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            page_size=page_size,
            lazy_compaction=True,
        )
        queue = ReleaseQueue((band,))
        pages = arena.raw[: view.num_pages * view.page_bytes].view(
            view.num_pages, view.page_bytes
        )
        row = band.alloc(4 * page_size)
        held = _physical_pages(band, row)
        pages[held] = 7
        queue.forward_launched()
        band.free(row, _pages=row[::page_size] // page_size)

        taken = band.alloc(band.available_size())
        torch.cuda.synchronize()
        self.assertFalse(
            set(held.tolist()) & set(_physical_pages(band, taken).tolist())
        )
        self.assertTrue(bool((pages[held] == 7).all()))

        queue.forward_retired()
        band.free(taken)
        again = band.alloc(band.available_size())
        torch.cuda.synchronize()
        reused = _physical_pages(band, again)
        self.assertTrue(set(held.tolist()) <= set(reused.tolist()))
        self.assertTrue(bool((pages[held] == 0).all()))


if __name__ == "__main__":
    unittest.main()
