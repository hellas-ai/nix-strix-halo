"""CPU admission regression using actual runtime methods; no torch/GPU imports.

Run with --source /runtime/site-packages/sglang; optionally use --scheduler
for a patched scheduler.py. --expect-bug reproduces the unpatched stall.
Only tensor/cache/forward work is replaced with CPU fixtures. Scheduler
admission, continuation bookkeeping, pool allocation and batch merging are
compiled from the specified runtime sources.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from contextlib import nullcontext
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS


def methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


class Tensor(list):
    def __getitem__(self, index):
        if isinstance(index, list):
            return Tensor(super(Tensor, self).__getitem__(i) for i in index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if isinstance(index, list):
            for i, v in zip(index, value):
                super().__setitem__(i, v)
        else:
            super().__setitem__(index, value)

    def __iadd__(self, value):
        if isinstance(value, int):
            self[:] = [x + value for x in self]
            return self
        return super().__iadd__(value)

    def to(self, *args, **kwargs):
        return self


class HiddenMode(IntEnum):
    NULL = 0

    def need_capture(self):
        return False


class Req:
    def __init__(self, rid, *, owned=None, remaining=7587, beam=None):
        self.rid = rid
        self.kv = NS(req_pool_idx=owned, holds_kv=owned is not None, kv_allocated_len=1 if owned else 0,
                     holds_mamba=False)
        self.beam_group = NS(beam_width=beam) if beam else None
        self.prefix_indices = []
        self.full_untruncated_fill_ids = [0] * remaining
        self.sampling_params = NS(max_new_tokens=512)
        self.retracted_stain = False
        self.inflight_middle_chunks = 0
        self.return_logprob = False
        self.grammar = None
        self.done = False

    def init_next_round_input(self, *args):
        pass

    def set_extend_range(self, start, end):
        self.extend_range = NS(start=start, end=end, length=end - start)

    def finished(self):
        return self.done


class AddReqResult(IntEnum):
    CONTINUE = 0
    OTHER = 1
    NO_TOKEN = 2


class ReachedAdmission(Exception):
    def __init__(self, batch):
        self.batch = batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--scheduler', type=Path)
    parser.add_argument('--expect-bug', action='store_true')
    args = parser.parse_args()
    managers = args.source / 'srt/managers'
    parallel = NS(pp_max_micro_batch_size=4)
    schedule = NS(prefill_max_requests=None)
    disagg = NS(PREFILL='prefill', NULL='null')
    torch = NS(zeros=lambda shape, **kw: Tensor([0] * (shape if isinstance(shape, int) else shape[0])),
               tensor=lambda x, **kw: Tensor(x), cat=lambda seq: Tensor(x for row in seq for x in row),
               int32='int32', int64='int64')
    namespace = {'torch': torch, 'get_parallel': lambda: parallel, 'get_schedule': lambda: schedule,
                 'Req': Req, 'DisaggregationMode': disagg, 'AddReqResult': AddReqResult,
                 'CLIP_MAX_NEW_TOKENS': 4096, 'TEST_RETRACT': False,
                 'strip_beam_tail': lambda batch: None, 'CaptureHiddenMode': HiddenMode,
                 'get_batch_return_hidden_states_mode': lambda reqs: HiddenMode.NULL,
                 'is_pin_memory_available': lambda device: False,
                 'set_time_batch': lambda *a: None,
                 'PrefillStats': NS(from_adder=lambda *a, **kw: None),
                 'TorchMemorySaverAdapter': NS(create=lambda **kw: NS(region=lambda *a: nullcontext())),
                 'GPU_MEMORY_TYPE_KV_CACHE': 'kv'}
    # 0020 calls this pure policy even when disabled. Load the installed module
    # rather than replacing batch_budget with a fixture or silently omitting it.
    adaptive_path = args.source / 'srt/arg_groups/glm53_adaptive_prefill.py'
    if adaptive_path.exists():
        spec = importlib.util.spec_from_file_location('installed_glm53_adaptive_prefill', adaptive_path)
        adaptive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adaptive)
        assert not adaptive.ENABLED, 'Admission control fixture requires SGLANG_GLM53_ADAPTIVE_PREFILL=0'
        namespace['glm53_adaptive_prefill'] = adaptive
    pool_methods = methods(args.source / 'srt/mem_cache/memory_pool.py', 'ReqToTokenPool',
                           ['__init__', 'available_size', 'alloc', 'alloc_rows', 'free_rows', 'free'], namespace)
    Pool = type('Pool', (), {**pool_methods, '_on_alloc_rows': None})
    batch_methods = methods(managers / 'schedule_batch.py', 'ScheduleBatch',
                            ['filter_batch', 'merge_batch'], namespace)

    class Batch:
        filter_batch = batch_methods['filter_batch']
        merge_batch = batch_methods['merge_batch']

        def __init__(self, reqs, *, pool=None, chunked_req=None):
            self.reqs = list(reqs)
            self.pool = pool
            self.chunked_req = chunked_req
            self.batch_is_full = False
            self.forward_mode = NS(is_extend=lambda: True)
            self.sampling_info = NS(merge_batch=lambda *a: None, filter_batch=lambda *a: None)
            self.model_config = NS(is_encoder_decoder=False)
            for name in ('req_pool_indices', 'req_pool_indices_cpu', 'seq_lens', 'orig_seq_lens', 'seq_lens_cpu'):
                setattr(self, name, Tensor([1] * len(reqs)))
            self.input_ids = None
            self.input_embeds = None
            self.multimodal_inputs = None
            self.return_logprob = False
            self.has_grammar = False
            self.return_hidden_states_mode = HiddenMode.NULL
            self.is_prefill_only = False
            self.spec_info = None
            self.device = 'cpu'

        @classmethod
        def init_new(cls, reqs, pool, *args, chunked_req=None):
            return cls(reqs, pool=pool, chunked_req=chunked_req)

        def prepare_for_extend(self):
            rows = self.pool.alloc(self.reqs)
            assert rows is not None, 'Admission over-allocated request rows'
            for req in self.reqs:
                req.kv.holds_kv = True
                req.kv.kv_allocated_len = 1

        def batch_size(self):
            return len(self.reqs)

        def is_empty(self):
            return not self.reqs

    namespace['ScheduleBatch'] = Batch
    continuation = methods(managers / 'schedule_policy.py', 'PrefillAdder', ['add_chunked_req'], namespace)

    class Adder:
        add_chunked_req = continuation['add_chunked_req']

        def __init__(self, *args, **kwargs):
            self.can_run_list = []
            self.new_chunked_req = None
            self.preempt_list = []
            self.dllm_config = None
            self.rem_chunk_tokens = 1024
            self.rem_total_tokens = 100000
            self.is_hybrid_swa = False
            self.prefill_delayer_single_pass = None
            self.exact_chunk_fill = True

        def _merge_tail(self, *a):
            return False

        def _mamba_gap_budget_for_req(self, req):
            return 0

        def _update_prefill_budget(self, prefix, length, *a, **kw):
            self.rem_chunk_tokens -= length

        def add_one_req(self, req, **kw):
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER
            assert not req.kv.holds_kv
            length = min(len(req.full_untruncated_fill_ids), self.rem_chunk_tokens)
            req.set_extend_range(0, length)
            self.can_run_list.append(req)
            self.rem_chunk_tokens -= length
            if length < len(req.full_untruncated_fill_ids):
                self.new_chunked_req = req
            return AddReqResult.CONTINUE

    namespace['PrefillAdder'] = Adder
    scheduler_methods = methods(args.scheduler or managers / 'scheduler.py', 'Scheduler',
                                ['get_num_allocatable_reqs', '_get_new_batch_prefill_raw', 'get_next_batch_to_run'], namespace)
    Scheduler = type('Scheduler', (), scheduler_methods)

    def setup(*, running=2, pool_size=4, pp=4, remaining=35, owned=True,
              candidate_beam=None, continuation_beam=None, pending=0, mode='null', continuing=True):
        parallel.pp_max_micro_batch_size = pp
        pool = Pool(pool_size, 131072, 'cpu', False)
        assert pool.available_size() == pool_size and 0 not in pool.free_slots
        live = [Req(f'decode{i}') for i in range(running)]
        chunk = Req('continuation', remaining=remaining, beam=continuation_beam) if continuing else None
        held = live + ([chunk] if chunk is not None and owned else [])
        assert pool.alloc(held) is not None
        for req in held:
            req.kv.holds_kv = True
            req.kv.kv_allocated_len = 1
        waiter = Req('waiting', beam=candidate_beam)
        active = Batch(live)
        scheduler = Scheduler()
        scheduler.req_to_token_pool = pool
        scheduler.running_batch = active
        scheduler.chunked_req = chunk
        scheduler.waiting_queue = [waiter]
        scheduler.beam_coordinator = NS(pending_member_rows=lambda batch: pending)
        scheduler.grammar_manager = NS(has_waiting_grammars=lambda: False)
        scheduler.min_free_slots_delayer = None
        scheduler.policy = NS(calc_priority=lambda *a, **kw: None)
        scheduler.processed_tokens_counter = 0
        scheduler.chunked_prefill_size = 1024
        scheduler.dynamic_chunk_sizer = None
        scheduler.page_size = 64
        scheduler.tree_cache = NS(buffer_pipeline=None, storage_prefetch_retries=None)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.new_token_ratio_tracker = NS(current=.5)
        scheduler.max_prefill_tokens = 16384
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 4
        scheduler.max_running_requests = 4
        scheduler.disaggregation_mode = mode
        scheduler.truncation_align_size = None
        scheduler.model_config = active.model_config
        scheduler.spec_algorithm = NS(is_none=lambda: True)
        scheduler.dllm_config = None
        scheduler.tp_worker = NS(model_runner=NS(attn_backend=NS(), prefill_aware_swa=False))
        scheduler.load_inquirer = NS(_get_num_pending_tokens=lambda **kw: 0)
        for name in ('enable_priority_preemption', 'is_hybrid_swa', 'is_mixed_chunk', 'enable_lora',
                     'enable_hicache_storage', 'enable_hierarchical_cache', 'enable_unified_cache_external_linker',
                     'enable_priority_scheduling', 'enable_overlap', 'enable_hisparse', 'enable_fpm'):
            setattr(scheduler, name, False)
        scheduler.process_pending_chunked_abort = lambda: None
        scheduler._process_hicache_events = lambda: None
        scheduler._should_defer_prefill = lambda: False
        scheduler.stash_chunked_request = lambda req: None
        return scheduler, active, chunk, waiter

    def admission(**kw):
        scheduler, active, chunk, waiter = setup(**kw)
        new, same = scheduler._get_new_batch_prefill_raw(None, active)
        assert same is active
        return scheduler, active, new, chunk, waiter

    scheduler, active, new, chunk, waiter = admission()
    assert new is not None and chunk in new.reqs
    assert (waiter in new.reqs) != args.expect_bug
    assert active.batch_is_full == args.expect_bug
    # Execute actual filtering/merge control flow, then intercept the following
    # admission call. The unpatched final chunk removes nothing and keeps full.
    scheduler.get_new_batch_prefill = lambda batch: (_ for _ in ()).throw(ReachedAdmission(batch))
    try:
        scheduler.get_next_batch_to_run(active, new)
    except ReachedAdmission as reached:
        active = reached.batch
    else:
        raise AssertionError('Expected next admission boundary')
    assert len(active.reqs) == 3 and active.batch_is_full == args.expect_bug
    assert scheduler.req_to_token_pool.available_size() == (1 if args.expect_bug else 0)
    if args.expect_bug:
        result, _ = scheduler._get_new_batch_prefill_raw(None, active)
        assert result is None and scheduler.waiting_queue == [waiter]
        print('PASS: actual source reproduces three decoders + waiting request, one free row, stale full flag')
        return

    # Continuation moved to decode; the newly admitted waiter retained its row.
    # Its final chunk can join decode even with no free rows, producing C4.
    assert scheduler.chunked_req is waiter
    waiter.prefix_indices = [0] * 7552
    new, _ = scheduler._get_new_batch_prefill_raw(None, active)
    assert new is not None and new.reqs == [waiter]
    try:
        scheduler.get_next_batch_to_run(active, new)
    except ReachedAdmission as reached:
        active = reached.batch
    assert len(active.reqs) == 4
    scheduler.waiting_queue = [Req('fifth')]
    result, _ = scheduler._get_new_batch_prefill_raw(None, active)
    assert result is None and active.batch_is_full

    # An owned continuation still consumes a PP position, and pending beam rows
    # remain reserved. No credit is given to unowned/missing/beam continuations
    # or split-prefill's independent prealloc/transfer row budget.
    for kw in ({'pp': 3}, {'pool_size': 3}, {'pending': 1}, {'owned': False, 'pool_size': 3},
               {'candidate_beam': 2}, {'continuation_beam': 2}, {'mode': 'prefill'}):
        s, b, n, c, w = admission(**kw)
        assert n is not None and c in n.reqs and w not in n.reqs, kw
        assert b.batch_is_full, kw
    # A full-sized continuing chunk has no token budget left but must not leave
    # the batch marked full solely because it owns a row.
    s, b, n, c, w = admission(remaining=2048)
    assert n.reqs == [c] and not b.batch_is_full and s.chunked_req is c
    # Without a continuation the ordinary new-row and PP admission is unchanged.
    s, b, n, c, w = admission(continuing=False, running=3)
    assert n is not None and n.reqs == [w]
    s, b, n, c, w = admission(continuing=False, running=4)
    assert n is None and b.batch_is_full
    print('PASS: actual scheduler/continuation/allocator/filter/merge C3->C4; pool, PP, pending beam, beam, split-prefill and full-chunk bounds')


if __name__ == '__main__':
    main()
