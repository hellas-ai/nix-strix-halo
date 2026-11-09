#!/usr/bin/env python3
"""Actual cache method bodies with CPU tensors and explicitly surrogate tree/pools.

No package initialization, model kernels, fleet calls or full state inspection.
The pre-0064 source must reproduce the missing-donor assertion. The proposal
must preserve that assertion and qualify the real admission/common/unified/
MambaComponent method chain, including its internal non-COW rematch.

--installed checks candidate method bodies directly from --source and reverses
the immutable patch only in a temporary tree for the historical counterexample.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import ast
from array import array
import contextlib
from copy import deepcopy
import dataclasses
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace as N
from typing import NamedTuple
from unittest.mock import patch

for variable in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCP_TOOL_LIBRARIES'):
    os.environ[variable] = ''
import torch

FILES = (
    'srt/managers/glm53_prefill_geometry_trace.py',
    'srt/managers/schedule_batch.py', 'srt/mem_cache/common.py',
    'srt/mem_cache/registry.py', 'srt/mem_cache/base_prefix_cache.py',
    'srt/mem_cache/radix_cache.py', 'srt/mem_cache/memory_pool.py',
    'srt/mem_cache/unified_radix_cache.py',
    'srt/mem_cache/unified_cache/components/mamba.py',
)
UNCHANGED = {
    'srt/managers/glm53_prefill_geometry_trace.py': '3d97933c30cda988a5a469693c4fe5462cf5896d17386329ef00b5dce703a80d',
    'srt/layers/sampler.py': '5ac36528a7de48a5796c376d3ee307e9ce1bb57d306eff70827f6fb169b58e5e',
    'srt/sampling/glm53_forced_decode_reference.py': 'd7c98eef773093a5f97175b0523d24cd7687dfa28c5c903c92cd9fa084b4d8d5',
}
UNIFIED_FILE = 'srt/mem_cache/unified_radix_cache.py'
UNIFIED_BASELINE_SHA = '92744ae4876479c3bf7838cb6c55888d2416629f9ea1b3b37a8d0c5288a3214b'
UNIFIED_CANDIDATE_SHA = '4f2ea48f6db04b835b3e0994f5f7aebd05d49841cca744a74798d7ea2876fe76'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(source, *, installed):
    for rel, expected in UNCHANGED.items():
        assert sha(source / rel) == expected, rel
    expected = UNIFIED_CANDIDATE_SHA if installed else UNIFIED_BASELINE_SHA
    assert sha(source / UNIFIED_FILE) == expected, 'Source does not match explicit installed mode'


def load_trace(source):
    path = source / FILES[0]
    spec = importlib.util.spec_from_file_location('actual_trace_0064', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ENABLED = True
    return module


@contextlib.contextmanager
def cpu_only():
    assert not torch.cuda.is_initialized()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('GPU initialization or selection attempted')
    with contextlib.ExitStack() as stack:
        for name in ('init', '_lazy_init', 'set_device', 'device', 'device_of', '_exchange_device', '_maybe_exchange_device', 'Stream', 'Event'):
            if hasattr(torch.cuda, name):
                stack.enter_context(patch.object(torch.cuda, name, forbidden))
        stack.enter_context(patch.object(torch, 'set_default_device', forbidden))
        for action in (torch.cuda.init, lambda: torch.cuda.set_device(0),
                       lambda: torch.set_default_device('cuda'), lambda: torch.empty(1, device='cuda')):
            reject(action)
        assert len(calls) == 4
        calls.clear()
        yield
        assert not calls and not torch.cuda.is_initialized()


def reject(action, literal=None):
    try:
        action()
    except (AssertionError, ValueError) as error:
        if literal is not None:
            assert str(error) == literal, (type(error).__name__, str(error))
        return
    raise AssertionError('Expected refusal did not occur')


def execute(nodes, path, namespace):
    copied = deepcopy(nodes)
    for node in copied:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # TP decorators are outside this CPU body proof.
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *copied], type_ignores=[])
    namespace.setdefault('__name__', __name__)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def methods(path, cls, names, namespace):
    tree = ast.parse(path.read_bytes())
    container = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    found = [n for n in container.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert set(n.name for n in found) == set(names)
    execute(found, path, namespace)
    return {name: namespace[name] for name in names}


def functions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert set(n.name for n in found) == set(names)
    execute(found, path, namespace)
    return {name: namespace[name] for name in names}


def data_classes(source):
    path = source / 'srt/mem_cache/base_prefix_cache.py'
    tree = ast.parse(path.read_bytes())
    names = {'MatchPrefixParams', 'InsertParams', 'InsertResult', 'MatchResult'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    namespace = dict(dataclasses=dataclasses, torch=torch, NamedTuple=NamedTuple)
    execute(nodes, path, namespace)
    keypath = source / 'srt/mem_cache/radix_cache.py'
    key = next(n for n in ast.parse(keypath.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == 'RadixKey')
    namespace['array'] = array
    execute([key], keypath, namespace)
    return namespace


class ComponentType(Enum):
    FULL = 1
    MAMBA = 2
    SWA = 3


class Pool:
    """CPU integer allocation/row surrogate, not actual GPU state storage."""
    def __init__(self):
        self.req_to_token = torch.arange(8 * 6000).reshape(8, 6000)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros((8, 2), dtype=torch.int64)
        self.next_slot = 30
        self.ops = []
        self.mamba_ckpt_pool = None
        self.mamba_pool = N(replayssm_write_pos=None)
        self.mamba_allocator = N(alloc=self.alloc, free=lambda value: self.ops.append(('free', value.tolist())))
        self.enable_mamba_extra_buffer = True
        self.enable_mamba_extra_buffer_lazy = False
        self.mamba_ping_pong_track_buffer_size = 2
        self.mamba_v2p_table = None
        self.mamba_translate_is_fusable = True
        self.active_by_row = {}

    def alloc(self, count):
        value = torch.arange(self.next_slot, self.next_slot + count)
        self.next_slot += count
        self.ops.append(('alloc', value.tolist()))
        return value

    def write(self, target, value):
        self.req_to_token[target] = value

    def get_mamba_indices(self, rows):
        return torch.tensor([self.active_by_row[int(row)] for row in rows.tolist()])

    def free_mamba_cache(self, req, mamba_ping_pong_track_buffer_to_keep=None):
        self.ops.append(('finish_free', int(req.kv.mamba_pool_idx), mamba_ping_pong_track_buffer_to_keep))
        req.kv.mamba_pool_idx = None
        req.kv.mamba_ping_pong_track_buffer = None
        req.kv.mamba_last_track_seqlen = None
        req.kv.mamba_cow_src_index = None

    def free(self, req):
        self.ops.append(('row_free', req.kv.req_pool_idx))


class Request:
    def __init__(self, name, slot, length, pool):
        self.rid = 'glm53-serial-prefix-control-v1-' + name
        self.origin_input_ids = [3] * length
        self.output_ids = []
        self.full_untruncated_fill_ids = []
        self.extend_range = N(start=0, end=length, length=length)
        self.extra_key = self.cache_salt = self.session_id = self.session = None
        self.kv_rotation_base = self.positional_embed_overrides = self.multimodal_inputs = None
        self.priority = 0
        self.return_logprob = True
        self.logprob_start_len = -1
        self.is_retracted = False
        self.swa_prefix_lock_released = False
        self.finished_reason = None
        self.last_node = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.mamba_branching_seqlen = None
        active, first, second = slot * 3 - 2, slot * 3 - 1, slot * 3
        self.kv = N(req_pool_idx=slot, mamba_pool_idx=torch.tensor(active),
            mamba_ping_pong_track_buffer=torch.tensor([first, second]),
            mamba_cow_src_index=None, mamba_needs_clear=True,
            mamba_last_track_idx=1, mamba_next_track_idx=0, mamba_last_track_seqlen=None,
            cache_protected_len=0, holds_mamba=True, holds_kv=True, is_kv_released=False,
            kv_allocated_len=length, kv_committed_len=length)
        self.kv.mark_kv_released = lambda: setattr(self.kv, 'is_kv_released', True)
        pool.active_by_row[slot] = active
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping[slot] = self.kv.mamba_ping_pong_track_buffer

    def is_dllm(self):
        return False

    def _refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def get_fill_ids(self):
        return self.origin_input_ids + self.output_ids

    def effective_kv_committed_len(self):
        return self.kv.kv_committed_len


class Fixture:
    def __init__(self, source, trace, rank=0, enabled=True):
        self.source, self.trace, self.rank = source, trace, rank
        trace.ENABLED = enabled
        trace.ledger = trace.Ledger()
        self.records = []
        self.log_patch = patch.object(trace.logger, 'info', side_effect=self.records.append)
        self.log_patch.start()
        self.types = data_classes(source)
        self.pool = Pool()
        namespace = dict(self.types, torch=torch, glm53_geometry=trace,
            glm53_cache_depth_diagnostic=N(ENABLED=False), ComponentType=ComponentType,
            BASE_COMPONENT_TYPE=ComponentType.FULL,
            envs=N(SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=N(get=lambda: False),
                   SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=N(get=lambda: False)),
            replace=dataclasses.replace)
        upath = source / 'srt/mem_cache/unified_radix_cache.py'
        names = ['match_prefix', 'insert', 'cache_unfinished_req', 'cache_finished_req']
        if '_glm53_trace_mamba_donation' in upath.read_text():
            names.append('_glm53_trace_mamba_donation')
        cache_methods = methods(upath, 'UnifiedRadixCache', names, namespace)
        Cache = type('SourceUnifiedCacheBodies', (), cache_methods)
        self.cache = Cache()
        c = self.cache
        c.disable = False
        c.is_mamba_enabled = c.enable_mamba_extra_buffer = True
        c.enable_session_radix_cache = False
        c.req_to_token_pool = self.pool
        c.page_size = 1024
        c.linker = None
        c.session = N(try_match_prefix=lambda params: None,
            try_cache_finished_req=lambda *a, **kw: False,
            try_cache_unfinished_req=lambda *a, **kw: False)
        c._apply_cache_actions = lambda actions: None
        c._dec_req_lock = lambda *a, **kw: None
        c.inc_lock_ref = lambda *a, **kw: N(to_dec_params=lambda: None)
        c.free_kv_row = lambda kv, spans: self.pool.ops.append(('kv_free', spans))
        c.supports_mamba = lambda: True
        c.swa_reprefill_tail_tokens = lambda: 0
        c.token_to_kv_pool_allocator = N(page_size=1024)
        self.states = {}
        self.match_calls = []
        self.decline = False
        c.tree_core = N(is_eagle=False, match_prefix=self.tree_match,
            get_component_device_value=lambda node, ct: self.states.get(node),
            has_ongoing_insert=lambda: False, begin_insert=self.tree_insert,
            end_insert=lambda: [])
        component_namespace = dict(torch=torch, glm53_cache_depth_diagnostic=N(ENABLED=False))
        mn = ['finalize_match_result_in_cache', 'prepare_for_caching_req', 'cleanup_after_caching_req',
              '_alloc_mamba_slot', '_free_mamba_value']
        Mamba = type('SourceMambaBodies', (), methods(source / 'srt/mem_cache/unified_cache/components/mamba.py', 'MambaComponent', mn, component_namespace))
        Mamba.int8_ckpt_pool = property(lambda component: None)
        mamba = Mamba(); mamba.cache = c; mamba.tree_core = c.tree_core
        mamba.component_type = ComponentType.MAMBA
        c._components_tuple = (mamba,)
        pool_namespace = dict(torch=torch, _MAMBA_DEBUG_ASSERTS=False)
        pn = ['get_mamba_ping_pong_keep_idx', 'get_mamba_ping_pong_other_idx',
              'set_mamba_ping_pong_slot', 'donate_mamba_ping_pong_slot']
        for name, fn in methods(source / 'srt/mem_cache/memory_pool.py', 'HybridReqToTokenPool', pn, pool_namespace).items():
            setattr(self.pool, name, fn.__get__(self.pool, Pool))
        request_namespace = dict(self.types, array=array,
            envs=N(SGLANG_RADIX_FORCE_MISS=N(get=lambda: False)))
        for name, fn in methods(source / 'srt/managers/schedule_batch.py', 'Req',
                ['init_next_round_input', '_compute_max_prefix_len', 'get_fill_ids', '_refresh_fill_ids'], request_namespace).items():
            setattr(Request, name, fn)
        common_namespace = dict(get_spec=lambda: N(speculative_algorithm=None),
            get_serving=lambda: N(strip_thinking_cache=False),
            ceil_align=lambda value, multiple: (value + multiple - 1) // multiple * multiple,
            HybridReqToTokenPool=Pool)
        self.common = functions(source / 'srt/mem_cache/common.py',
            ['maybe_cache_unfinished_req', 'release_kv_cache', '_release_overallocated_kv_indices'], common_namespace)
        self.schedule = N(tree_cache=c, model_config=N(hf_text_config=N(model_type='glm5_next_text', mamba_chunk_size=64)))
        track_namespace = dict(glm53_adaptive_prefill=N(ENABLED=False),
            mamba_cache_chunk_size=lambda: 64, mamba_checkpoint_grid=lambda page_size: page_size,
            get_exec=lambda: N(mamba=N(enable_mamba_extra_buffer_lazy=False)),
            _MambaRadixCacheV2TrackEntry=lambda **kw: N(**kw))
        self.schedule._mamba_radix_cache_v2_req_prepare_for_extend = methods(
            source / 'srt/managers/schedule_batch.py', 'ScheduleBatch',
            ['_mamba_radix_cache_v2_req_prepare_for_extend'], track_namespace
        )['_mamba_radix_cache_v2_req_prepare_for_extend'].__get__(self.schedule)
        self.schedule.req_to_token_pool = self.pool
        if enabled:
            trace.reset_completed(rank)

    def close(self):
        self.log_patch.stop()

    def parsed(self):
        return [json.loads(record.split(' ', 1)[1]) for record in self.records]

    def tree_match(self, params):
        self.match_calls.append((params.cow_mamba, None if params.req is None else params.req.rid))
        tokens = tuple(params.key.raw_token_ids())
        candidates = [key for key in self.states if tokens[:len(key)] == key]
        key = max(candidates, key=len) if candidates else ()
        return self.types['MatchResult'](torch.arange(len(key)), key, key, key)

    def tree_insert(self, params):
        key = tuple(params.key.raw_token_ids())
        existed = not key or key in self.states
        if key and not existed and not self.decline:
            self.states[key] = params.mamba_value.clone()
        result = self.types['InsertResult'](prefix_len=len(key), last_device_node=key,
            mamba_exist=existed, rotation_tail_declined=self.decline)
        return N(actions=[], result=result)

    def make(self, name, slot=1, length=3788):
        return Request(name, slot, length, self.pool)

    def extend(self, req, depth, prefix=None, corrupt_cow=False):
        prefix = len(req.prefix_indices) if prefix is None else prefix
        req.extend_range = N(start=prefix, end=depth, length=depth - prefix)
        entry = self.schedule._mamba_radix_cache_v2_req_prepare_for_extend(req)
        b = N(reqs=[req], enable_overlap=True, decoding_reqs=None,
            spec_algorithm=N(is_none=lambda: True), model_config=self.schedule.model_config)
        b.glm53_geometry_prepared = self.trace.snapshot_extend(b,
            [req.origin_input_ids[prefix:depth]], [depth], [prefix], [depth-prefix],
            [entry.track_seqlen], [entry.track_mask], enabled=True)
        active = req.kv.mamba_pool_idx.reshape(1)
        cow = req.kv.mamba_cow_src_index
        fb = N(forward_mode=N(is_extend=lambda: True, is_decode=lambda: False),
            glm53_geometry_packet=None, input_ids=torch.ones(depth-prefix, dtype=torch.int64),
            batch_size=1, req_pool_indices=torch.tensor([req.kv.req_pool_idx]),
            seq_lens=torch.tensor([depth]), mamba_track_indices=torch.tensor([entry.track_index]),
            mamba_track_mask=torch.tensor([entry.track_mask]),
            mamba_track_seqlens=torch.tensor([entry.track_seqlen]),
            mamba_cow_src_indices=None if cow is None else cow.reshape(-1).clone(),
            mamba_cow_dst_indices=None if cow is None else active.clone(),
            mamba_clear_indices=active.clone() if req.kv.mamba_needs_clear else None)
        runner = N(is_draft_worker=False, spec_algorithm=N(is_none=lambda: True),
            model_config=self.schedule.model_config, ps=N(tp_rank=self.rank, pp_size=1),
            req_to_token_pool=self.pool)
        packet = self.trace.start(b, fb, runner, enabled=True)
        self.trace.after_padding(fb); self.trace.dispatch(fb, 'eager')
        if corrupt_cow:
            packet.prepared = dataclasses.replace(packet.prepared,
                peers=(dataclasses.replace(packet.prepared.peers[0], cow_source=None),))
        packet.copy_to_host(lambda tensor: tensor.clone())
        self.trace.complete(N(glm53_geometry_packet=packet))
        req.kv.mamba_cow_src_index = None
        req.kv.mamba_needs_clear = False
        req.kv.kv_committed_len = req.kv.kv_allocated_len = depth
        return packet

    def warm(self, name='scores-NORMAL-warm-1'):
        req = self.make(name)
        req.init_next_round_input(self.cache)
        for depth in (1024, 2048, 3072):
            packet = self.extend(req, depth)
            donor = int(packet.slots_host[0][req.kv.mamba_last_track_idx + 1])
            keep = req.kv.mamba_last_track_idx
            self.common['maybe_cache_unfinished_req'](req, self.cache, chunked=True)
            assert int(req.kv.mamba_ping_pong_track_buffer[keep]) != donor
            assert int(self.states[tuple(req.origin_input_ids[:depth])][0]) == donor
        self.extend(req, 3788)
        req.output_ids = [1654]
        self.common['release_kv_cache'](req, self.cache)
        return req


def factory_selection(source):
    """Execute actual factory bodies; constructor/import edges are CPU stubs."""
    namespace = dict(get_disagg=lambda: N(disaggregation_decode_retraction_backup=None),
        get_memory=lambda: N(enable_unified_cache_external_linker=False, enable_lmcache=False, enable_flexkv=False),
        use_mlx=lambda: False, envs=N(SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=N(get=lambda: False)))
    functions(source / 'srt/mem_cache/registry.py', ['default_radix_cache_factory', '_create_unified_radix_cache'], namespace)
    modules = {}
    comp = ModuleType('sglang.srt.mem_cache.unified_cache.components'); comp.ComponentType = ComponentType
    unified = ModuleType('sglang.srt.mem_cache.unified_radix_cache')
    unified.UnifiedRadixCache = lambda params: N(params=params, constructor='UnifiedRadixCache')
    for name in ('sglang', 'sglang.srt', 'sglang.srt.mem_cache', 'sglang.srt.mem_cache.unified_cache'):
        modules[name] = ModuleType(name)
    modules[comp.__name__] = comp; modules[unified.__name__] = unified
    params = N(req_to_token_pool=N())
    ctx = N(server_args=N(), params=params, disable_radix_cache=False,
        effective_chunked_prefill_size=1024, is_hybrid_swa=False, is_hybrid_ssm=True,
        enable_hierarchical_cache=False)
    with patch.dict(sys.modules, modules):
        cache = namespace['default_radix_cache_factory'](ctx)
    assert cache.constructor == 'UnifiedRadixCache'
    assert params.tree_components == (ComponentType.FULL, ComponentType.MAMBA)
    return 'Actual default factory creates unified FULL+MAMBA; constructor/import edges stubbed, nondefault cache flags not observed live.'


def lifecycle(source, trace, rank):
    f = Fixture(source, trace, rank)
    try:
        f.warm()
        kinds = [r['kind'] for r in f.parsed()]
        assert kinds.count('donation') == 3 and kinds.count('request_finished') == 1
        assert [r['depth'] for r in f.parsed() if r['kind'] == 'match'] == [0]
        measured = f.make('scores-NORMAL-measure-1', slot=2)
        measured.init_next_round_input(f.cache)
        assert len(measured.prefix_indices) == 3072
        assert measured._glm53_geometry_match is not None
        f.extend(measured, 3788)
        matched = [r for r in f.parsed() if r['kind'] == 'match'][-1]
        assert matched['depth'] == 3072 and matched['cow'] is True
        completed = [r for r in f.parsed() if r['kind'] == 'completed'][-1]
        assert completed['live_M'] == 716 and completed['peers'][0]['cow_match_event'] == matched['lifecycle_id']
        assert any(cow is False and rid is not None for cow, rid in f.match_calls)
        # No new checkpoint in the cached tail: real component returns0 and
        # actual finished flow inserts an empty key. Finish is still recorded.
        measured.output_ids = [1654]
        f.common['release_kv_cache'](measured, f.cache)
        assert len([r for r in f.parsed() if r['kind'] == 'donation']) == 3
        assert len([r for r in f.parsed() if r['kind'] == 'request_finished']) == 2
        trace.before_reset(); trace.reset_completed(rank)
        assert f.parsed()[-1]['pending'] == 0
        return {'rank': rank, 'donations': 3, 'finished': 2, 'cached_extend_M': 716,
                'internal_non_COW_rematches': sum(not cow for cow, _ in f.match_calls)}
    finally:
        f.close()


def old_failure(source, trace):
    f = Fixture(source, trace)
    try:
        f.warm()
        assert not any(r['kind'] in ('match', 'donation', 'request_finished') for r in f.parsed())
        measured = f.make('scores-NORMAL-measure-1', slot=2)
        measured.init_next_round_input(f.cache)
        assert measured.kv.mamba_cow_src_index is not None
        reject(lambda: f.extend(measured, 3788), 'COW source lacks a tracked cache donation')
        return 'Actual old common/unified/component chain stores/reuses a COW checkpoint but emits no lifecycle; complete310 refuses it.'
    finally:
        f.close()


def negative_donor(source, trace):
    f = Fixture(source, trace)
    try:
        f.warm()
        measured = f.make('scores-NORMAL-measure-1', slot=2)
        measured.init_next_round_input(f.cache)
        reject(lambda: f.extend(measured, 3788, corrupt_cow=True), 'COW source lacks a tracked cache donation')
    finally:
        f.close()


def skipped_inserts(source, trace):
    f = Fixture(source, trace)
    try:
        f.decline = True
        req = f.make('scores-NORMAL-warm-0', length=1024)
        req.init_next_round_input(f.cache); f.extend(req, 1024)
        f.common['maybe_cache_unfinished_req'](req, f.cache, chunked=True)
        assert not any(r['kind'] == 'donation' for r in f.parsed())
        # A declined insert frees the unused donor via existing component
        # cleanup; it cannot become a future selected checkpoint.
        assert not f.states
        req.output_ids = [1654]
        f.common['release_kv_cache'](req, f.cache, is_insert=False)
        assert len([r for r in f.parsed() if r['kind'] == 'request_finished']) == 1
        trace.before_reset()
    finally:
        f.close()


def finished_and_duplicate(source, trace):
    f = Fixture(source, trace)
    try:
        for name, slot in [('scores-NORMAL-warm-0', 1), ('scores-NORMAL-warm-1', 2)]:
            req = f.make(name, slot, length=1024)
            req.init_next_round_input(f.cache)
            f.extend(req, 1024)
            req.output_ids = [1654]
            f.common['release_kv_cache'](req, f.cache)
        donations = [r for r in f.parsed() if r['kind'] == 'donation']
        assert [r['inserted'] for r in donations] == [True, False]
        assert all(r['finished'] is True for r in donations)
        # Duplicate donation must not replace the retained original donor.
        assert trace.ledger.cache[trace.ids_sha([3] * 1024)][3] == donations[0]['lifecycle_id']
        assert len([r for r in f.parsed() if r['kind'] == 'request_finished']) == 2
        trace.before_reset()
    finally:
        f.close()


def source_closure(base, changed):
    relative = 'srt/mem_cache/unified_radix_cache.py'
    before = ast.parse((base / relative).read_bytes())
    after = ast.parse((changed / relative).read_bytes())
    removed = dict(imports=0, helper=0, calls=0, gates=0)
    new_body = []
    for node in after.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'sglang.srt.managers':
            assert [(name.name, name.asname) for name in node.names] == [('glm53_prefill_geometry_trace', 'glm53_geometry')]
            removed['imports'] += 1
            continue
        if isinstance(node, ast.ClassDef) and node.name == 'UnifiedRadixCache':
            class_body = []
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == '_glm53_trace_mamba_donation':
                    removed['helper'] += 1
                    continue
                if isinstance(fn, ast.FunctionDef):
                    body = []
                    for statement in fn.body:
                        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                                and isinstance(statement.value.func, ast.Attribute)
                                and statement.value.func.attr == '_glm53_trace_mamba_donation'):
                            assert fn.name in ('cache_finished_req', 'cache_unfinished_req')
                            removed['calls'] += 1
                            continue
                        if isinstance(statement, ast.If) and any(
                                isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                                and isinstance(x.func.value, ast.Name) and x.func.value.id == 'glm53_geometry'
                                and x.func.attr in ('cache_match', 'cache_finished') for x in ast.walk(statement)):
                            assert fn.name in ('match_prefix', 'cache_finished_req')
                            removed['gates'] += 1
                            continue
                        body.append(statement)
                    fn.body = body
                class_body.append(fn)
            node.body = class_body
        new_body.append(node)
    after.body = new_body
    # One donation call is nested under is_insert in finished handling;
    # remove that exact extra statement from the remaining nested body.
    for node in ast.walk(after):
        if isinstance(node, ast.If):
            filtered = []
            for statement in node.body:
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Attribute)
                        and statement.value.func.attr == '_glm53_trace_mamba_donation'):
                    removed['calls'] += 1
                    continue
                filtered.append(statement)
            node.body = filtered
    assert removed == dict(imports=1, helper=1, calls=2, gates=2), removed
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    return removed


def off_equivalence(base, changed, trace):
    outputs = []
    for source in (base, changed):
        f = Fixture(source, trace, enabled=False)
        try:
            req = f.make('scores-NORMAL-warm-0', length=1024)
            req.init_next_round_input(f.cache)
            # No trace forward in off mode; simulate existing track metadata.
            req.kv.mamba_last_track_seqlen = 1024; req.kv.mamba_last_track_idx = 0
            f.common['maybe_cache_unfinished_req'](req, f.cache, chunked=True)
            req.output_ids = [1654]
            f.common['release_kv_cache'](req, f.cache)
            assert not f.records
            outputs.append({'ops': f.pool.ops,
                'states': sorted((len(key), value.tolist()) for key, value in f.states.items()),
                'row': f.pool.req_to_token.tolist(), 'released': req.kv.is_kv_released})
        finally:
            f.close()
    assert outputs[0] == outputs[1]
    return hashlib.sha256(json.dumps(outputs[0], sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--installed', action='store_true',
                        help='Test the installed candidate directly; reverse the patch only in a temporary baseline')
    parser.add_argument('--patch', type=Path, default=Path(__file__).resolve().parents[1] / 'patches/0064-glm53-unified-cache-geometry-lifecycle.patch')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    validate_source(source, installed=args.installed)
    with tempfile.TemporaryDirectory(prefix='glm53-unified-lifecycle-cpu-') as temporary:
        copied = Path(temporary) / 'sglang'
        for rel in sorted(set(FILES) | set(UNCHANGED)):
            path = copied / rel; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, path)
        command = ['patch', '--batch', '-p1', '--fuzz=0', '--input', str(args.patch.resolve())]
        if args.installed:
            command.append('--reverse')
        result = subprocess.run(command, cwd=temporary, capture_output=True, check=True)
        patch_output = (result.stdout + result.stderr).lower()
        assert b'offset' not in patch_output and b'fuzz' not in patch_output
        baseline, target = (copied, source) if args.installed else (source, copied)
        assert sha(baseline / UNIFIED_FILE) == UNIFIED_BASELINE_SHA
        assert sha(target / UNIFIED_FILE) == UNIFIED_CANDIDATE_SHA
        for rel in FILES:
            if rel != UNIFIED_FILE:
                assert (baseline / rel).read_bytes() == (target / rel).read_bytes(), rel
        reject(lambda: validate_source(baseline, installed=True))
        reject(lambda: validate_source(target, installed=False))
        trace = load_trace(source)
        with cpu_only():
            results = {'factory': factory_selection(baseline), 'old_failure': old_failure(baseline, trace),
                'rank_lifecycles': [lifecycle(target, trace, rank) for rank in range(4)]}
            negative_donor(target, trace)
            skipped_inserts(target, trace)
            finished_and_duplicate(target, trace)
            results['negative_donor_assertion_preserved'] = True
            results['declined_insert_and_noninsert_finish'] = True
            results['positive_finished_and_duplicate_donation'] = True
            results['unchanged_AST_after_removing_only_observer_additions'] = source_closure(baseline, target)
            results['off_CPU_effects_digest'] = off_equivalence(baseline, target, trace)
            results['explicit_mode_rejects_baseline_as_installed_and_candidate_as_baseline'] = True
        receipt = {'version': 2, 'status': 'PASS_CPU_SOURCE_ENTRYPOINTS_NATIVE_UNTESTED',
            'results': results, 'patch_sha256': sha(args.patch), 'test_sha256': sha(Path(__file__)),
            'source_mode': 'installed_candidate_reverse_patch' if args.installed else 'baseline_forward_patch',
            'input_source': str(source),
            'baseline_source': 'temporary_reverse_of_installed_candidate' if args.installed else str(source),
            'baseline_pins': {rel: sha(baseline / rel) for rel in FILES},
            'proposed_pins': {rel: sha(target / rel) for rel in FILES},
            'installed_binding': {'source': str(source),
                'pins': {rel: sha(source / rel) for rel in sorted(set(FILES) | set(UNCHANGED))},
                'actual_candidate_method_bodies_executed': True,
                'temporary_tree_contains_only_reconstructed_baseline': True} if args.installed else None,
            'unchanged_sampler_reference_trace': UNCHANGED,
            'scope': 'Actual method bodies, CPU Torch tensors; tree traversal, allocator/locks, constructor/import edges and state storage are surrogates. No TP collectives, GPU/model or actual fleet cache-class observation.',
            'GPU_initialized': torch.cuda.is_initialized(), 'GPU_qualified': False,
            'deployed': False, 'numeric_policy_changed': False,
            'old_bundle_or_run_changed': False}
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
        if args.receipt:
            with args.receipt.open('x') as stream:
                stream.write(encoded)
        print(encoded, end='')


if __name__ == '__main__':
    main()
