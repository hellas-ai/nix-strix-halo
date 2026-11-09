"""Real pointer-layout round trips and KDA lifecycle checks, CPU tensors only."""
import ast
import ctypes
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from typing import NamedTuple
from pathlib import Path
import sys
from types import SimpleNamespace as NS, MethodType
import unittest

import torch

ROOT = Path(sys.argv.pop(1))
SRT = ROOT / 'sglang/srt'
if not SRT.exists():
    SRT = next(ROOT.glob('lib/python*/site-packages/sglang/srt'))


def definitions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(
        x.body for x in tree.body if getattr(x, 'name', '') == class_name)
    selected = [x for x in body if getattr(x, 'name', '') in names]
    assert len(selected) == len(names)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), namespace)


PoolName = NS(KV='kv', INDEXER='indexer', MAMBA='mamba')
Policy = NS(ALL_PAGES='all', TRAILING_PAGES='trailing')
Phase = Enum('Phase', 'ABORT PREPARE COMMIT')
LinkPhase = Enum('LinkPhase', 'LOOKUP LOAD OFFLOAD')

@dataclass
class Transfer:
    name: str
    keys: object = None
    device_indices: object = None
    host_indices: object = None
    hit_policy: object = None
    indices_from_pool: object = None

ns = dict(torch=torch, replace=replace, PoolName=PoolName, PoolHitPolicy=Policy,
          PoolTransfer=Transfer, ExternalLinkerLoadPhase=Phase,
          LinkerTransferPhase=LinkPhase, EvictParams=lambda **kw: NS(**kw))
definitions(SRT/'mem_cache/hybrid_cache/linker_pool_assembler.py',
            {'DevicePoolEntry', 'DevicePoolGroup', 'PackedDSADevicePoolEntry'}, ns)
Entry, Packed, Group = (ns[x] for x in ('DevicePoolEntry', 'PackedDSADevicePoolEntry', 'DevicePoolGroup'))
definitions(SRT/'mem_cache/unified_cache/components/mamba.py',
            {'build_external_linker_transfer', 'update_external_linker_load'}, ns,
            'MambaComponent')


def physical_indices(pages):
    return (torch.tensor(pages)[:, None]*64+torch.arange(64)).flatten()


def gather(entry, indices):
    ptrs, sizes = entry.get_page_buffer_meta(indices)
    return [ctypes.string_at(p, n) for p, n in zip(ptrs, sizes)]


def scatter(entry, indices, objects):
    ptrs, sizes = entry.get_page_buffer_meta(indices)
    assert list(map(len, objects)) == sizes
    for p, n, value in zip(ptrs, sizes, objects):
        ctypes.memmove(p, value, n)


class LayoutTests(unittest.TestCase):
    def pool(self, *, indexer=False, logical=256, capacity=40):
        buffers = [torch.arange(capacity*(64 if not indexer else 1)*12,
                               dtype=torch.int32).reshape(-1, 12) + i*100000
                   for i in range(2)]
        return Packed(name='indexer' if indexer else 'kv', indices_from_pool='kv',
                      device_pool=None, components=[buffers], layer_mapping={0:0, 4:1},
                      logical_page_size=logical, physical_page_size=64,
                      index_kpool=4, indexer=indexer)

    def test_noncontiguous_pages_round_trip_both_logical_grids(self):
        for logical in (256, 1024):
            n=logical//64
            src=physical_indices(list(range(1,n+1))[::-1])
            dst=physical_indices(list(range(n+2,2*n+2)))
            for indexer in (False, True):
                with self.subTest(logical=logical, indexer=indexer):
                    source=self.pool(indexer=indexer, logical=logical)
                    target=self.pool(indexer=indexer, logical=logical)
                    for b in target.kv_buffer: b.fill_(-1)
                    blobs=gather(source, src)
                    scatter(target, dst, blobs)
                    self.assertEqual(gather(target,dst), blobs)
                    self.assertEqual(len(blobs), 2*(n//4 if indexer else n))
                    selected = dst.reshape(-1,64)[::4,0]//64 if indexer else dst
                    for original, restored in zip(source.kv_buffer,target.kv_buffer):
                        mask=torch.ones(restored.shape[0],dtype=torch.bool)
                        mask[selected]=False
                        self.assertTrue(torch.all(restored[mask] == -1))

    def test_index_anchors_follow_first_physical_page_in_each_group(self):
        entry=self.pool(indexer=True, logical=1024)
        ids=physical_indices([7,2,8,1, 13,5,6,3, 21,9,10,11, 30,15,16,17])
        ptrs,sizes=entry.get_page_buffer_meta(ids)
        for layer in range(2):
            self.assertEqual(ptrs[layer*4:layer*4+4],
                [entry.kv_buffer[layer][i].data_ptr() for i in (7,13,21,30)])
        self.assertEqual(len(sizes),8)

    def test_mamba_temporal_and_all_conv_fields_round_trip(self):
        tensors=[torch.arange(2*8*width,dtype=dtype).reshape(2,8,width)
                 for width,dtype in ((13,torch.float32),(5,torch.bfloat16),(7,torch.bfloat16))]
        def entry(buffers):
            return Entry(name='mamba',indices_from_pool='mamba',device_pool=None,
                         components=[[b[i] for i in range(2)] for b in buffers],
                         layer_mapping={1:0,2:1},page_size=1,rows_are_pages=True,packed=False)
        dest=[torch.zeros_like(x) for x in tensors]
        blobs=gather(entry(tensors),torch.tensor([3]))
        scatter(entry(dest),torch.tensor([6]),blobs)
        for a,b in zip(tensors,dest):
            self.assertTrue(torch.equal(a[:,3],b[:,6]))
            self.assertTrue(torch.all(b[:,0]==0))
        self.assertEqual(len(blobs),6)

    def test_invalid_partial_or_misaligned_pages_fail_closed(self):
        entry=self.pool()
        for ids in (torch.arange(255),torch.arange(1,257),torch.arange(4096,4352)):
            with self.assertRaises(ValueError): entry.get_page_buffer_meta(ids)
        with self.assertRaises(ValueError): self.pool(logical=64)
        with self.assertRaises(RuntimeError): entry.prepare_locations(torch.arange(256))

    def test_expansion_requires_kda_for_lookup_and_preserves_rank_sharding(self):
        kv=self.pool(); ix=self.pool(indexer=True)
        state=Entry(name='mamba',indices_from_pool='mamba',device_pool=None,
                    components=[[torch.zeros(8,3)]],layer_mapping={1:0},
                    page_size=1,rows_are_pages=True)
        group=Group([kv,ix,state],2,256,rank_replicated=False)
        full=Transfer('kv',keys=['a'],device_indices=physical_indices([1,3,2,4]))
        self.assertEqual(group.resolve_transfers([full]),[])
        transfers=group.resolve_transfers([full,Transfer('mamba',keys=['a'],device_indices=torch.tensor([2]))])
        self.assertEqual([t.name for t in transfers],['kv','indexer','mamba'])
        self.assertFalse(group.rank_replicated)


class StateTests(unittest.TestCase):
    def component(self, holds=False, fail=False):
        frees=[]
        allocator=NS(alloc=lambda n: None if fail else torch.tensor([3,4]),
                     free=lambda x:frees.extend(x.tolist()))
        tree=NS(get_component_device_value=lambda *args:torch.tensor([3]))
        comp=NS(cache=NS(req_to_token_pool=NS(mamba_allocator=allocator),
                        evict_for_alloc=lambda x:None), tree_core=tree,component_type='mamba')
        req=NS(kv=NS(holds_mamba=holds,mamba_pool_idx=7 if holds else None,
                     mamba_cow_src_index=None,mamba_needs_clear=True))
        return comp,req,frees

    def test_allocation_and_abort_preserve_request(self):
        comp,req,frees=self.component()
        x=ns['build_external_linker_transfer'](comp,LinkPhase.LOAD,None,['a','b'])
        self.assertEqual(x.keys,['b'])
        self.assertEqual(x.device_indices.tolist(),[3,4])
        ns['update_external_linker_load'](comp,Phase.ABORT,req,None,x,512)
        self.assertEqual(frees,[3,4]);self.assertIsNone(req.kv.mamba_pool_idx)

    def test_commit_private_state_or_canonical_cow(self):
        for holds in (False,True):
            comp,req,frees=self.component(holds)
            x=ns['build_external_linker_transfer'](comp,LinkPhase.LOAD,None,['a'])
            ns['update_external_linker_load'](comp,Phase.COMMIT,req,None,x,256,
                                             insert_result=NS(last_device_node=8))
            self.assertEqual(req.kv.mamba_pool_idx,7 if holds else 4)
            self.assertEqual(frees,[4] if holds else [])
            self.assertFalse(req.kv.mamba_needs_clear)
            if holds:self.assertEqual(req.kv.mamba_cow_src_index.tolist(),[3])

    def test_failed_allocation_returns_miss(self):
        comp,req,frees=self.component(fail=True)
        self.assertIsNone(ns['build_external_linker_transfer'](comp,LinkPhase.LOAD,None,['a']))
        self.assertEqual(frees,[])


class ProducerFenceTests(unittest.TestCase):
    def run_stash(self, *, overlap=True, external=True, mamba=True):
        events = []
        state = {'checkpoint': 'old'}
        class Stream:
            def __init__(self): self.pending = []
            def synchronize(self):
                while self.pending: self.pending.pop(0)()
            def wait_stream(self, other):
                events.append('fence')
                self.pending.append(other.synchronize)
        producer, schedule = Stream(), Stream()
        producer.pending.append(lambda: state.update(checkpoint='completed'))
        scheduler = NS(enable_overlap=overlap, enable_unified_cache_external_linker=external,
                       schedule_stream=schedule, forward_stream=producer, tree_cache=object())
        req = NS(kv=NS(holds_mamba=mamba))
        def cache(request, tree, *, chunked):
            self.assertIs(request, req)
            self.assertIs(tree, scheduler.tree_cache)
            self.assertTrue(chunked)
            events.append('stash')
            # Existing offload ready_event fences the schedule stream only.
            # Its worker must inherit a dependency on the producer's writes.
            schedule.synchronize()
            events.append(state['checkpoint'])
        namespace = {'maybe_cache_unfinished_req': cache}
        definitions(SRT/'managers/scheduler.py', {'stash_chunked_request'}, namespace, 'Scheduler')
        namespace['stash_chunked_request'](scheduler, req)
        return events

    def test_external_kda_stash_fences_before_offload_event(self):
        self.assertEqual(self.run_stash(), ['fence', 'stash', 'completed'])

    def test_unrelated_stash_paths_do_not_add_a_fence(self):
        for kwargs in ({'overlap': False}, {'external': False}, {'mamba': False}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.run_stash(**kwargs), ['stash', 'old'])


# Exercise the actual wrapper with KDA allocations, including every-rank failure
# agreement and an already-present canonical checkpoint. No GPU/model import.
Comp = Enum('Comp', 'FULL MAMBA')
wns = dict(ns, NamedTuple=NamedTuple, ComponentType=Comp,
           logger=logging.getLogger('glm-mooncake-test'),
           InsertParams=lambda **kw:NS(**kw))
definitions(SRT/'mem_cache/unified_cache/unified_cache_linker.py',
            {'ExternalCacheHitMarker','_PendingRestore','UnifiedCacheLinkerWrapper'},wns)
Wrapper=wns['UnifiedCacheLinkerWrapper']
Marker=wns['ExternalCacheHitMarker']
definitions(SRT/'mem_cache/unified_cache/components/full.py',
            {'update_external_linker_load'},wns,'FullComponent')


class Votes:
    def __init__(self):
        self.barrier=threading.Barrier(4,timeout=10); self.values=[None]*4
    def rank(self,rank):
        def reduce(state,op):
            self.values[rank]=state.clone(); self.barrier.wait()
            state.copy_(torch.stack(self.values).min(0).values); self.barrier.wait()
        return reduce


def wrapper_fixture(*, success=True, duplicate=False, holds=False, vote=None, allocation=True,
                    gate=None):
    events=[]; live=set(); freed=[]
    def alloc(n):
        if not allocation:return None
        ids=torch.tensor([3,4]);live.update(ids.tolist());return ids
    def free(ids):
        for i in ids.tolist():
            assert i in live, 'KDA double free'
            live.remove(i);freed.append(i)
    allocator=NS(alloc=alloc,free=free)
    tree=NS(empty_match_result=NS(device_indices=torch.empty(0,dtype=torch.int64)),
            get_component_device_value=lambda *args:torch.tensor([9 if duplicate else 3]),
            collect_full_device_indices=lambda *args:torch.arange(64,320),
            mark_external_cache_stored_path=lambda *args:events.append('stored'))
    cache=NS(page_size=256,tree_core=tree,req_to_token_pool=NS(mamba_allocator=allocator),
             _all_reduce_attn_groups=vote or (lambda *args:None),evict_for_alloc=lambda x:None)
    def insert(params):
        events.append('publish')
        assert params.mamba_value.tolist()==[3]
        return NS(last_device_node=8,mamba_exist=duplicate,adopted_ranges={Comp.FULL:[(0,256)]})
    cache.insert=insert
    full_alloc=NS(free=lambda ids:events.append('free_full'))
    full=NS(component_type=Comp.FULL,_full_allocator=lambda:full_alloc,
            build_external_linker_transfer=lambda *args:Transfer('kv',['a'],torch.arange(64,320)))
    full.update_external_linker_load=MethodType(wns['update_external_linker_load'],full)
    mamba=NS(cache=cache,tree_core=tree,component_type=Comp.MAMBA)
    for method in ('build_external_linker_transfer','update_external_linker_load'):
        setattr(mamba,method,MethodType(ns[method],mamba))
    worker=ThreadPoolExecutor(max_workers=1)
    def restore(rid,transfers):
        if gate is not None:gate.wait(10)
        events.append('restored');return success
    def submit_restore(rid,transfers):
        events.append('submitted');return worker.submit(restore,rid,transfers)
    backend=NS(restores_before_publish=True,submit_restore=submit_restore,
               lookup=lambda *args:events.append('lookup'),cancel_queued_load=lambda rid:False)
    wrapper=Wrapper.__new__(Wrapper)
    wrapper.cache=cache;wrapper.cache_linker=backend
    wrapper._components=(full,mamba);wrapper._skip_swa=False
    wrapper.hit_markers={'r':Marker(list(range(256)),['a'],0,(256,0,1))}
    wrapper.failed_loads=set();wrapper.pending_loads={};wrapper.pending_offloads=[]
    wrapper.restores={}
    req=NS(rid='r',last_node=0,prefix_indices=torch.empty(0,dtype=torch.int64),priority=0,
           host_hit_length=256,swa_host_hit_length=0,mamba_host_hit_length=1,
           kv=NS(holds_mamba=holds,mamba_pool_idx=7 if holds else None,
                 mamba_cow_src_index=None,mamba_needs_clear=True,swa_evicted_seqlen=0))
    return NS(wrapper=wrapper,req=req,events=events,live=live,freed=freed,worker=worker)


def settle(wrapper):
    """One scheduler step of check_hicache_events, restore part only."""
    wrapper.settle_restores(wrapper.num_settled_restores())


def restore_and_publish(f):
    """Submit, let the DMA finish, settle, then publish, as the scheduler does."""
    assert f.wrapper.load_back(f.req) is None
    f.worker.submit(lambda: None).result()
    settle(f.wrapper)
    return f.wrapper.load_back(f.req)


MatchResult = NamedTuple('MatchResult', [('device_indices', object), ('best_match_node', int),
    ('last_host_node', int), ('host_hit_length', int), ('swa_host_hit_length', int),
    ('mamba_host_hit_length', int)])


def match(f, device_hit_len):
    key = type('Key', (), {'maybe_to_bigram_view': lambda self, eagle: (self, None),
                           '__len__': lambda self: 512})()
    f.wrapper.cache.tree_core.is_eagle = False
    result = MatchResult(torch.arange(device_hit_len), 5, 0, 0, 0, 0)
    return f.wrapper.match(key, f.req, result)


class PublicationTests(unittest.TestCase):
    def test_checkpoint_duplicate_and_existing_request_ownership(self):
        for duplicate in (False,True):
            for holds in (False,True):
                f=wrapper_fixture(duplicate=duplicate,holds=holds)
                result=restore_and_publish(f)
                self.assertEqual(result[0].numel(),256)
                self.assertLess(f.events.index('restored'),f.events.index('publish'))
                self.assertEqual(f.freed,([3] if duplicate else [])+([4] if holds else []))
                self.assertEqual(f.req.kv.mamba_pool_idx,7 if holds else 4)
                self.assertEqual(f.req.kv.mamba_cow_src_index.tolist(),[9 if duplicate else 3])
                self.assertFalse(f.wrapper.pending_loads)

    def test_failure_on_one_rank_rolls_back_all_checkpoints_before_publication(self):
        for failure in ('storage','allocation'):
            votes=Votes()
            ranks=[wrapper_fixture(success=not(failure=='storage' and rank==2),
                                   allocation=not(failure=='allocation' and rank==2),
                                   vote=votes.rank(rank)) for rank in range(4)]
            def rank(f):
                first=f.wrapper.load_back(f.req)
                if first is not None:return first  # allocation vote failed
                f.worker.submit(lambda:None).result()
                settle(f.wrapper)
                return f.wrapper.load_back(f.req)
            with ThreadPoolExecutor(max_workers=4) as workers:
                results=list(workers.map(rank,ranks))
            for f,result in zip(ranks,results):
                self.assertEqual(result[0].numel(),0)
                self.assertFalse(f.live)
                self.assertNotIn('publish',f.events)
                self.assertIsNone(f.req.kv.mamba_pool_idx)
                self.assertEqual(f.req.host_hit_length,0)
                self.assertIn('r',f.wrapper.failed_loads)
                if failure=='allocation':self.assertNotIn('restored',f.events)

    def test_restore_runs_off_the_scheduler_and_request_waits_in_queue(self):
        gate=threading.Event()
        f=wrapper_fixture(gate=gate)
        self.assertIsNone(f.wrapper.load_back(f.req))
        self.assertEqual(f.live,{3,4})  # private destinations reserved
        # The scheduler keeps stepping while the DMA is in flight: nothing
        # settles, admission keeps the promise, and the store is not re-probed.
        for _ in range(3):
            settle(f.wrapper)
            self.assertEqual(match(f,0).host_hit_length,256)
            self.assertIsNone(f.wrapper.load_back(f.req))
        self.assertNotIn('restored',f.events)
        self.assertNotIn('lookup',f.events)
        gate.set()
        f.worker.submit(lambda:None).result()
        settle(f.wrapper)
        self.assertEqual(f.wrapper.load_back(f.req)[0].numel(),256)
        self.assertEqual(f.events[:3],['submitted','restored','publish'])
        self.assertFalse(f.wrapper.restores)

    def test_settlement_waits_for_the_slowest_rank_in_fifo_order(self):
        votes=Votes(); gates=[threading.Event() for _ in range(4)]
        ranks=[wrapper_fixture(vote=votes.rank(r),gate=gates[r]) for r in range(4)]
        for f in ranks:
            f.wrapper.hit_markers['s']=f.wrapper.hit_markers['r']
        def submit(f):
            first=f.wrapper.load_back(f.req)
            second=f.wrapper.load_back(NS(**{**vars(f.req),'rid':'s'}))
            return first,second
        with ThreadPoolExecutor(max_workers=4) as workers:
            self.assertEqual(list(workers.map(submit,ranks)),[(None,None)]*4)
        for r in (0,1,2):gates[r].set()
        for r in (0,1,2):ranks[r].worker.submit(lambda:None).result()
        counts=[f.wrapper.num_settled_restores() for f in ranks]
        self.assertEqual(counts,[2,2,2,0])
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lambda f:f.wrapper.settle_restores(min(counts)),ranks))
        self.assertTrue(all(p.outcome is None for f in ranks for p in f.wrapper.restores.values()))
        gates[3].set();ranks[3].worker.submit(lambda:None).result()
        counts=[f.wrapper.num_settled_restores() for f in ranks]
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lambda f:f.wrapper.settle_restores(min(counts)),ranks))
        self.assertEqual(min(counts),2)
        self.assertTrue(all(p.outcome for f in ranks for p in f.wrapper.restores.values()))

    def test_abort_or_prefix_change_frees_slots_only_after_dma(self):
        for orphan in ('release','prefix'):
            with self.subTest(orphan=orphan):
                gate=threading.Event()
                f=wrapper_fixture(gate=gate)
                self.assertIsNone(f.wrapper.load_back(f.req))
                if orphan=='release':
                    f.wrapper.release_request('r')
                else:
                    self.assertEqual(match(f,256).host_hit_length,0)
                    self.assertIn('r',f.wrapper.failed_loads)
                settle(f.wrapper)
                self.assertEqual(f.live,{3,4})  # still under DMA: not freed
                self.assertNotIn('free_full',f.events)
                gate.set();f.worker.submit(lambda:None).result()
                settle(f.wrapper)
                self.assertFalse(f.live)
                self.assertIn('free_full',f.events)
                self.assertNotIn('publish',f.events)
                self.assertFalse(f.wrapper.restores)

    def test_orphan_after_settlement_frees_immediately(self):
        for orphan in ('release','prefix'):
            with self.subTest(orphan=orphan):
                f=wrapper_fixture()
                self.assertIsNone(f.wrapper.load_back(f.req))
                f.worker.submit(lambda:None).result()
                settle(f.wrapper)
                if orphan=='release':f.wrapper.release_request('r')
                else:match(f,256)
                self.assertFalse(f.live)
                self.assertNotIn('publish',f.events)
                self.assertFalse(f.wrapper.restores)


log=logging.getLogger("glm-mooncake-backend-test")
log.addHandler(logging.NullHandler());log.propagate=False
device = NS(current_stream=lambda: NS(synchronize=lambda: None))
backend_ns = dict(
    torch=torch, logger=log, UnifiedCacheLinker=object, device_module=device,
    hashlib=hashlib, json=json, os=os, time=time,
)
definitions(
    SRT / "mem_cache/storage/mooncake_store/mooncake_direct_linker.py",
    {"MooncakeDirectLinker"},
    backend_ns,
)
Mooncake = backend_ns["MooncakeDirectLinker"]


def backend(results):
    instance = Mooncake.__new__(Mooncake)
    instance.stats = {"load": 0}
    transfers = [
        NS(name=name, keys=[name], host_indices=torch.arange(2))
        for name in ("kv", "indexer", "mamba")
    ]
    instance.pool_group = NS(resolve_transfers=lambda _: transfers)
    instance.pools = {
        name: NS(get_page_buffer_meta=lambda _: ([100, 200], [8, 8]))
        for name in ("kv", "indexer", "mamba")
    }
    calls = []

    def get(keys, pointers, sizes):
        calls.append(keys[0])
        result = results[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    instance.storage = NS(
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
        _pack_multi_buffer_meta=lambda keys, pointers, sizes: ([pointers], [sizes]),
        _get_batch_zero_copy_impl=get,
    )
    return instance, calls


class BackendTests(unittest.TestCase):
    def test_byte_diagnostic_hashes_exact_strided_substorage_spans(self):
        backing = torch.arange(2 * 8 * 5, dtype=torch.int32).reshape(2, 8, 5)
        entry = Entry(name='mamba', indices_from_pool='mamba', device_pool=None,
                      components=[[backing[i, :, 1:4] for i in range(2)]],
                      layer_mapping={1: 0, 2: 1}, page_size=1, rows_are_pages=True)
        ids = torch.tensor([3])
        transfer = Transfer('mamba', ['opaque-key'], host_indices=ids)
        b = Mooncake.__new__(Mooncake)
        b.verify_bytes = True
        b.pools = {'mamba': entry}
        b.storage = NS(_get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
                       _tag_keys=lambda keys: keys,
                       _pack_multi_buffer_meta=lambda keys, ptrs, sizes: ([ptrs], [sizes]))
        expected = hashlib.sha256(b''.join(gather(entry, ids))).hexdigest()
        with self.assertLogs(log, level='INFO') as capture:
            b._diagnostic_transfer_bytes('restore_after', transfer)
        record = json.loads(capture.records[0].getMessage().split(': ', 1)[1])
        self.assertEqual(record['sha256'], expected)
        self.assertEqual(record['bytes'], 24)
        self.assertEqual(record['segments'], 2)
        self.assertEqual(record['indices'], 1)
        self.assertEqual(record['key_sha256'], hashlib.sha256(b'opaque-key').hexdigest())
        self.assertNotIn('opaque-key', capture.records[0].getMessage())
        b.verify_bytes = False
        b._diagnostic_transfer_bytes('unused', None)

    def test_byte_diagnostic_rejects_outside_registered_storage(self):
        b, _ = backend([])
        b.verify_bytes = True
        buffer = torch.zeros(8, dtype=torch.uint8)
        b.pools['kv'] = NS(get_hybrid_pool_buffer=lambda: [buffer],
                          get_page_buffer_meta=lambda _: ([buffer.data_ptr() + 7], [2]))
        with self.assertRaisesRegex(ValueError, 'outside registered storage'):
            b._diagnostic_transfer_bytes('restore_after', Transfer('kv', ['a'], host_indices=torch.tensor([0])))

    def test_byte_diagnostic_rejects_incomplete_metadata(self):
        b, _ = backend([])
        b.verify_bytes = True
        b.pools['kv'].get_hybrid_pool_buffer = lambda: []
        b.pools['kv'].get_page_buffer_meta = lambda _: ([100], [])
        with self.assertRaisesRegex(ValueError, 'object metadata cardinality'):
            b._diagnostic_transfer_bytes('restore_after', Transfer('kv', ['a'], host_indices=torch.tensor([0])))

    def test_duplicate_remote_keys_fail_before_io(self):
        b, calls = backend([[16]])
        b.storage._get_hybrid_page_component_keys = lambda keys, transfer: (["same", "same"], 1)
        self.assertFalse(b.restore("r", []))
        self.assertEqual(calls, [])

    def test_one_remote_checkpoint_then_private_cow_with_keyed_backend(self):
        # Model Mooncake RealClient::batch_get_into_multi_buffers_internal:
        # destinations are an unordered_map keyed by object name. Duplicate
        # keys overwrite the first destination but both byte counts succeed.
        buffers = [torch.full((2, 8, width), -1, dtype=dtype)
                   for width, dtype in ((13, torch.float32), (5, torch.bfloat16),
                                        (7, torch.bfloat16))]
        for i, buf in enumerate(buffers):
            buf[:, 1] = torch.arange(buf[:, 1].numel()).reshape_as(buf[:, 1]) + i * 100
        state = Entry(name='mamba', indices_from_pool='mamba', device_pool=None,
                      components=[[buf[i] for i in range(2)] for buf in buffers],
                      layer_mapping={1: 0, 2: 1}, page_size=1, rows_are_pages=True,
                      packed=False)
        full = LayoutTests().pool()
        group = Group([full, state], 3, 256, rank_replicated=False)
        objects = gather(state, torch.tensor([1]))
        suffixes = ['temporal', 'conv0', 'conv1']
        saved = {'a_' + suffix: b''.join(objects[i * 2:i * 2 + 2])
                 for i, suffix in enumerate(suffixes)}
        calls = []
        def get(keys, pointers, sizes):
            calls.append(list(keys))
            by_key = dict(zip(keys, zip(pointers, sizes)))
            for key, (ptrs, lens) in by_key.items():
                offset = 0
                for ptr, size in zip(ptrs, lens):
                    ctypes.memmove(ptr, saved[key][offset:offset + size], size)
                    offset += size
            return [sum(lens) for lens in sizes]
        def pack(keys, pointers, sizes):
            n = len(pointers) // len(keys)
            return ([pointers[i:i+n] for i in range(0, len(pointers), n)],
                    [sizes[i:i+n] for i in range(0, len(sizes), n)])
        def keys(page_keys, transfer):
            return ([key + '_' + suffix for key in page_keys for suffix in suffixes], 3)

        # The formerly used two identical endpoints produce successful byte
        # receipts while leaving the first checkpoint stale.
        duplicate_keys, _ = keys(['a', 'a'], None)
        pointers, sizes = state.get_page_buffer_meta(torch.tensor([3, 4]))
        pointers, sizes = pack(duplicate_keys, pointers, sizes)
        self.assertEqual(get(duplicate_keys, pointers, sizes), [sum(x) for x in sizes])
        self.assertTrue(all(torch.all(buf[:, 3] == -1) for buf in buffers))
        self.assertTrue(all(torch.equal(buf[:, 4], buf[:, 1]) for buf in buffers))
        for buf in buffers: buf[:, 4] = -1
        calls.clear()

        comp, req, frees = StateTests().component()
        transfer = ns['build_external_linker_transfer'](comp, LinkPhase.LOAD, None, ['a'])
        self.assertEqual(transfer.device_indices.tolist(), [3, 4])
        expanded = group.resolve_transfers([Transfer('kv', ['a'], physical_indices([1, 2, 3, 4])), transfer])
        remote = next(t for t in expanded if t.name == 'mamba')
        self.assertEqual(remote.keys, ['a'])
        self.assertEqual(remote.host_indices.tolist(), [3])
        b = Mooncake.__new__(Mooncake)
        b.stats = {'load': 0}
        b.pool_group = NS(resolve_transfers=lambda _: [remote])
        b.pools = {'mamba': state}
        b.storage = NS(_get_hybrid_page_component_keys=keys, _tag_keys=lambda x: x,
                       _pack_multi_buffer_meta=pack, _get_batch_zero_copy_impl=get)
        self.assertTrue(b.restore('r', [transfer]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), len(set(calls[0])))
        self.assertTrue(all(torch.equal(buf[:, 3], buf[:, 1]) for buf in buffers))
        self.assertTrue(all(torch.all(buf[:, 4] == -1) for buf in buffers))
        ns['update_external_linker_load'](comp, Phase.COMMIT, req, None, transfer, 256,
                                        insert_result=NS(last_device_node=8))
        self.assertEqual(req.kv.mamba_cow_src_index.tolist(), [3])
        self.assertEqual(int(req.kv.mamba_pool_idx), 4)
        for buf in buffers:
            buf[:, req.kv.mamba_pool_idx] = buf[:, req.kv.mamba_cow_src_index[0]]
            self.assertTrue(torch.equal(buf[:, 4], buf[:, 1]))
        self.assertEqual(frees, [])

    def test_exact_byte_counts_all_pools(self):
        b, calls = backend([[16], [16], [16]])
        self.assertTrue(b.restore("r", []))
        self.assertEqual(calls, ["kv", "indexer", "mamba"])
        self.assertEqual(b.stats["load"], 1)

    def test_long_disk_restore_is_bounded_and_failure_never_publishes(self):
        # The real failed workload had 65 packed DSA objects of 11 MiB per
        # rank. Four simultaneous requests exceed the shared 1 GiB arena.
        for fail_batch in (None, 2):
            b, _ = backend([[16], [16], [16]])
            keys = [f"page-{i}" for i in range(65)]
            sizes = [[5 * 1024 * 1024, 6 * 1024 * 1024] for _ in keys]
            pointers = [[100 + i, 200 + i] for i in range(len(keys))]
            transfer = NS(name="kv", keys=keys, host_indices=torch.arange(65))
            b.pool_group.resolve_transfers = lambda _: [transfer]
            b.pools["kv"].get_page_buffer_meta = lambda _: (pointers, sizes)
            calls = []

            def get(batch_keys, batch_ptrs, batch_sizes):
                calls.append(list(batch_keys))
                self.assertEqual(len(batch_keys), len(batch_ptrs))
                self.assertEqual(len(batch_keys), len(batch_sizes))
                expected = [sum(size) for size in batch_sizes]
                self.assertLessEqual(sum(expected), 128 * 1024 * 1024)
                if len(calls) == fail_batch:
                    expected[-1] -= 1
                return expected

            b.storage._get_batch_zero_copy_impl = get
            self.assertEqual(b.restore("r", [transfer]), fail_batch is None)
            self.assertEqual(b.stats["load"], int(fail_batch is None))
            if fail_batch is None:
                self.assertEqual([key for batch in calls for key in batch], keys)
                self.assertGreater(len(calls), 1)
            else:
                self.assertEqual(len(calls), fail_batch)

    def test_partial_restore_and_all_error_forms_are_misses(self):
        for result in (
            [-600],
            [8],
            [32],
            [],
            None,
            -600,
            RuntimeError("transport failed"),
        ):
            with self.subTest(result=result):
                b, calls = backend([[16], [16], result])
                self.assertFalse(b.restore("r", []))
                self.assertEqual(calls, ["kv", "indexer", "mamba"])
                self.assertEqual(b.stats["load"], 0)

    def test_device_error_is_not_swallowed_as_storage_miss(self):
        def sync_error():
            raise RuntimeError("device failure")

        b, calls = backend([[16], [16], [16]])
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            b._restore_when_ready(NS(synchronize=sync_error), "r", [])
        self.assertEqual(calls, [])

    def test_submitted_restores_run_on_a_worker_in_order_after_the_fence(self):
        b, calls = backend([[16]] * 6)
        b.restore_executor = ThreadPoolExecutor(max_workers=1)
        order = []
        threads = set()
        original = b.restore
        def restore(rid, transfers):
            threads.add(threading.get_ident())
            order.append(rid)
            return original(rid, transfers)
        b.restore = restore
        fences = []
        device.Event = lambda: NS(record=lambda: fences.append('record'),
                                  synchronize=lambda: fences.append('wait'))
        futures = [b.submit_restore(rid, []) for rid in ("a", "b")]
        self.assertEqual([f.result() for f in futures], [True, True])
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(sorted(fences), ['record', 'record', 'wait', 'wait'])
        self.assertEqual(fences[0], 'record')
        self.assertNotIn(threading.get_ident(), threads)
        b.restore_executor.shutdown()


trigger_ns=dict(ComponentType=Comp)
definitions(SRT/'mem_cache/unified_cache/unified_tree_core.py',
            {'_should_backup_after_insert'},trigger_ns,'UnifiedTreeCore')


class ChunkCheckpointTests(unittest.TestCase):
    def fixture(self, *, complete=True, stored=False, pending=None, restored=False):
        node=NS(component_data={Comp.MAMBA:NS(value=torch.tensor([3]) if complete else None)},
                external_cache_stored=stored,write_through_pending_id=pending,evicted=False)
        core=NS(enable_external_cache_linker=True,enable_hicache=False,is_write_back=False,root_node=object(),
                components_by_type={Comp.FULL:object(),Comp.MAMBA:object()},
                _needs_external_linker_offload=lambda n:not n.external_cache_stored and n.write_through_pending_id is None,
                _inc_hit_count_and_check=lambda node,chunked:not chunked)
        state=NS(is_new_leaf=True,target_node=node,
                 params=NS(chunked=True,track_adopted_ranges=restored))
        return core,state

    def test_donated_chunk_checkpoint_is_backed_up_without_second_hit(self):
        core,state=self.fixture()
        self.assertFalse(core._inc_hit_count_and_check(state.target_node,True))
        self.assertTrue(trigger_ns['_should_backup_after_insert'](core,state))

    def test_incomplete_pending_stored_and_restored_checkpoints_do_not_duplicate(self):
        for kwargs in ({'complete':False},{'stored':True},{'pending':7},{'restored':True}):
            core,state=self.fixture(**kwargs)
            self.assertFalse(trigger_ns['_should_backup_after_insert'](core,state))

    def test_root_evicted_and_writeback_do_not_enqueue(self):
        for kind in ('root','evicted','writeback'):
            core,state=self.fixture()
            if kind=='root':core.root_node=state.target_node
            elif kind=='evicted':state.target_node.evicted=True
            else:core.is_write_back=True
            self.assertFalse(trigger_ns['_should_backup_after_insert'](core,state))

    def test_plain_kv_and_normal_insert_trigger_unchanged(self):
        core,state=self.fixture();core.components_by_type={Comp.FULL:object()}
        self.assertFalse(trigger_ns['_should_backup_after_insert'](core,state))
        state.params.chunked=False
        self.assertTrue(trigger_ns['_should_backup_after_insert'](core,state))


if __name__ == '__main__': unittest.main()
