#!/usr/bin/env python3
"""CPU qualification against the actual patched wheel sources; no GPU calls."""
import argparse
import ast
import contextlib
from copy import deepcopy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch

# Set visibility before importing the ROCm-enabled torch package. Qualification
# must fail if a tested method tries to initialize or select any GPU.
for variable in ("ROCP_TOOL_LIBRARIES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ[variable] = ""
import torch

sys.dont_write_bytecode = True

UNCHANGED_SOURCE_SHA256 = {
    'srt/layers/sampler.py': '5ac36528a7de48a5796c376d3ee307e9ce1bb57d306eff70827f6fb169b58e5e',
    'srt/sampling/glm53_forced_decode_reference.py': 'd7c98eef773093a5f97175b0523d24cd7687dfa28c5c903c92cd9fa084b4d8d5',
    'srt/sampling/sampling_batch_info.py': '0818cb0896222bd062f1f227eebc83ec1fcb0c00f188469e4a7a620309130d3d',
}


@contextlib.contextmanager
def cpu_only():
    assert not torch.cuda.is_initialized()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("GPU initialization/selection is forbidden in CPU qualification")

    with contextlib.ExitStack() as stack:
        for name in ("init", "_lazy_init", "set_device", "device", "device_of", "_exchange_device", "_maybe_exchange_device", "Stream", "Event"):
            if hasattr(torch.cuda, name):
                stack.enter_context(patch.object(torch.cuda, name, forbidden))
        stack.enter_context(patch.object(torch, "set_default_device", forbidden))
        # Verify the guards themselves, then require the actual checks to make
        # zero forbidden calls, even if source code catches their exceptions.
        for operation in (torch.cuda.init, lambda: torch.cuda.set_device(0), lambda: torch.cuda.device(0),
                          lambda: torch.set_default_device("cuda"), lambda: torch.empty(1, device="cuda")):
            fails(operation)
        assert len(calls) == 5
        calls.clear()
        yield
        assert not calls and not torch.cuda.is_initialized()


def repository_checker():
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "lib/bench/glm53-prefill-geometry-trace-check.py"
        if candidate.is_file():
            return candidate
    raise ValueError("Standalone execution requires --checker with the immutable checker path")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fails(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError("Invalid geometry accepted")


def method(path, cls, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    execute([fn], path, namespace)
    return namespace[name]


def execute(nodes, path, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def req(name="scores-NORMAL-warm-0", slot=1, length=1024):
    return N(rid="glm53-serial-prefix-control-v1-" + name, origin_input_ids=[3] * length,
        extra_key=None, cache_salt=None, _glm53_geometry_match=None,
        kv=N(req_pool_idx=slot, mamba_pool_idx=torch.tensor(slot * 3 - 2),
            mamba_ping_pong_track_buffer=torch.tensor([slot * 3 - 1, slot * 3]),
            mamba_cow_src_index=None, mamba_needs_clear=True,
            mamba_last_track_seqlen=length, mamba_last_track_idx=0))


def batch(reqs):
    return N(reqs=reqs, enable_overlap=True, decoding_reqs=None,
        spec_algorithm=N(is_none=lambda: True),
        model_config=N(hf_text_config=N(model_type="glm5_next_text")))


def runner():
    return N(is_draft_worker=False, spec_algorithm=N(is_none=lambda: True),
        model_config=N(hf_text_config=N(model_type="glm5_next_text")),
        ps=N(tp_rank=0, pp_size=1), req_to_token_pool=N(mamba_ckpt_pool=None,
            enable_mamba_extra_buffer_lazy=False, mamba_v2p_table=None,
            mamba_translate_is_fusable=True, get_mamba_indices=lambda rows:rows*3-2))


def forward(mode="EXTEND", lengths=(1024,)):
    return N(forward_mode=N(is_extend=lambda: mode == "EXTEND", is_decode=lambda: mode == "DECODE"),
        glm53_geometry_packet=None, seq_lens_cpu=torch.tensor(lengths),
        input_ids=torch.ones(sum(lengths), dtype=torch.long), batch_size=len(lengths),
        seq_lens=torch.tensor(lengths), req_pool_indices=torch.arange(1, len(lengths)+1),
        mamba_track_indices=torch.tensor([3*i+2 for i in range(len(lengths))]),
        mamba_track_mask=torch.tensor([True]*len(lengths)),
        mamba_track_seqlens=torch.tensor(lengths), mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=torch.tensor([3*i+1 for i in range(len(lengths))]))


def prepare(trace, b, chunks=None, lengths=None, prefixes=None, encoded=None, masks=None):
    lengths = lengths or [len(r.origin_input_ids) for r in b.reqs]
    prefixes = prefixes or [0] * len(b.reqs)
    chunks = chunks or [[3] * (n - p) for n, p in zip(lengths, prefixes)]
    b.glm53_geometry_prepared = trace.snapshot_extend(b, chunks, lengths, prefixes,
        [len(x) for x in chunks], encoded or lengths, masks or [True] * len(b.reqs), enabled=True)
    return b.glm53_geometry_prepared


def settle(trace, packet):
    packet.copy_to_host(lambda tensor: tensor.clone())
    trace.complete(N(glm53_geometry_packet=packet))


def result_class(source, events):
    path = source / "srt/managers/utils.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GenerationBatchResult")
    def copy(tensor):
        events.append("D2H")
        return tensor.clone()
    namespace = dict(torch=torch, dataclasses=dataclasses, _async_d2h=copy)
    execute([cls], path, namespace)
    return namespace[cls.name]


def metadata_and_lifecycle(source, trace):
    trace.ledger = trace.Ledger()
    records = []
    with patch.object(trace.logger, "info", side_effect=records.append):
        trace.reset_completed(0)
        r = req()
        trace.cache_match(r, r.origin_input_ids, 0, True)
        peer = req("scores-NORMAL-warm-1", slot=2, length=72)
        peer.rid = "unmarked-peer"
        peer.kv.mamba_last_track_seqlen = None
        b = batch([r, peer])
        chunks = [list(r.origin_input_ids), list(peer.origin_input_ids)]
        frozen = prepare(trace, b, chunks=chunks, encoded=[1024, -1], masks=[True, False])
        chunks[0][0] = 99
        r.origin_input_ids[0] = 99
        assert frozen.peers[0].origin_sha256 == frozen.peers[0].chunk_sha256 == trace.ids_sha([3] * 1024)
        assert not frozen.peers[1].marked and len(frozen.peers) == 2
        fb = forward(lengths=(1024, 72))
        fb.mamba_track_mask[1] = False; fb.mamba_track_seqlens[1] = -1
        packet = trace.start(b, fb, runner(), enabled=True)
        # Existing eager preparation can pad the executed shape independently.
        fb.input_ids = torch.ones(1100, dtype=torch.long)
        fb.batch_size = 4
        trace.after_padding(fb)
        trace.dispatch(fb, "eager")
        # Donation happens before completion: only the eventual completion can emit it.
        trace.cache_donation(r, [3] * 1024, 1024, 0, True)
        assert trace.ledger.pending and not any('"kind":"donation"' in x for x in records)
        fails(trace.before_reset)
        # Buffer replacement after the snapshot cannot alter saved source slot labels.
        r.kv.mamba_ping_pong_track_buffer[0] = 22
        settle(trace, packet)
        assert packet.slots_host.tolist()[0][:5] == [1, 2, 3, -1, 2]
        assert any('"kind":"donation"' in x for x in records)
        assert packet.executed_M == 1100 and packet.executed_bs == 4
        fails(lambda: trace.complete(N(glm53_geometry_packet=packet)))
        fails(lambda: packet.copy_to_host(lambda t: t))

        r2 = req("scores-NORMAL-measure-1", slot=2, length=1096)
        r2.origin_input_ids = [3] * 1096
        r2.kv.mamba_last_track_seqlen = 1024
        r2.kv.mamba_last_track_idx = 1
        trace.cache_match(r2, r2.origin_input_ids, 1024, True)
        r2.kv.mamba_cow_src_index = torch.tensor([2])
        r2.kv.mamba_needs_clear = False
        b2 = batch([r2])
        prepare(trace, b2, lengths=[1096], prefixes=[1024], encoded=[-1], masks=[False])
        fb2 = forward(lengths=(72,))
        fb2.seq_lens = torch.tensor([1096]); fb2.req_pool_indices = torch.tensor([2])
        fb2.mamba_track_indices = torch.tensor([5]); fb2.mamba_track_mask = torch.tensor([False])
        fb2.mamba_track_seqlens = torch.tensor([-1]); fb2.mamba_clear_indices = None
        fb2.mamba_cow_src_indices = torch.tensor([2]); fb2.mamba_cow_dst_indices = torch.tensor([4])
        p2 = trace.start(b2, fb2, runner(), enabled=True)
        trace.after_padding(fb2); trace.dispatch(fb2, "eager")
        settle(trace, p2)
        assert p2.slots_host.tolist()[0][:6] == [4, 5, 6, 2, 5, 4]
        # The continuation no longer owns a COW operation after its consumption.
        r2.kv.mamba_cow_src_index = None
        prepared = prepare(trace, b2, lengths=[1096], prefixes=[1024], encoded=[-1], masks=[False])
        assert prepared.peers[0].cow_source is None
        trace.cache_finished(r); trace.cache_finished(r2)
        trace.before_reset(); trace.reset_completed()
        assert trace.ledger.epoch == 2 and not trace.ledger.cache
        fails(lambda: trace.cache_match(req("scores-A-warm-0"), [3]*1024, 1024, True))

    # Check immutable schema, fixed namespace, missing sources and unsupported pools.
    fails(lambda: trace.marked(N(rid=trace.PREFIX + "arbitrary")))
    fails(lambda: trace.ids_sha([True]))
    fails(lambda: trace.ids_sha([-1]))
    fails(lambda: trace.snapshot_extend(batch([req()]), [[3]], [1024], [0], [1], [1024], [True], enabled=True))
    r3 = req("full-B-measure-3")
    trace.ledger = trace.Ledger(); b3 = batch([r3]); prepare(trace, b3)
    for changes in ({"is_draft_worker": True}, {"spec_algorithm": N(is_none=lambda: False)},
                    {"ps": N(tp_rank=0, pp_size=2)}):
        run = runner()
        for key, value in changes.items(): setattr(run, key, value)
        fails(lambda: trace.start(b3, forward(), run, enabled=True))
    for key, value in (("mamba_ckpt_pool", object()), ("enable_mamba_extra_buffer_lazy", True),
                       ("mamba_v2p_table", object()), ("mamba_translate_is_fusable", False)):
        run = runner(); setattr(run.req_to_token_pool, key, value)
        fails(lambda: trace.start(b3, forward(), run, enabled=True))
    fails(lambda: trace.cache_donation(r3, [3]*1024, 1024, 0, True))
    b3.decoding_reqs=[r3]
    fails(lambda:prepare(trace,b3))
    b3.decoding_reqs=None
    for field,value in (("req_pool_indices",torch.tensor([9])), ("mamba_track_indices",torch.tensor([9])),
                        ("mamba_cow_src_indices",torch.tensor([9])), ("mamba_clear_indices",None)):
        trace.ledger=trace.Ledger(); fb3=forward(); setattr(fb3,field,value)
        packet3=trace.start(b3,fb3,runner(),enabled=True)
        trace.after_padding(fb3);trace.dispatch(fb3,"eager")
        fails(lambda:settle(trace,packet3))
    trace.ledger=trace.Ledger()
    decoded=forward("DECODE",(1025,));decoded.seq_lens_cpu=torch.empty(1,device="meta")
    b3.mamba_track_mask_cpu=[False]
    fails(lambda:trace.start(b3,decoded,runner(),enabled=True))
    trace.ledger.forward_count = trace.MAX_FORWARDS
    fails(lambda: trace.start(b3, forward(), runner(), enabled=True))
    trace.ledger = trace.Ledger(); trace.ledger.serial = trace.MAX_RECORDS
    fails(lambda: trace.ledger.emit("anything"))
    return records


def actual_transport(source, trace):
    events = []
    cls = result_class(source, events)
    trace.ledger = trace.Ledger()
    r = req("full-NORMAL-warm-0"); b = batch([r]); prepare(trace, b)
    fb = forward(); p = trace.start(b, fb, runner(), enabled=True)
    trace.after_padding(fb); trace.dispatch(fb, "eager")
    output = N(next_token_logprobs=None, input_token_logprobs=None,
        hidden_states=None, auxiliary_device_output=None, sampling_mask_output=None)
    result = cls(logits_output=output, next_token_ids=torch.tensor([1]), glm53_geometry_packet=p,
        copy_done=N(record=lambda: events.append("event-record"), synchronize=lambda: events.append("existing-wait")))
    result.copy_to_cpu(False, return_hidden_states=False)
    assert events == ["D2H", "D2H", "D2H", "event-record"] and not p.completed
    # Execute the actual processor prefix through existing completion/trace ordering.
    path = source / "srt/managers/scheduler_components/batch_result_processor.py"
    tree = ast.parse(path.read_text()); clsnode = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(isinstance(f, ast.FunctionDef) and f.name == "process_batch_result_prefill" for f in n.body))
    for name in ("process_batch_result_prefill", "process_batch_result_decode"):
        fn = next(n for n in clsnode.body if isinstance(n, ast.FunctionDef) and n.name == name)
        # Stop immediately after the real complete hook to avoid unrelated output work.
        body = fn.body
        if name.endswith("prefill"):
            generation = next(n for n in body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.is_generation")
            body = generation.body
        stop = next(i for i,n in enumerate(body) if isinstance(n, ast.Expr) and ast.unparse(n.value) == "glm53_geometry.complete(result)")
        selected = ast.FunctionDef(name="settle_actual", args=fn.args, body=body[:stop+1], decorator_list=[])
        namespace = dict(glm53_geometry=trace)
        execute([selected], path, namespace)
        if name.endswith("decode"):
            p.completed=False; trace.ledger.completed_count -= 1
        with patch.object(trace.logger, "info", side_effect=lambda line: events.append("trace")):
            namespace["settle_actual"](N(), N(), result)
        assert events[-2:] == ["existing-wait", "trace"]
    # No sync was added anywhere in the diagnostic or result copy.
    text = (source / "srt/managers/glm53_prefill_geometry_trace.py").read_text()
    assert ".synchronize(" not in text and ".item(" not in text and ".cpu(" not in text and "torch.cuda" not in text
    return events


def actual_source_order(source):
    schedule = (source / "srt/managers/schedule_batch.py").read_text()
    path = source / "srt/managers/schedule_batch.py"
    tree = ast.parse(schedule); clsnode = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScheduleBatch")
    fn = next(n for n in clsnode.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_for_extend")
    raw = ast.get_source_segment(schedule, fn)
    assert raw.index("glm53_geometry.snapshot_extend(") < raw.index("self._collect_deferred_mamba_cow_and_clear(reqs)")
    assert raw.index("mamba_track_seqlens_cpu.append(") < raw.index("glm53_geometry.snapshot_extend(")
    model = (source / "srt/model_executor/model_runner.py").read_text()
    tree = ast.parse(model); clsnode = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
    fn = next(n for n in clsnode.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_raw")
    raw = ast.get_source_segment(model, fn)
    assert raw.index("self._prepare_eager_forward_batch(forward_batch)") < raw.index("glm53_geometry.after_padding(forward_batch)") < raw.index("self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)")
    assert raw.index('glm53_geometry.dispatch(forward_batch, "prefill_graph")') < raw.index("self.prefill_cuda_graph_runner.execute(")
    cache = (source / "srt/mem_cache/mamba_radix_cache.py").read_text()
    assert cache.count("glm53_geometry.cache_donation(") == 2 and cache.count("glm53_geometry.cache_match(") == 1
    # Source-proven static identity mapping guard matches the actual pool API.
    pool = (source / "srt/mem_cache/memory_pool.py").read_text()
    assert "def mamba_translate_is_fusable" in pool and "def translate_mamba_indices" in pool
    assert "type(self).translate_mamba_indices\n            is HybridReqToTokenPool.translate_mamba_indices" in pool
    path=source/"srt/mem_cache/memory_pool.py"
    tree=ast.parse(pool);hybrid=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="HybridReqToTokenPool")
    funcs=[n for n in hybrid.body if isinstance(n,ast.FunctionDef) and n.name in
        ("mamba_v2p_table","mamba_translate_is_fusable","translate_mamba_indices")]
    namespace={}
    selected=ast.ClassDef(name="HybridReqToTokenPool",bases=[],keywords=[],body=funcs,decorator_list=[])
    execute([selected],path,namespace);static=namespace["HybridReqToTokenPool"]()
    assert static.mamba_v2p_table is None and static.mamba_translate_is_fusable
    tensor=torch.tensor([2,5]);assert static.translate_mamba_indices(tensor) is tensor
    class Moving(namespace["HybridReqToTokenPool"]):
        def translate_mamba_indices(self, indices):return indices+1
    assert not Moving().mamba_translate_is_fusable


def actual_worker_preservation(source, trace, rank=0):
    events=[]
    result=result_class(source, events)
    namespace={"torch":torch, "GenerationBatchResult":result, "glm53_geometry":trace,
        "capture_pre_sample_logits":lambda *args:None,
        "envs":N(SGLANG_ENABLE_DELAY_SAMPLE=N(get=lambda:False))}
    worker=method(source/"srt/managers/tp_worker.py", "TpModelWorker", "forward_batch_generation", namespace)
    raw_namespace={"contextlib":contextlib, "glm53_geometry":trace,
        "has_forward_context":lambda:True, "get_global_dwdp_manager":lambda:None,
        "EagerRunner":type("EagerRunner", (), {}),
        "ModelRunnerOutput":lambda **kw:N(**kw, expert_distribution_metrics=None,
            routed_experts_output=None, indexer_topk_output=None)}
    raw=method(source/"srt/model_executor/model_runner.py", "ModelRunner", "_forward_raw", raw_namespace)
    padding=method(source/"srt/model_executor/model_runner.py", "ModelRunner", "_prepare_eager_forward_batch", {})
    snapshots=[]
    for enabled in (False, True):
        trace.ENABLED=enabled; trace.ledger=trace.Ledger()
        r=req("scores-NORMAL-measure-1"); b=batch([r]); b.hicache_consumer_index=0
        prepare(trace, b)
        fb=forward(); fb.forward_mode.is_cpu_graph=lambda:False
        fb.forward_mode.is_split_prefill=lambda:False
        fb.forward_mode.is_extend=lambda **kwargs:True
        fb.apply_deprecated_skip_attn_backend_init=lambda value:None
        fb.is_prefill_only=False; fb.sampling_info=N(grammars=None)
        fb.global_num_tokens_cpu=None; fb.global_num_token_non_padded=None
        fb.prepare_attn_tp_scatter_input=lambda run:setattr(fb, "input_ids", torch.zeros(1028, dtype=torch.long))
        namespace["ForwardBatch"]=N(init_new=lambda *args, **kwargs:fb)
        run=runner(); run.device="cpu"; run.decode_cuda_graph_runner=None
        run.ps.tp_rank=rank
        run.hisparse_coordinator=None; run.prefill_cuda_graph_runner=None
        run.pp_group=N(is_last_rank=True)
        run._prepare_eager_forward_batch=lambda batch:padding(run, batch)
        # Only the COW/clear consumer is stubbed; its exact source ordering is checked separately.
        run._maybe_execute_deferred_mamba_cow_and_clear=lambda batch:events.append("COW-before-model")
        run.eager_runner=N(execute=lambda batch, **kwargs:N(next_token_logits=torch.randn(1,7)))
        run.forward=lambda batch, **kwargs:raw(run, batch, None)
        run.sample=lambda output, batch:torch.multinomial(output.next_token_logits.softmax(-1), 1).flatten()
        owner=N(model_runner=run, set_hicache_consumer=lambda value:None,
            is_dllm=lambda:False, pp_group=N(is_last_rank=True), enable_overlap=True, enable_spec=False)
        torch.manual_seed(993)
        value=worker(owner, b)
        snapshots.append((value.next_token_ids.clone(), value.logits_output.next_token_logits.clone(), torch.random.get_rng_state().clone()))
        assert (value.glm53_geometry_packet is not None) == enabled
        if enabled:
            assert value.glm53_geometry_packet.rank == rank
            assert value.glm53_geometry_packet.executed_M == 1028
            assert value.glm53_geometry_packet.dispatch == "eager"
    assert all(torch.equal(a,b) for a,b in zip(*snapshots))
    assert events == ["COW-before-model", "COW-before-model"]
    trace.ENABLED=True


def actual_rank_flow(source, trace):
    path=source/"srt/distributed/parallel_state.py";tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="initialize_model_parallel")
    assign=next(n for n in fn.body if isinstance(n,ast.AnnAssign) and ast.unparse(n.target)=="num_pipeline_model_parallel_groups")
    start=fn.body.index(assign)
    groups=next(n for n in fn.body[start:] if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="group_ranks")
    loop=next(n for n in fn.body[start:] if isinstance(n,ast.For) and ast.unparse(n.target)=="pp_group_idx")
    ns=dict(world_size=4,pipeline_model_parallel_size=1)
    execute([assign,groups,loop],path,ns)
    assert ns["group_ranks"]==[[0],[1],[2],[3]]
    last=method(path,"GroupCoordinator","is_last_rank",{})
    assert all(last.fget(N(rank=rank,last_rank=rank)) for rank in range(4))
    scheduler=(source/"srt/managers/scheduler.py").read_text()
    # Actual plain overlap path is unconditional by TP rank.
    tree=ast.parse(scheduler);clsnode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Scheduler")
    fn=next(n for n in clsnode.body if isinstance(n,ast.FunctionDef) and n.name=="run_batch")
    raw=ast.get_source_segment(scheduler,fn)
    assert "tp_rank" not in raw and "batch_result.copy_to_cpu(" in raw
    assert raw.index("glm53_geometry.guard_scheduler(self, batch)") < raw.index("self.model_worker.forward_batch_generation(")
    path=source/"srt/managers/scheduler_components/request_receiver.py"
    assert "recv_reqs = broadcast_pyobj(" in path.read_text()
    b=batch([req()]);b.forward_mode=N(is_extend=lambda:True,is_decode=lambda:False,is_split_prefill=lambda:False)
    owner=N(is_generation=True,enable_overlap=True,ps=N(pp_size=1),disaggregation_mode=N(value="null"))
    trace.guard_scheduler(owner,b)
    for key,value in (("enable_overlap",False),("is_generation",False),("ps",N(pp_size=2)),
                      ("disaggregation_mode",N(value="prefill"))):
        broken=deepcopy(owner);setattr(broken,key,value)
        fails(lambda:trace.guard_scheduler(broken,b))
    b.forward_mode.is_split_prefill=lambda:True
    fails(lambda:trace.guard_scheduler(owner,b))


def journal_loss_checks(records, checker_path):
    checker=load(checker_path, "geometry_actual_inventory_check")
    data=[json.loads(line.split("GLM53_PREFILL_GEOMETRY ",1)[1]) for line in records]
    rids=["glm53-serial-prefix-control-v1-scores-NORMAL-warm-0", "glm53-serial-prefix-control-v1-scores-NORMAL-measure-1"]
    receipt=checker.validate(data, rids, 0)
    assert receipt["requests"] == 2 and receipt["forwards"] == 2
    for damaged in (data[:-1], data[:1]+data[2:], data[:2]+data[1:]):
        fails(lambda:checker.validate(damaged, rids, 0))
    damaged=deepcopy(data); damaged[-1]["pending"]=1
    fails(lambda:checker.validate(damaged, rids, 0))
    damaged=deepcopy(data); next(row for row in damaged if row["kind"]=="completed")["peers"][0]["seq_len"]+=1
    fails(lambda:checker.validate(damaged, rids, 0))
    fails(lambda:checker.validate(data, rids[:1], 0))
    fails(lambda:checker.validate(data, rids, 1))


def actual_cache_methods(source, trace):
    """Run actual installed cache match, unfinished donation, finished donation."""
    trace.ledger=trace.Ledger()
    r=req("full-A-warm-0");r.output_ids=[];r.kv.cache_protected_len=0
    r.kv.holds_mamba=True;r.get_fill_ids=lambda:list(r.origin_input_ids)
    b=batch([r]);prepare(trace,b);fb=forward();p=trace.start(b,fb,runner(),enabled=True)
    trace.after_padding(fb);trace.dispatch(fb,"eager");settle(trace,p)
    r.last_node=N(parent=None, mamba_value=None)
    root=N(parent=None,mamba_value=None)
    node=N(parent=root,mamba_value=torch.tensor([2]))
    class Key:
        def __init__(self, ids, *args, **kwargs):self.ids=list(ids)
        def raw_token_ids(self):return self.ids
    ns=dict(torch=torch,glm53_geometry=trace,_MAMBA_DEBUG_ASSERTS=False,
        InsertParams=N,MatchPrefixParams=N,RadixKey=Key,MatchResult=N,get_last_access_time=lambda:1)
    path=source/"srt/mem_cache/mamba_radix_cache.py"
    unfinished=method(path,"MambaRadixCache","cache_unfinished_req",ns)
    finished=method(path,"MambaRadixCache","cache_finished_req",ns)
    match=method(path,"MambaRadixCache","_match_post_processor",ns)
    donate=method(source/"srt/mem_cache/memory_pool.py","HybridReqToTokenPool","donate_mamba_ping_pong_slot",dict(torch=torch,_MAMBA_DEBUG_ASSERTS=False))
    pool=N(req_to_token=torch.arange(2048).repeat(5,1),mamba_pool=N(),
        get_mamba_ping_pong_keep_idx=lambda req:req.kv.mamba_last_track_idx,
        set_mamba_ping_pong_slot=lambda req,idx,value:req.kv.mamba_ping_pong_track_buffer.__setitem__(idx,value),
        write=lambda *args:None,free_mamba_cache=lambda *args,**kwargs:None)
    pool.donate_mamba_ping_pong_slot=lambda req,slot:donate(pool,req,slot)
    def insert(params):
        node.mamba_value=params.mamba_value
        return N(prefix_len=0,mamba_exist=False)
    cache=N(disable=False,enable_mamba_extra_buffer=True,int8_ckpt_pool=None,page_size=1,
        req_to_token_pool=pool,token_to_kv_pool_allocator=N(free_segment=lambda *a,**kw:None),
        _alloc_mamba_slot=lambda:torch.tensor([50]),insert=insert,
        match_prefix=lambda params:N(device_indices=torch.arange(1024),last_device_node=node),
        dec_lock_ref=lambda *a:None,inc_lock_ref=lambda *a:None,
        full_lru_list=N(reset_node_and_parents_mru=lambda *a:None),
        mamba_lru_list=N(reset_node_mru=lambda *a:None),root_node=root,mamba_cache_chunk_size=512)
    unfinished(cache,r,chunked=True)
    assert r.kv.mamba_ping_pong_track_buffer.tolist()==[50,3] and r.kv.mamba_last_track_seqlen is None
    matched=req("full-A-measure-1",slot=2,length=1096);matched.kv.holds_mamba=True
    result=match(cache,N(cow_mamba=True,req=matched,key=Key([3]*1096)),[torch.arange(1024)],node,1)
    assert len(result.device_indices)==1024 and matched.kv.mamba_cow_src_index.tolist()==[2]
    assert matched._glm53_geometry_match is not None
    trace.cache_finished(r)
    # Finished donation selects its actual tracked keep slot before releasing the request.
    r2=req("full-B-warm-0",slot=2);r2.kv.cache_protected_len=0;r2.output_ids=[];r2.last_node=root
    b2=batch([r2]);prepare(trace,b2);fb2=forward()
    fb2.req_pool_indices=torch.tensor([2]);fb2.mamba_track_indices=torch.tensor([5]);fb2.mamba_clear_indices=torch.tensor([4])
    p2=trace.start(b2,fb2,runner(),enabled=True);trace.after_padding(fb2);trace.dispatch(fb2,"eager");settle(trace,p2)
    # Change the prefix to avoid the earlier cache entry while retaining the source hash.
    trace.ledger.cache.clear()
    finished(cache,r2,kv_len_to_handle=1024)
    assert trace.request_sha(r2) in trace.ledger.finished and not trace.ledger.pending


def qualify(source, baseline, checker_path):
    with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_GEOMETRY_TRACE":"1"}):
        trace=load(source/"srt/managers/glm53_prefill_geometry_trace.py", "actual_prefill_geometry")
    # Disabled/ordinary paths never inspect model, request, logits or device fields.
    before=torch.random.get_rng_state().clone()
    assert trace.snapshot_extend(object(), None, None, None, None, None, None, enabled=False) is None
    assert trace.start(object(), object(), object(), enabled=False) is None
    assert trace.start(batch([N(rid="ordinary")]), object(), object(), enabled=True) is None
    assert torch.equal(before, torch.random.get_rng_state())
    records=metadata_and_lifecycle(source, trace)
    actual_transport(source, trace)
    actual_source_order(source)
    for rank in range(4):actual_worker_preservation(source, trace,rank)
    actual_rank_flow(source,trace)
    journal_loss_checks(records, checker_path)
    actual_cache_methods(source, trace)
    unchanged=("srt/layers/sampler.py", "srt/sampling/glm53_forced_decode_reference.py", "srt/sampling/sampling_batch_info.py")
    assert all((source/name).read_bytes() == (baseline/name).read_bytes() for name in unchanged)
    assert all(hashlib.sha256((source/name).read_bytes()).hexdigest() == digest for name,digest in UNCHANGED_SOURCE_SHA256.items())
    for path in source.rglob("*.py"):
        if path.name in {"glm53_prefill_geometry_trace.py", "schedule_batch.py", "tp_worker.py", "utils.py", "model_runner.py", "forward_batch_info.py", "scheduler.py", "mamba_radix_cache.py", "batch_result_processor.py"}:
            ast.parse(path.read_text())
    source_files=("srt/managers/glm53_prefill_geometry_trace.py", "srt/managers/schedule_batch.py",
        "srt/managers/tp_worker.py", "srt/managers/utils.py", "srt/model_executor/model_runner.py",
        "srt/model_executor/forward_batch_info.py", "srt/managers/scheduler.py",
        "srt/managers/scheduler_components/batch_result_processor.py", "srt/mem_cache/mamba_radix_cache.py",
        "srt/mem_cache/memory_pool.py", "srt/distributed/parallel_state.py",
        "srt/managers/scheduler_components/request_receiver.py", *unchanged)
    return {"status":"PASS_CPU_PREFILL_GEOMETRY_TRACE", "baseline":str(baseline),
        "checker_sha256":hashlib.sha256(checker_path.read_bytes()).hexdigest(),
        "trace_sha256":hashlib.sha256((source/"srt/managers/glm53_prefill_geometry_trace.py").read_bytes()).hexdigest(),
        "source_sha256":{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in source_files},
        "unchanged_sampler_and_reference":list(unchanged),
        "checks":["disabled and ordinary early returns preserve RNG", "immutable all-peer CPU snapshots", "readonly pre-consumption slot snapshot",
            "actual result D2H before event record", "actual processor completion wait before trace", "padding before COW ordering",
            "deferred donation bound after producer completion", "COW match and reset lifecycle", "fixed namespace/budgets/unsupported pools fail closed",
            "actual worker and ModelRunner enabled/disabled tokens/logits/RNG identical", "missing/dropped/duplicate journal records and wrong rank fail closed",
            "actual static pool identity properties and cache match/unfinished/finished methods",
            "actual PP1 singleton groups and worker execution on all four TP ranks; unsupported scheduler modes refuse"],
        "lifecycle_records_tested":len(records), "gpu_initialization_and_selection_forbidden":True,
        "gpu_qualified":False, "state_contents_attested":False}


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True)
    p.add_argument("--baseline-source",type=Path)
    p.add_argument("--checker",type=Path,help="Immutable checker input; defaults to the repository checker for local review")
    p.add_argument("--patch",type=Path,default=Path(__file__).parents[1]/"patches/0063-glm53-prefill-geometry-trace.patch")
    args=p.parse_args();baseline=(args.baseline_source or args.source).resolve()
    checker_path=(args.checker or repository_checker()).resolve()
    with tempfile.TemporaryDirectory(prefix="glm53-0063-cpu-") as scratch:
        source=args.source.resolve()
        if not (source/"srt/managers/glm53_prefill_geometry_trace.py").exists():
            copied=Path(scratch)/"sglang"
            names=("srt/managers/schedule_batch.py", "srt/managers/tp_worker.py", "srt/managers/utils.py",
                "srt/model_executor/model_runner.py", "srt/model_executor/forward_batch_info.py",
                "srt/managers/scheduler.py", "srt/managers/scheduler_components/batch_result_processor.py",
                "srt/mem_cache/mamba_radix_cache.py", "srt/mem_cache/memory_pool.py", "srt/layers/sampler.py",
                "srt/sampling/glm53_forced_decode_reference.py", "srt/sampling/sampling_batch_info.py",
                "srt/distributed/parallel_state.py", "srt/managers/scheduler_components/request_receiver.py")
            for name in names:
                target=copied/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
            subprocess.run(["patch","--batch","--fuzz=0","-p1","-i",str(args.patch.resolve())],cwd=scratch,
                check=True,stdout=subprocess.DEVNULL)
            source=copied
        with cpu_only():
            receipt=qualify(source,baseline,checker_path)
        print(json.dumps(receipt,sort_keys=True))


if __name__ == "__main__": main()
