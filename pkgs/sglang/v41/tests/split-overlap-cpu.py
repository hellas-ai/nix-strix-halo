#!/usr/bin/env python3
"""CPU-only checks of the two-half prefill pipeline (0102): default off, split rows 128-aligned in absolute position with
both halves >= 256 rows, split_reduce is a plain call outside the pipeline, the two half threads strictly alternate at
their reductions (and the survivor runs on alone once the other finishes), and both reduction sites (attention output,
routed-expert output) go through split_reduce.

  split-overlap-cpu.py RUNTIME_ROOT
"""
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import torch

root = Path(sys.argv[1]) / "lib/python3.13/site-packages/sglang"
os.environ.pop("SGLANG_DSV41_PREFILL_SPLIT_OVERLAP", None)
spec = importlib.util.spec_from_file_location("split_under_test", root / "srt/models/deepseek_common/amd/dsv41_split_overlap.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # dataclasses resolve string annotations through sys.modules
spec.loader.exec_module(mod)
assert mod._ENABLED is False, "must be default off"

cases = 0
for prefix in (0, 256, 1536, 4864, 15360, 16384 - 128):
    for rows in (1024, 1280, 1536, 1792, 2047):
        s = mod._split_row(prefix, rows)
        assert s is not None, (prefix, rows)
        assert (prefix + s) % 128 == 0 and 256 <= s <= rows - 256, (prefix, rows, s)
        assert abs(s - rows // 2) <= 64, (prefix, rows, s)
        cases += 1
assert mod._split_row(0, 400) is None

x = torch.ones(4)
assert mod.split_reduce(x, lambda t: t * 4).sum() == 16, "outside the pipeline split_reduce is the reduction itself"

# turn-taking: half 0 does 3 stages, half 1 does 2; the trace must alternate, then half 0 finishes alone
backend = NS(forward_metadata=None, candidate_masks=None, token_to_kv_pool=NS(request_window=None))
halves = [NS(meta=("meta", 0), masks=None), NS(meta=("meta", 1), masks=None)]
coop = mod._Coop(backend, halves)
trace = []
def body(h, stages):
    coop.wait_turn(h)
    for k in range(stages):
        assert backend.forward_metadata == ("meta", h), "resumed half must see its own metadata"
        trace.append((h, k))
        coop.switch(h)
    coop.finish(h)
threads = [threading.Thread(target=body, args=(0, 3)), threading.Thread(target=body, args=(1, 2))]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=10)
    assert not t.is_alive(), "turn-taking deadlocked"
assert trace == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)], trace

# join mode: two halves meet at the routed experts, compute runs once on the merged rows, each half gets its own rows
calls = []
def compute(merged, fb, ids, ids_global):
    calls.append(merged.shape[0])
    return merged * 2
join = mod._Join([None, None], [None, None])
coop2 = mod._Coop(NS(forward_metadata=None, candidate_masks=None, token_to_kv_pool=NS(request_window=None)), [NS(meta=0, masks=None), NS(meta=1, masks=None)])
outs = [None, None]
def body2(h):
    mod._tls.state = mod._ThreadState(h, None, None, coop2, [], join, 3, None, None, None)
    coop2.wait_turn(h)
    outs[h] = mod.moe_join(torch.arange(3.0) if h == 0 else torch.arange(3.0, 5.0), compute)
    mod._tls.state = None
    coop2.finish(h)
threads = [threading.Thread(target=body2, args=(h,)) for h in (0, 1)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=10)
    assert not t.is_alive(), "join deadlocked"
assert calls == [5], calls
assert outs[0].tolist() == [0.0, 2.0, 4.0] and outs[1].tolist() == [6.0, 8.0], outs
assert mod.moe_capture(torch.ones(1), None, None) is False, "capture only inside the merged call"

v4 = (root / "srt/models/deepseek_v4.py").read_text()
moe = (root / "srt/models/deepseek_common/amd/deepseek_v2_hip_moe.py").read_text()
assert "o = split_reduce(o, reduce)" in v4, "attention reduction must go through split_reduce"
assert "dsv41_split_overlap.plan(" in v4 and "dsv41_split_overlap.run(" in v4, "layer-loop driver missing"
assert "return split_reduce(hidden_states, post_experts_all_reduce)" in moe, "routed-expert reduction must go through split_reduce"
v2 = (root / "srt/models/deepseek_v2.py").read_text()
assert "dsv41_split_overlap.moe_join_active()" in v4 and "dsv41_split_overlap.moe_join(" in v4, "routed-expert join hook missing"
assert "dsv41_split_overlap.moe_capture(" in v2 and "final_hidden_states, self._all_reduce_output, after_reduce" in v2, "routed-expert capture hook missing"
print(f"PASS {cases} CPU split-pipeline row cases; split_reduce passthrough; strict half alternation; merged-rows expert join; reduction, join and capture hooks; default off")
