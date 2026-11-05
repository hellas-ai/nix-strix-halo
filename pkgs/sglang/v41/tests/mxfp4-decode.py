"""GPU qualification of the owner-routed MXFP4 decode chain against the installed sequence (gfx1151).

Every kernel is checked in isolation on the installed sequence's own intermediates, so a mismatch names the
kernel; the end-to-end chain (called directly and through `fused_experts_mxfp4`) is then compared bitwise with
the installed sequence (official routed policy, SGLANG_DSV41_MXFP4_DECODE=0) at every row count the guard admits.

Default policy (SGLANG_DSV41_MXFP4_GATE_UP_ROWS=live): the installed FMA live-row gate/up is kept at 2/4/8 rows,
so the chain is bitwise the installed sequence at every row count.  Opt-in policy 'wmma': the WMMA gate/up serves
every row count; it is bitwise the installed WMMA kernel (live rows forced off) and its drift from the installed
FMA live-row kernel at 2/4/8 rows is measured and reported (~0.3% of elements differ on random weights; never silently accepted).

usage: mxfp4-decode.py RUNTIME   (K2_OVERLAY=<dir with lib/python3.13/site-packages/sglang/...> loads patched files
                                   from a work tree instead of the installed runtime)
"""

import contextlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["SGLANG_USE_AITER"] = "0"
SWITCH = "SGLANG_DSV41_MXFP4_DECODE"
ROWS = "SGLANG_DSV41_MXFP4_GATE_UP_ROWS"
GEMM = "SGLANG_DSV41_MXFP4_DECODE_GEMM"
os.environ.pop(ROWS, None)
os.environ[SWITCH] = "0"  # installed sequence unless a case turns the new path on

import torch

runtime = Path(sys.argv[1]).resolve()
overlay = os.environ.get("K2_OVERLAY")
if overlay:
    # Load the patched files as the named modules before anything imports them. The parent packages are imported
    # normally first so the new modules become attributes of them (sglang.srt is a namespace package).
    site = Path(overlay).resolve() / "lib/python3.13/site-packages"
    importlib.import_module("sglang.kernels.ops.moe")
    importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils")
    for name, rel in (
        ("sglang.kernels.ops.moe.dsv41_mxfp4_decode", "sglang/kernels/ops/moe/dsv41_mxfp4_decode.py"),
        ("sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd",
         "sglang/srt/layers/moe/moe_runner/triton_utils/mxfp4_moe_amd.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, site / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent, _, leaf = name.rpartition(".")
        setattr(importlib.import_module(parent), leaf, module)

live = importlib.import_module("sglang.kernels.ops.gemm.dsv41_mxfp4_gate_up")
fast = importlib.import_module("sglang.kernels.ops.moe.dsv41_mxfp4_decode")
moe = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.mxfp4_moe_amd")
assert sys.modules[moe.__name__] is moe and hasattr(fast, "try_decode_routed")
assert "try_decode_routed" in Path(moe.__file__).read_text(), "hook missing: not the patched module"

E, K1, H, N1, N2, TOPK = 384, 5120, 576, 1152, 5120, 6
KW = dict(
    activation="silu", is_gated=True, inplace=False, no_combine=False,
    apply_router_weight_on_input=False, routed_scaling_factor=1.5, swiglu_limit=10,
    official_routed_policy=True,
)


def emit(name, passed, **details):
    print(json.dumps(dict(event="gate", name=name, passed=bool(passed), **details), default=str), flush=True)
    assert passed, name


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def ordered(t):
    """Monotone integer key of every finite float (+0 and -0 share key 0), for last-place distances."""
    width, sign = (torch.int16, 0x7FFF) if t.dtype == torch.bfloat16 else (torch.int32, 0x7FFFFFFF)
    raw = t.contiguous().view(width).long()
    magnitude = raw & sign
    return torch.where(raw < 0, -magnitude, magnitude)


def drift(a, b):
    """(values that differ numerically, max last-place distance) between two same-dtype tensors."""
    assert a.shape == b.shape and a.dtype == b.dtype
    differ = a != b
    differ &= ~(a.isnan() & b.isnan())
    distance = (ordered(a) - ordered(b)).abs()
    return int(differ.sum()), int(distance.max())


def make_layer(device, seed, scale_lo=112, scale_hi=124):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    w13 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, device=device, generator=g)
    w2 = torch.randint(0, 256, (E, N2, H // 2), dtype=torch.uint8, device=device, generator=g)
    s13 = torch.randint(scale_lo, scale_hi + 1, (E, N1, K1 // 32), dtype=torch.uint8, device=device, generator=g)
    s2 = torch.randint(scale_lo, scale_hi + 1, (E, N2, H // 32), dtype=torch.uint8, device=device, generator=g)
    return w13, w2, s13, s2


def activations(m, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(m, K1, device=device, generator=g)
    channel = torch.exp(0.6 * torch.randn(K1, device=device, generator=g))
    channel[torch.randint(0, K1, (8,), device=device, generator=g)] *= 25.0
    return (x * channel * 0.8).bfloat16()


PATTERNS = {
    "random": None,
    "shared": [[2, 7, 11, 19, 23, 31], [31, 23, 19, 11, 7, 2], [2, 5, 11, 40, 23, 99], [383, 0, 7, 11, 100, 101]],
    "disjoint": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [377, 378, 379, 380, 381, 382]],
    "duplicates": [[2, 2, 7, 7, 11, 11], [2, 7, 7, 11, 11, 13], [5, 5, 5, 5, 5, 5], [383, 383, 0, 0, 1, 1]],
    "reversed": [[31, 23, 19, 11, 7, 2], [2, 7, 11, 19, 23, 31], [5, 9, 13, 17, 21, 25], [25, 21, 17, 13, 9, 5]],
    "one_expert": [[5, 5, 5, 5, 5, 5]] * 4,  # 6*M routes on one expert: the down kernel needs several 16-row passes
}


def routes(pattern, m, device, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    if PATTERNS[pattern] is None:
        rows = [torch.randperm(E, generator=g)[:TOPK].tolist() for _ in range(m)]
    else:
        rows = [PATTERNS[pattern][t % 4] for t in range(m)]
        if m > 4 and pattern != "one_expert":  # keep the later rows distinct from the first four rows
            rows = [[(e + 3 * (t // 4)) % E for e in r] for t, r in enumerate(rows)]
    ids = torch.tensor(rows, dtype=torch.int32, device=device)
    w = torch.rand(m, TOPK, generator=g) + 0.05
    return ids, (w / w.sum(1, keepdim=True)).float().to(device)


@contextlib.contextmanager
def env(**values):
    saved = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def installed(x, w13, w2, s13, s2, weights, ids, force_wmma=False):
    """The installed sequence: (out, a1, gu, a2, ro); live rows can be forced off (installed WMMA control)."""
    records = []
    gemm, live_rows = moe._run_mxfp4_gemm, live.try_gate_up_live_rows

    def spy_gemm(*a, **k):
        gemm(*a, **k)
        records.append((a[0].clone(), a[2].clone()))

    moe._run_mxfp4_gemm = spy_gemm
    if force_wmma:
        live.try_gate_up_live_rows = lambda *a, **k: False
    try:
        with env(**{SWITCH: "0"}):
            out = moe.fused_experts_mxfp4(x, w13, w2, weights, ids, s13.view(torch.float8_e8m0fnu),
                                          s2.view(torch.float8_e8m0fnu), **KW)
    finally:
        moe._run_mxfp4_gemm, live.try_gate_up_live_rows = gemm, live_rows
    assert len(records) == 2
    return out, records[0][0], records[0][1], records[1][0], records[1][1]


def run_fast(x, w13, w2, s13, s2, weights, ids):
    return fast.decode_routed_experts(x, w13, w2, s13.view(torch.float8_e8m0fnu), s2.view(torch.float8_e8m0fnu),
                                      weights, ids, 1.5, 10, stages=True)


def through_hook(x, w13, w2, s13, s2, weights, ids):
    with env(**{SWITCH: "1"}):
        return moe.fused_experts_mxfp4(x, w13, w2, weights, ids, s13.view(torch.float8_e8m0fnu),
                                       s2.view(torch.float8_e8m0fnu), **KW)


def kernel_cases(layer, m, pattern, seed):
    """P, G, E, D, C each on the installed sequence's own inputs."""
    w13, w2, s13, s2 = layer
    device = w13.device
    x = activations(m, device, seed)
    ids, weights = routes(pattern, m, device, seed + 1)
    label = f"M{m}:{pattern}:{seed}"
    prod = installed(x, w13, w2, s13, s2, weights, ids)
    ctrl = installed(x, w13, w2, s13, s2, weights, ids, force_wmma=True)
    a1 = torch.empty_like(x)
    fast.launch_prep(x, a1)
    emit(label + ":P", bits_equal(a1, prod[1]) and bits_equal(a1, ctrl[1]))
    gu = torch.full_like(ctrl[2], float("nan"))
    fast.launch_gate_up_wmma(ctrl[1], w13, s13, ids, gu)
    emit(label + ":G_vs_installed_wmma", bits_equal(gu, ctrl[2]))
    for tag, ref in (("installed", prod), ("installed_wmma", ctrl)):
        a2 = torch.empty_like(ref[3])
        fast.launch_swiglu(ref[2], weights, a2, 1.5, fast.SILU_MODE)
        emit(label + f":E_on_{tag}", bits_equal(a2, ref[3]))
    if m in (2, 4, 8):  # the installed FMA live-row kernel vs the cheaper-dequant copy (and the installed kernel via the switch)
        for value, tag in (("1", "live_fast"), ("0", "live_installed_via_switch")):
            gu_live = torch.full_like(prod[2], float("nan"))
            with env(SGLANG_DSV41_MXFP4_LIVE_FAST=value):
                emit(label + f":{tag}_launched", fast._launch_live_rows_gate_up(prod[1], w13, s13, ids, gu_live))
            emit(label + f":{tag}_equals_installed", bits_equal(gu_live, prod[2]))
    ro = torch.full_like(prod[4], float("nan"))
    fast.launch_down_wmma(prod[3], w2, s2, ids, ro)
    emit(label + ":D", bits_equal(ro, prod[4]))
    out = torch.empty((m, N2), dtype=torch.float32, device=device)
    fast.launch_combine(prod[4], ids, out)
    emit(label + ":C", bits_equal(out, prod[0]))
    return prod, ctrl, (x, ids, weights, label)


def chain_cases(layer, m, pattern, seed, report):
    w13, w2, s13, s2 = layer
    prod, ctrl, (x, ids, weights, label) = kernel_cases(layer, m, pattern, seed)
    # default policy (live rows kept): every stage and the output are the installed sequence's bits, directly and
    # through the dispatch hook of fused_experts_mxfp4
    with env(**{ROWS: None}):
        a1, gu, a2, ro, out = run_fast(x, w13, w2, s13, s2, weights, ids)
        emit(label + ":default_stages_equal_installed",
             all(bits_equal(a, b) for a, b in zip((a1, gu, a2, ro), prod[1:])))
        emit(label + ":default_chain_equals_installed", bits_equal(out, prod[0]))
        emit(label + ":hook_equals_installed", bits_equal(through_hook(x, w13, w2, s13, s2, weights, ids), prod[0]))
    # AITER backend: only P/E/C are new; the installed GEMMs run, so every stage is bitwise the installed one at every row count
    with env(**{GEMM: "aiter", ROWS: None}):
        a1, gu, a2, ro, out = run_fast(x, w13, w2, s13, s2, weights, ids)
        emit(label + ":aiter_backend_stages_equal_installed",
             all(bits_equal(a, b) for a, b in zip((a1, gu, a2, ro, out), (prod[1], prod[2], prod[3], prod[4], prod[0]))))
    # opt-in WMMA policy: bitwise the installed WMMA kernel at every row count
    with env(**{ROWS: "wmma"}):
        a1, gu, a2, ro, out = run_fast(x, w13, w2, s13, s2, weights, ids)
        emit(label + ":wmma_chain_equals_installed_wmma", bits_equal(out, ctrl[0]))
        n_gu, ulp_gu = drift(gu, prod[2])
        n_out, ulp_out = drift(out, prod[0])
        row = dict(case=label, rows=m, gate_up_differing=n_gu, gate_up_total=gu.numel(), gate_up_max_ulp=ulp_gu,
                   out_differing=n_out, out_total=out.numel(), out_max_ulp=ulp_out)
        report.append(row)
        print(json.dumps(dict(event="wmma_policy_drift", **row)), flush=True)
        if m in (2, 4, 8):  # installed gate/up is the FMA kernel: the drift is reported above and loosely bounded
            # (strix-2 measurement: ~0.3% of elements differ; the last-place distance is large only where the result
            # cancels to near zero, so only the fraction is asserted and the maximum is reported)
            emit(label + ":wmma_gate_up_drift_bounded", n_gu <= 1e-2 * gu.numel(),
                 differing=n_gu, max_ulp=ulp_gu)
        else:
            emit(label + ":wmma_gate_up_equals_installed", n_gu == 0)
            emit(label + ":wmma_chain_equals_installed", bits_equal(out, prod[0]))


def graph_cases(layer):
    w13, w2, s13, s2 = layer
    device = w13.device
    s13e, s2e = s13.view(torch.float8_e8m0fnu), s2.view(torch.float8_e8m0fnu)
    for m in (1, 2, 3, 4):
        x = activations(m, device, 71)
        ids, weights = routes("shared", m, device, 72)
        with env(**{ROWS: None}):
            fast.decode_routed_experts(x, w13, w2, s13e, s2e, weights, ids, 1.5, 10)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fast.decode_routed_experts(x, w13, w2, s13e, s2e, weights, ids, 1.5, 10)
            results = []
            for label, seed in (("A", 73), ("B", 74), ("A-again", 73)):
                x.copy_(activations(m, device, seed))
                new_ids, new_weights = routes("random" if seed == 74 else "shared", m, device, seed + 100)
                ids.copy_(new_ids)
                weights.copy_(new_weights)
                graph.replay()
                torch.cuda.synchronize()
                replay = captured.clone()
                eager = fast.decode_routed_experts(x, w13, w2, s13e, s2e, weights, ids, 1.5, 10)
                reference = installed(x, w13, w2, s13, s2, weights, ids)[0]
                emit(f"graph:M{m}:{label}", bits_equal(replay, eager) and bits_equal(replay, reference))
                results.append(replay)
        emit(f"graph:M{m}:changed_inputs_change_output", not bits_equal(results[0], results[1]))
        emit(f"graph:M{m}:return_to_A", bits_equal(results[0], results[2]))


def fallback_and_switch_cases(layer):
    w13, w2, s13, s2 = layer
    device = w13.device
    s13e, s2e = s13.view(torch.float8_e8m0fnu), s2.view(torch.float8_e8m0fnu)
    for m in (17, 24):  # outside the decode envelope: the installed sequence runs, bit for bit
        x = activations(m, device, 5)
        ids, weights = routes("random", m, device, 6)
        with env(**{SWITCH: "1"}):
            assert fast.try_decode_routed(x, w13, w2, s13e, s2e, weights, ids, 1.5, 10) is None
        emit(f"fallback:M{m}", bits_equal(through_hook(x, w13, w2, s13, s2, weights, ids),
                                          installed(x, w13, w2, s13, s2, weights, ids)[0]))
    x = activations(2, device, 7)
    ids, weights = routes("random", 2, device, 8)
    for value, expect in (("1", True), ("0", False), ("false", False), ("off", False), ("no", False)):
        with env(**{SWITCH: value}):
            got = fast.try_decode_routed(x, w13, w2, s13e, s2e, weights, ids, 1.5, 10) is not None
        emit(f"switch:{value}", got == expect)
    with env(**{SWITCH: "1"}):
        # a different dtype/geometry never enters the chain
        assert fast.try_decode_routed(x.float(), w13, w2, s13e, s2e, weights, ids, 1.5, 10) is None
        assert fast.try_decode_routed(x, w13[:383], w2, s13e, s2e, weights, ids, 1.5, 10) is None
    emit("guard_rejects_other_geometry", True)


def corner_scale_cases():
    """E8M0 scale bytes 0..255 (reserved 0/255 included): G and D keep the installed WMMA kernel's classification."""
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(91)
    w13 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, device=device, generator=g)
    w2 = torch.randint(0, 256, (E, N2, H // 2), dtype=torch.uint8, device=device, generator=g)
    for lo, hi in ((0, 255), (0, 6), (248, 255), (100, 140)):
        s13 = torch.randint(lo, hi + 1, (E, N1, K1 // 32), dtype=torch.uint8, device=device, generator=g)
        s2 = torch.randint(lo, hi + 1, (E, N2, H // 32), dtype=torch.uint8, device=device, generator=g)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 7, 8, 9, 10]], dtype=torch.int32, device=device)
        weights = torch.full((2, TOPK), 1 / 6, dtype=torch.float32, device=device)
        x = activations(2, device, 92)
        ctrl = installed(x, w13, w2, s13, s2, weights, ids, force_wmma=True)
        gu = torch.full_like(ctrl[2], float("nan"))
        fast.launch_gate_up_wmma(ctrl[1], w13, s13, ids, gu)
        ro = torch.full_like(ctrl[4], float("nan"))
        fast.launch_down_wmma(ctrl[3], w2, s2, ids, ro)

        def same(a, b):  # equal bits, with every NaN treated as the same value
            return torch.equal(a.isnan(), b.isnan()) and bits_equal(torch.nan_to_num(a, nan=0.0), torch.nan_to_num(b, nan=0.0))

        emit(f"corner_scales[{lo},{hi}]:G", same(gu, ctrl[2]))
        emit(f"corner_scales[{lo},{hi}]:D", same(ro, ctrl[4]))
        for rows_ in (2, 4):  # installed FMA live rows vs the cheaper-dequant copy (scale bytes outside [1,252] take the installed arithmetic)
            ids4 = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 7, 8, 9, 10], [3, 3, 7, 7, 0, 0], [11, 10, 9, 8, 7, 6]][:rows_],
                                dtype=torch.int32, device=device)
            x4 = activations(rows_, device, 93)
            prod4 = installed(x4, w13, w2, s13, s2, torch.full((rows_, TOPK), 1 / 6, dtype=torch.float32, device=device), ids4)
            gu_live = torch.full_like(prod4[2], float("nan"))
            assert fast._launch_live_rows_gate_up(prod4[1], w13, s13, ids4, gu_live)
            emit(f"corner_scales[{lo},{hi}]:live_rows{rows_}", same(gu_live, prod4[2]))
    del w13, w2


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.06, 0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    assert Path(moe.__file__).resolve().is_relative_to(runtime) or overlay
    layer = make_layer("cuda", 1234)
    report = []
    for m in range(1, 17):  # every row count the guard admits
        for pattern in PATTERNS if m <= 8 else ("random", "shared", "duplicates", "one_expert"):
            for seed in (11, 12) if m <= 8 else (11,):
                chain_cases(layer, m, pattern, seed, report)
    graph_cases(layer)
    fallback_and_switch_cases(layer)
    del layer
    torch.cuda.empty_cache()
    corner_scale_cases()
    print(json.dumps(dict(event="complete", passed=True, cases=len(report),
                          peak_gpu_allocated=torch.cuda.max_memory_allocated())), flush=True)


if __name__ == "__main__":
    main()
