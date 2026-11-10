#!/usr/bin/env python3
"""Independent lattice, scale, selection and graph checks for the gfx1151 indexer.

Run with the candidate sglang-python and candidate runtime as the first argument.
The advisory serving lock is held before importing Torch or initializing a GPU.
"""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

if sys.flags.optimize:
    raise SystemExit("Assertions must remain enabled")
runtime = Path(sys.argv[1]).resolve()
lock = open("/tmp/ds41-gpu.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.environ["SGLANG_USE_AITER"] = "0"
os.environ["SGLANG_DSV4_FP4_LOGITS_BUDGET_MB"] = "64"
import torch
import sgl_kernel  # noqa: F401 - registers the native top-k Torch operators
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4 import fp4_indexer_gfx1151 as native
from sglang.kernels.ops.attention.dsv4 import fp4_indexer_hip as fp4

assert Path(native.__file__).resolve().is_relative_to(runtime)
assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.03)
fixture = Path(__file__).with_name("indexer.py")
spec = importlib.util.spec_from_file_location("literal_fp4_oracle", fixture)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def scalar_logits(q, qs, k, ks, weights, table, lens, scale, out):
    """Retained scalar arithmetic, independent of the WMMA path's guard."""
    native._paged_logits[(q.shape[0], triton.cdiv(out.shape[1], 16))](
        q,
        qs,
        k,
        ks,
        weights,
        table,
        lens,
        out,
        q.stride(0),
        qs.stride(0),
        weights.stride(0),
        table.stride(0),
        WIDTH=out.shape[1],
        PAGE_COLS=table.shape[1],
        PAGES=k.shape[0],
        HEADS=q.shape[1],
        WEIGHT_SCALE=scale,
        BLOCK=16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


# The unchanged scalar ABI remains available for 16,48,64 heads.
for heads in (48, 64):
    oracle.run_case([0, 33, 513], heads, 107 + heads, True)

# Packed exponent/code tests compare finite values and NaN/Inf classes to the
# deployed scalar, including scales outside ordinary model activations.
torch.manual_seed(112)
q = torch.randint(0, 256, (1, 32, 64), dtype=torch.uint8, device="cuda")
qs = torch.empty((1, 1, 4, 16, 4), dtype=torch.uint8, device="cuda")
k = torch.randint(0, 256, (1, 1, 4, 64, 16), dtype=torch.uint8, device="cuda")
ks = torch.empty((1, 1, 4, 64), dtype=torch.uint8, device="cuda")
w = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
pt = torch.tensor([[0]], device="cuda", dtype=torch.int32)
lens = torch.tensor([63], device="cuda", dtype=torch.int32)
a = torch.empty((1, 65), device="cuda")
b = torch.empty_like(a)
patterns = [
    (0, 127),
    (1, 127),
    (2, 128),
    (114, 114),
    (126, 126),
    (127, 127),
    (140, 120),
    (200, 127),
    (250, 114),
    (252, 114),
    (253, 114),
    (254, 114),
    (254, 0),
    (255, 127),
    (114, 254),
    (63, 63),
    (80, 90),
    (100, 100),
    (180, 180),
    (200, 200),
]
patterns += [
    ([114, 126, 140, 127], [127, 127, 127, 127]),
    ([126, 127, 128, 129], [130, 128, 125, 124]),
    ([250, 240, 252, 220], [114, 124, 114, 145]),
]
for i, (qe, ke) in enumerate(patterns):
    for g in range(4):
        qs[:, :, g].fill_(qe[g] if isinstance(qe, list) else qe)
        ks[:, :, g].fill_(ke[g] if isinstance(ke, list) else ke)
    scalar_logits(q, qs, k, ks, w, pt, lens, 1.0, a)
    native.paged_fp4_logits(q, qs, k, ks, w, pt, lens, 1.0, b)
    torch.cuda.synchronize()
    assert torch.equal(a.isnan(), b.isnan()), ("NaN class", i)
    assert torch.equal(a.isposinf(), b.isposinf()), ("positive Inf class", i)
    assert torch.equal(a.isneginf(), b.isneginf()), ("negative Inf class", i)
    finite = a.isfinite()
    torch.testing.assert_close(a[finite], b[finite], rtol=2e-5, atol=2e-4)
    qex = qe if isinstance(qe, list) else [qe] * 4
    kex = ke if isinstance(ke, list) else [ke] * 4
    safe = (
        min(qex) >= 2
        and max(qex) <= 252
        and min(kex) >= 2
        and max(kex) <= 252
        and min(qex) + min(kex) >= 130
        and max(qex) + max(kex) <= 368
        and max(qex) - min(qex) + max(kex) - min(kex) <= 8
    )
    emit(
        "scale-case",
        index=i,
        qscale=qe,
        kscale=ke,
        wmma_branch=safe,
        finite=int(finite.sum()),
        exact=bool(torch.equal(a[finite], b[finite])),
        max_error=float((a[finite] - b[finite]).abs().max()) if finite.any() else 0.0,
    )


@triton.jit
def probe(Q, QS, K, KS, O, RAW):
    p = tl.arange(0, 32)
    h = tl.arange(0, 32)
    d = native._group_dot(
        Q,
        QS,
        K,
        KS,
        0,
        tl.full((32,), 0, tl.int32),
        p,
        tl.full((32,), True, tl.int1),
        2048,
        256,
        0,
        32,
        32,
    )
    tl.store(O + h[:, None] * 32 + p[None, :], d)
    dim = tl.arange(0, 32)
    qb = tl.load(Q + h[:, None] * 64 + dim[None, :] // 2).to(tl.int32)
    kb = tl.load(K + p[None, :] * 16 + dim[:, None] // 2).to(tl.int32)
    q = native._e2m1_bf16((qb >> ((dim[None, :] % 2) * 4)) & 15)
    k = native._e2m1_bf16((kb >> ((dim[:, None] % 2) * 4)) & 15)
    raw = tl.dot(q, k)
    tl.store(RAW + h[:, None] * 32 + p[None, :], raw)


def packed(codes):
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


qs = torch.full((1, 1, 4, 16, 4), 127, dtype=torch.uint8, device="cuda")
ks = torch.full((1, 1, 4, 64), 127, dtype=torch.uint8, device="cuda")
o = torch.empty((32, 32), device="cuda")
raw = torch.empty_like(o)
max_raw = 0.0
cases = 0
patterns = []
# All16x16 code pairs, including negative zero; maximal positive/negative sums.
qc = torch.arange(32, dtype=torch.uint8).remainder(16)[:, None].expand(32, 128).clone()
kc = qc.repeat(2, 1)
patterns.append(("all-code-pairs", qc, kc))
for label, qpattern, kpattern in (
    ("max-pos", [7], [7]),
    ("max-neg", [7], [15]),
    ("large-cancel", [7], [7, 15]),
    ("mixed-cancel", [7, 15], [7]),
    ("signed-zero", [0, 8], [15, 7]),
    ("zero", [0], [7]),
    ("interleaved-codes", list(range(16)), list(reversed(range(16)))),
):
    q = (
        torch.tensor(qpattern, dtype=torch.uint8)
        .repeat(128 // len(qpattern))[None, :]
        .repeat(32, 1)
    )
    k = (
        torch.tensor(kpattern, dtype=torch.uint8)
        .repeat(128 // len(kpattern))[None, :]
        .repeat(64, 1)
    )
    patterns.append((label, q, k))
for seed in range(128):
    gen = torch.Generator().manual_seed(seed + 2301)
    patterns.append(
        (
            "seed-" + str(seed),
            torch.randint(0, 16, (32, 128), dtype=torch.uint8, generator=gen),
            torch.randint(0, 16, (64, 128), dtype=torch.uint8, generator=gen),
        )
    )
for name, qc, kc in patterns:
    q = packed(qc)[None].cuda()
    k = packed(kc).reshape(64, 4, 16).permute(1, 0, 2).contiguous()[None, None].cuda()
    probe[(1,)](q, qs, k, ks, o, raw, enable_fp_fusion=False, num_warps=4)
    qv = oracle.LEVELS[(qc & 7).long()] * torch.where((qc & 8) != 0, -1.0, 1.0)
    kv = oracle.LEVELS[(kc & 7).long()] * torch.where((kc & 8) != 0, -1.0, 1.0)
    expected = qv[:, :32] @ kv[:32, :32].T
    got = o.cpu().double()
    err = float((raw.cpu().double() - expected).abs().max())
    assert torch.equal(got, expected), (name, float((got - expected).abs().max()))
    assert err < 0.125, (name, err)
    max_raw = max(max_raw, err)
    cases += 1
    emit("lattice-case", name=name, restored_exact=True, unrounded_max_error=err)
# Equal-score rows with different nibble order must preserve logical tie order.

for label in ("positive-ties", "signed-zero-ties"):
    rows = 1
    length = 1024
    pages = 16
    qcodes = torch.full(
        (rows, 32, 128), 1 if label == "positive-ties" else 8, dtype=torch.uint8
    )
    kcodes = torch.stack(
        [
            torch.arange(128, dtype=torch.uint8).remainder(8).roll(i % 128)
            for i in range(length)
        ]
    )
    q = packed(qcodes).cuda()
    k = (
        packed(kcodes)
        .reshape(pages, 64, 4, 16)
        .permute(0, 2, 1, 3)
        .contiguous()[:, None]
        .cuda()
    )
    sf = torch.full((pages, 1, 4, 64), 127, dtype=torch.uint8, device="cuda")
    weights = torch.arange(1, 33, device="cuda", dtype=torch.bfloat16)[None]
    table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32)[None]
    lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    scores = torch.empty((1, 1089), device="cuda")
    native.paged_fp4_logits(q, qs, k, sf, weights, table, lens, 1.0, scores)
    expected = torch.full((1, 1089), -torch.inf, dtype=torch.float64)
    # All positive-code rows have identical sums, and every head is identical.
    value = (
        0.0
        if label == "signed-zero-ties"
        else float((oracle.LEVELS[:8].sum() * 16 * 0.5) * sum(range(1, 33)))
    )
    expected[:, :length] = value
    oracle.check_scores(scores, expected)
    oracle.check_selection(scores, expected, lens, table, 128)
    emit("tie-case", name=label, score=value, selection="exact")
emit(
    "lattice-complete",
    cases=cases,
    unrounded_max_error=max_raw,
    restored="FP64-exact",
    selection="exact",
)

for lengths in ([0, 15, 16, 17], [31, 32, 33], [0, 513, 1537]):
    oracle.run_case(lengths, 32, 401 + len(lengths), graph=True)
# Empty batches return the caller-owned output without a launch.
q = torch.empty((0, 32, 64), dtype=torch.uint8, device="cuda")
qs = torch.empty((0, 1, 4, 16, 4), dtype=torch.uint8, device="cuda")
out = torch.empty((0, 65), device="cuda")
assert (
    native.paged_fp4_logits(
        q,
        qs,
        k,
        sf,
        torch.empty((0, 32), device="cuda"),
        torch.empty((0, 1), device="cuda", dtype=torch.int32),
        torch.empty((0,), device="cuda", dtype=torch.int32),
        1.0,
        out,
    )
    is out
)

# Bounded decode/prefill chunk controls, including more rows than the CTA budget.
for batch, width in [(15, 1052672), (64, 4096), (321, 2048)]:
    torch.manual_seed(901 + batch)
    q0 = torch.randn(batch, 32, 128, device="cuda", dtype=torch.bfloat16)
    qp, qs = fp4.pack_fp4_query_flydsl(q0)
    k0 = torch.randn(2048, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(32, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    ks = torch.empty(32, 1, 4, 64, dtype=torch.uint8, device="cuda")
    fp4.store_fp4_index_k_cache_split(
        k0, k, ks, torch.arange(2048, device="cuda"), page_size=64, rne=True
    )
    cols = (width + 63) // 64
    pt = (torch.arange(cols, device="cuda") % 32).to(torch.int32)[None].repeat(batch, 1)
    lens = torch.tensor(
        ([0, 1, 15, 16, 17, 31, 32, 33, 512, 513, 1024] * ((batch + 10) // 11))[:batch],
        device="cuda",
        dtype=torch.int32,
    )
    weights = torch.randn(batch, 64, device="cuda", dtype=torch.bfloat16)[
        :, ::2
    ].contiguous()
    out = torch.empty(batch, width, device="cuda")
    ref = torch.empty_like(out)
    snapshots = {
        name: t.clone()
        for name, t in [
            ("q", qp),
            ("qs", qs),
            ("k", k),
            ("ks", ks),
            ("w", weights),
            ("pt", pt),
            ("lens", lens),
        ]
    }

    def run(output=out):
        return native.paged_fp4_logits(qp, qs, k, ks, weights, pt, lens, 0.37, output)

    run()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for phase in ("A", "B", "A-return"):
        for name, t in [
            ("q", qp),
            ("qs", qs),
            ("k", k),
            ("ks", ks),
            ("w", weights),
            ("pt", pt),
            ("lens", lens),
        ]:
            t.copy_(snapshots[name])
        if phase == "B":
            qp.bitwise_xor_(136)
            k.bitwise_xor_(136)
            weights.neg_()
            pt.copy_(pt.flip(1))
            pt[:, 0] = -1
            pt[:, 1] = 32
            lens.copy_(lens.flip(0))
            lens[0] = min(width, 2048)
        scalar_logits(qp, qs, k, ks, weights, pt, lens, 0.37, ref)
        eager = run().clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), (batch, phase, "capture mismatch")
        assert torch.equal(out.isneginf(), ref.isneginf())
        finite = ref.isfinite()
        torch.testing.assert_close(out[finite], ref[finite], rtol=2e-5, atol=2e-4)
        emit(
            "shape-phase",
            batch=batch,
            width=width,
            phase=phase,
            eager_graph_exact=True,
            baseline_exact=torch.equal(out, ref),
            max_error=float((out[finite] - ref[finite]).abs().max())
            if finite.any()
            else 0.0,
        )
    del out, ref, eager, graph, snapshots
    torch.cuda.empty_cache()

emit(
    "complete",
    oracle_max_error=oracle.MAX_ERROR,
    logits_rtol=2e-5,
    logits_atol=2e-4,
    selection="exact",
)
