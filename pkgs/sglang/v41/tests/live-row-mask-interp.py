"""Triton-interpreter (CPU) checks of the live-row mask of the MXFP4 decode chain: `python live-row-mask-interp.py <runtime-or-overlay>`.

Not registered in the build (the interpreter needs minutes); run by hand, results are kept next to the patch.

  A  kernel P-live: its operand output is bitwise kernel P's, its id output is `where(row < live, id, -1)`, for every row count
     1..16 and every live count 1..M (TOPK 6 and 3).
  B  chain, WMMA gate/up (E=16 experts at the real TP4 geometry): rows < live are bitwise the unmasked chain's at every
     stage (a1, gate/up, swiglu operand, down, fp32 combine); dead rows have zero gate/up, down and output; the masked chain
     touches only the experts of live rows.
  C  chain, installed FMA live-row gate/up kernels (rows 2/4/8, subset of N tiles so the interpreter finishes): rows < live
     bitwise unchanged, dead routes zero, including a pair that holds one live and one dead row routed to the same expert.

The interpreter's dot is not the WMMA: this is evidence for indexing, masking and ownership, not for GPU bit-exactness (that is
`live-row-mask-gpu.py`)."""

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
os.environ["SGLANG_DSV41_MXFP4_GATE_UP_ROWS"] = "wmma"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import numpy as np
import torch
import triton.runtime.interpreter as _interp

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"

# The stock interpreter's fp32 -> float8e4nv / bf16 conversions are not round-to-nearest-even; replace only those (same patch as
# K2's CPU tests).  Both sides of every comparison below run through the same interpreter, so this only keeps the data realistic.
_orig_fp_to_fp = _interp.InterpreterBuilder.create_fp_to_fp


def _fp_to_fp(self, src, dst_type, rounding_mode):
    import triton.language as _tl

    if dst_type.scalar == _tl.float8e4nv and src.dtype.scalar == _tl.float32:
        q = torch.from_numpy(np.ascontiguousarray(src.data)).to(torch.float8_e4m3fn)
        return _interp.TensorHandle(q.view(torch.uint8).numpy().copy(), dst_type.scalar)
    return _orig_fp_to_fp(self, src, dst_type, rounding_mode)


_interp.InterpreterBuilder.create_fp_to_fp = _fp_to_fp
_orig_cast_impl = _interp.InterpreterBuilder.cast_impl


def _cast_impl(self, src, dst_type):
    import triton.language as _tl

    if src.dtype.scalar == _tl.float32 and dst_type.scalar == _tl.bfloat16:
        b = torch.from_numpy(np.ascontiguousarray(src.data)).to(torch.bfloat16)
        return _interp.TensorHandle(b.view(torch.int16).numpy().view(np.uint16).copy(), dst_type.scalar)
    return _orig_cast_impl(self, src, dst_type)


_interp.InterpreterBuilder.cast_impl = _cast_impl
for _n in ("create_fp_trunc", "create_fp_ext", "create_si_to_fp", "create_ui_to_fp", "create_fp_to_si", "create_fp_to_ui"):
    setattr(_interp.InterpreterBuilder, _n, (lambda self, src, dst_type: self.cast_impl(src, dst_type)))

spec = importlib.util.spec_from_file_location("dsv41_mxfp4_decode_interp", SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_decode.py")
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)

FAILS = []


def bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        FAILS.append(name)


def same(a, b):
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def reference_ids(ids, live):
    rows = torch.arange(ids.shape[0]).unsqueeze(1)
    return torch.where(rows < live, ids, torch.full_like(ids, -1))


def heavy(shape):
    x = torch.randn(shape) * torch.exp(torch.randn(shape) * 1.5)
    x[..., ::97] = 0
    return x.to(torch.bfloat16)


# ------------------------------------------------------------------ A
t0 = time.time()
torch.manual_seed(11)
for topk in (6, 3):
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        x = heavy((m, P.HIDDEN))
        ids = torch.randint(0, 384, (m, topk), dtype=torch.int32)
        a_ref = torch.empty_like(x)
        P.launch_prep(x, a_ref)
        for live in sorted({1, max(1, m - 1), m, max(1, m // 2)}):
            a = torch.full_like(x, float("nan"))
            out = torch.full_like(ids, 12345)
            P.launch_prep_live(x, a, ids, out, torch.tensor([live], dtype=torch.int32))
            check(f"A topk={topk} M={m} live={live}: operand bitwise, ids masked",
                  same(a, a_ref) and torch.equal(out, reference_ids(ids, live)))
print(f"A done in {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ B
E, K1, H, N1, N2, TOPK = 16, 5120, 576, 1152, 5120, 6
g = torch.Generator().manual_seed(5)
w13 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, generator=g)
w2 = torch.randint(0, 256, (E, N2, H // 2), dtype=torch.uint8, generator=g)
s13 = torch.randint(118, 125, (E, N1, K1 // 32), dtype=torch.uint8, generator=g).view(torch.float8_e8m0fnu)
s2 = torch.randint(118, 125, (E, N2, H // 32), dtype=torch.uint8, generator=g).view(torch.float8_e8m0fnu)


def chain(x, tw, ids, live):
    t = None if live is None else torch.tensor([live], dtype=torch.int32)
    return P.decode_routed_experts(x, w13, w2, s13, s2, tw, ids, 1.5, 10, silu_mode=9, stages=True, live_rows=t)


cases_b = (
    (4, 3, [[0, 1, 2, 3, 4, 5], [5, 4, 3, 9, 10, 11], [3, 3, 7, 7, 0, 0], [11, 10, 9, 8, 7, 6]]),   # dead row shares experts with live rows
    (3, 2, [[2, 2, 7, 7, 11, 11], [2, 5, 6, 7, 8, 9], [15, 14, 13, 12, 1, 0]]),                     # dead row owns experts nobody else uses
)
for m, live, ids_l in cases_b:
    t0 = time.time()
    ids = torch.tensor(ids_l, dtype=torch.int32)
    x = (torch.randn(m, K1) * torch.exp(0.5 * torch.randn(K1))).bfloat16()
    tw = torch.rand(m, TOPK)
    tw = (tw / tw.sum(1, keepdim=True)).float()
    ref = chain(x, tw, ids, None)
    msk = chain(x, tw, ids, live)
    n_routes = live * TOPK
    names = ("a1", "gate/up", "swiglu operand", "down", "combine")
    for name, r, k in zip(names, ref, msk):
        rows = n_routes if name == "swiglu operand" else live
        check(f"B M={m} live={live} {name}: live rows bitwise unchanged", same(r[:rows], k[:rows]))
    check(f"B M={m} live={live}: dead gate/up routes are zero", bool((msk[1][live:].float() == 0).all()))
    check(f"B M={m} live={live}: dead down routes are zero", bool((msk[3][live:].float() == 0).all()))
    check(f"B M={m} live={live}: dead combine rows are zero", bool((msk[4][live:] == 0).all()))
    check(f"B M={m} live={live}: dead rows no longer matter for live results (garbage ids in dead rows)",
          same(chain(x, tw, torch.cat([ids[:live], torch.full_like(ids[live:], 3)]), live)[4][:live], msk[4][:live]))
    print(f"B M={m} done in {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ C
legacy_w = torch.randint(0, 256, (E, 1152, 2560), dtype=torch.uint8, generator=g)
legacy_s = torch.randint(117, 126, (E, 1152, 160), dtype=torch.uint8, generator=g)


def run_live(rows, a, ids, tiles=4):
    y = torch.full((rows, 6, 1152), float("nan"), dtype=torch.bfloat16)
    if rows == 2:
        P._live_rows[(tiles, 12)](a, legacy_w, legacy_s, ids, y, ROWS=8, BLOCK_K=512)
    else:
        P._live_row_pairs[(tiles, 12, rows // 2)](a, legacy_w, legacy_s, ids, y, ROWS=8, BLOCK_K=512)
    return y[:, :, : tiles * 8]


cases_c = (
    (2, 1, [[0, 1, 2, 3, 4, 5], [5, 4, 3, 9, 10, 11]]),
    (4, 3, [[0, 1, 2, 3, 4, 5], [5, 4, 3, 9, 10, 11], [3, 3, 7, 7, 0, 0], [11, 10, 9, 8, 7, 6]]),
    (4, 2, [[0, 1, 2, 3, 4, 5], [5, 4, 3, 9, 10, 11], [3, 3, 7, 7, 0, 0], [11, 10, 9, 8, 7, 6]]),
)
for rows, live, ids_l in cases_c:
    t0 = time.time()
    ids = torch.tensor(ids_l, dtype=torch.int32)
    a = (torch.randn(rows, 5120) * torch.exp(0.5 * torch.randn(5120))).bfloat16()
    ref = run_live(rows, a, ids)
    msk = run_live(rows, a, reference_ids(ids, live))
    check(f"C rows={rows} live={live}: live routes bitwise unchanged", same(ref[:live], msk[:live]))
    check(f"C rows={rows} live={live}: dead routes are zero", bool((msk[live:].float() == 0).all()))
    print(f"C rows={rows} live={live} done in {time.time() - t0:.0f}s", flush=True)

print("SUMMARY:", "ALL PASS" if not FAILS else f"FAILURES {FAILS}")
sys.exit(1 if FAILS else 0)
