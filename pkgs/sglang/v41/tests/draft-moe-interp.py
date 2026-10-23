"""Triton-interpreter (CPU) checks of the DSpark draft MoE chain (`dsv41_mxfp4_draft.py`): `python draft-moe-interp.py <runtime-or-overlay>`.

Not registered in the build (minutes of interpreter time); results are kept next to the patch.  E = 16 experts at the real draft
stage geometry (hidden 5120, per-rank intermediate 576, top-3).  The interpreter's dot is numpy's, so gate/up and down are compared
with an fp64 dequantised reference to one bf16 ulp; S (SwiGLU) and C (combine) are compared bitwise with the torch bf16 expressions
of the installed fallback on the chain's own intermediates (S uses numpy's exp in place of ocml's, so a handful of 1-ulp bf16
differences would be an interpreter artefact; the GPU test `draft-moe-gpu.py` checks the real kernels bitwise).

  rows 3 and 6 (the draft rows at batch 1 and 2), rows 1, 2, 4, 5, 7, 8;  live-row mask (rows 4/3, 8/6): live rows unchanged, dead rows zero.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import numpy as np
import torch
import torch.nn.functional as F
import triton.language as tl
import triton.runtime.interpreter as _interp

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"

_orig_cast_impl = _interp.InterpreterBuilder.cast_impl


def _cast_impl(self, src, dst_type):
    if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:   # stock path truncates; use RNE
        b = torch.from_numpy(np.ascontiguousarray(src.data)).to(torch.bfloat16)
        return _interp.TensorHandle(b.view(torch.int16).numpy().view(np.uint16).copy(), dst_type.scalar)
    return _orig_cast_impl(self, src, dst_type)


_interp.InterpreterBuilder.cast_impl = _cast_impl
for _n in ("create_fp_trunc", "create_fp_ext", "create_si_to_fp", "create_ui_to_fp", "create_fp_to_si", "create_fp_to_ui"):
    setattr(_interp.InterpreterBuilder, _n, (lambda self, src, dst_type: self.cast_impl(src, dst_type)))

import types

spec = importlib.util.spec_from_file_location("dsv41_mxfp4_draft_interp", SITE / "sglang/kernels/ops/moe/dsv41_mxfp4_draft.py")
D = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = D
spec.loader.exec_module(D)
D.libdevice = types.SimpleNamespace(exp=lambda x: tl.exp(x))   # the interpreter has no ocml; numpy's fp32 exp stands in

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def bits(t):
    return t.contiguous().view(torch.int16)


def ulp_ok(a, b, label, frac=1e-3):
    a, b = a.float(), b.float()
    tol = 2.0 ** -7 * b.abs() + 2.0 ** -14 * b.pow(2).mean().sqrt()
    bad = int(((a - b).abs() > tol).sum())
    check(label, bad == 0, f"beyond-1ulp={bad}")


VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def deq(w, s):
    codes = torch.stack((w & 15, w >> 4), -1).flatten(-2).long()
    return VALUES[codes] * torch.exp2(s.double() - 127).repeat_interleave(32, -1)


E, K1, H, N1, N2, TOPK = 16, D.HIDDEN, D.INTER, D.N_GATE_UP, D.N_DOWN, D.TOPK
assert TOPK == 3
g = torch.Generator().manual_seed(7)
w13 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, generator=g)
w2 = torch.randint(0, 256, (E, N2, H // 2), dtype=torch.uint8, generator=g)
s13 = torch.randint(118, 125, (E, N1, K1 // 32), dtype=torch.uint8, generator=g)
s2 = torch.randint(118, 125, (E, N2, H // 32), dtype=torch.uint8, generator=g)
D13 = [deq(w13[e], s13[e]) for e in range(E)]
D2 = [deq(w2[e], s2[e]) for e in range(E)]


def routes(m, seed, repeat_rows=False):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(m)]).to(torch.int32)
    if repeat_rows and m > 1:
        ids[1] = ids[0].flip(0)
    w = torch.rand(m, TOPK, generator=gen) + 0.05
    return ids, (w / w.sum(1, keepdim=True)).float()


def chain(x, tw, ids, live=None, factor=1.5, out=None):
    t = None if live is None else torch.tensor([live], dtype=torch.int32)
    return D.decode_draft_experts(x, w13, w2, s13.view(torch.float8_e8m0fnu), s2.view(torch.float8_e8m0fnu), tw, ids, factor,
                                  out=out, live_rows=t, stages=True)


for m in (3, 6, 1, 2, 4, 5, 7, 8):
    t0 = time.time()
    ids, tw = routes(m, 100 + m, repeat_rows=True)
    x = (torch.randn(m, K1) * torch.exp(0.5 * torch.randn(K1))).bfloat16()
    gu, act, ro, out = chain(x, tw, ids)
    # G vs fp64 emulation of the dequantised weights
    gref = torch.zeros(m, TOPK, N1)
    for t in range(m):
        for s in range(TOPK):
            gref[t, s] = (D13[int(ids[t, s])] @ x[t].double()).float()
    ulp_ok(gu, gref.bfloat16(), f"rows={m} G vs fp64 emulation")
    # S bitwise vs the torch bf16 sequence on the kernel's own gate/up
    gate, up = gu.chunk(2, dim=-1)
    sref = (F.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)).reshape(m * TOPK, H)
    nbad = int((bits(act) != bits(sref)).sum())
    check(f"rows={m} S vs torch bf16 swiglu", nbad <= 2, f"mismatching elements={nbad} of {act.numel()}")
    # D vs fp64: (A @ W^T) * w, bf16
    rref = torch.zeros(m, TOPK, N2)
    for t in range(m):
        for s in range(TOPK):
            rref[t, s] = ((D2[int(ids[t, s])] @ act[t * TOPK + s].double()).float() * tw[t, s])
    ulp_ok(ro, rref.bfloat16(), f"rows={m} D (weighted) vs fp64 emulation")
    # C bitwise vs torch sum + mul_
    cref = ro.sum(dim=1)
    cref.mul_(1.5)
    check(f"rows={m} C vs torch sum(dim=1).mul_", torch.equal(bits(out), bits(cref)), f"mismatch={int((bits(out) != bits(cref)).sum())}")
    # no scaling factor
    _, _, _, out_nf = chain(x, tw, ids, factor=None)
    check(f"rows={m} C without routed_scaling_factor", torch.equal(bits(out_nf), bits(ro.sum(dim=1))))
    # in-place output (the installed tail copies the result into hidden_states)
    xi = x.clone()
    _, _, _, out_ip = chain(xi, tw, ids, out=xi)
    check(f"rows={m} in-place result", out_ip is xi and torch.equal(bits(xi), bits(out)))
    print(f"rows={m} done in {time.time() - t0:.0f}s", flush=True)

for m, live in ((4, 3), (8, 6)):
    t0 = time.time()
    ids, tw = routes(m, 200 + m)
    ids[live:] = ids[0]                       # dead rows share experts with a live row
    x = (torch.randn(m, K1) * torch.exp(0.5 * torch.randn(K1))).bfloat16()
    ref = chain(x, tw, ids)
    msk = chain(x, tw, ids, live=live)
    for name, r, k, rows in zip(("gate/up", "swiglu", "down", "combine"), ref, msk, (live, live * TOPK, live, live)):
        check(f"mask rows={m} live={live} {name}: live rows bitwise unchanged", torch.equal(bits(r[:rows]), bits(k[:rows])))
    check(f"mask rows={m} live={live}: dead gate/up, down and output rows are zero",
          bool((msk[0][live:].float() == 0).all() and (msk[2][live:].float() == 0).all() and (msk[3][live:].float() == 0).all()))
    print(f"mask rows={m} live={live} done in {time.time() - t0:.0f}s", flush=True)

print("SUMMARY:", "ALL PASS" if not FAILS else f"FAILURES {FAILS}")
sys.exit(1 if FAILS else 0)
