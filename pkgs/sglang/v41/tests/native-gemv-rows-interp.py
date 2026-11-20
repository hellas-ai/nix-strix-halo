"""Triton-interpreter (CPU) checks of the row-generic native FP8 GEMV: `python native-gemv-rows-interp.py <runtime-or-overlay> [baseline-gemv.py]`.

Not registered in the build (the interpreter needs minutes); run by hand, results are kept next to the patch.

  A  exact dyadic operands: every row count 1..8 (C1 / C2 / C3..C8 kernels, ROWS=1..8 constexpr) equals the float64 reference
     rounded to bf16 at every (token, output row); no element outside the live tokens is written.
  B  heavy random operands (all finite E4M3FN codes, arbitrary fp32 scales): a token's result is bitwise the same for every row
     count 1..8 (batch invariance), including tokens placed at every position of the call; three tokens equal the first three rows
     of the 4- and 8-token calls.
  D  mHC post (`native_hc_post._hc_post_c2`): rows 1..8 equal the fp32 expression reference ((c0*r0 + c1*r1) + c2*r2) + c3*r3 with
     the +0.0 identity first, `post*x + total`, one bf16 rounding) bitwise, and a row's result does not depend on the row count or
     on its position.
  C  (when a baseline file is given) the baseline kernels of patches 0014/0024 and the patched kernels agree bitwise at 1, 2, 4
     and 8 rows on the same random operands.

The interpreter's reduction order is its own, so this is evidence for indexing, masking and per-token independence, not for
GPU bit-exactness (that is `native-gemv-rows-gpu.py` / the microbench in the artifacts)."""

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import numpy as np
import torch
import triton.runtime.interpreter as _interp

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
BASELINE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NATIVE_ROWS_BASELINE")

# The stock interpreter's fp32 -> bf16 conversion truncates; replace only that (same patch as K2's CPU tests).
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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SITE))
G = load(SITE / "sglang/kernels/ops/gemm/deepseek_v41_gemv.py", "gemv_rows_interp")
FAILS = []


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        FAILS.append(name)


def bits(t):
    return t.contiguous().view(torch.int16)


# (K, BLOCK_K) of the seven projections, at a reduced N so the interpreter finishes: the kernel takes N only through the grid
# and the scale-row index, the K loop is the real one.
GEOMETRY = ((1280, 256), (576, 64), (2048, 512), (5120, 512), (15360, 512))
N = 64


def launch(m, q, qs, w, ws, k, block_k, kernels=G):
    out = torch.full((m, N), float("nan"), dtype=torch.bfloat16)
    kernels._launch_rows(q[:m].contiguous(), qs[:m].contiguous(), w, ws, out, N, k, m, block_k)
    return out


def dyadic(k, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.tensor([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=torch.uint8)       # 0, .5, 1, 2, -.5, -1, -2
    w = codes[torch.randint(0, len(codes), (N, k), generator=g)].view(torch.float8_e4m3fn)
    q = codes[torch.randint(0, len(codes), (8, k), generator=g)].view(torch.float8_e4m3fn)
    ws = torch.exp2(torch.randint(-2, 3, (N // 32, k // 32), generator=g).float())
    qs = torch.exp2(torch.randint(-2, 3, (8, k // 32), generator=g).float())
    return q, qs, w, ws


def reference(q, qs, w, ws, k):
    qf = q.float().double() * qs.double().repeat_interleave(32, 1)                      # [t, K]
    wf = w.float().double() * ws.double().repeat_interleave(32, 0).repeat_interleave(32, 1)  # [N, K]
    return (qf @ wf.T).to(torch.float32).to(torch.bfloat16)


def heavy(k, seed):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (N, k), dtype=torch.uint8, generator=g)
    rawq = torch.randint(0, 256, (8, k), dtype=torch.uint8, generator=g)
    for t in (raw, rawq):
        t[(t & 0x7F) == 0x7F] = 0x38                                   # keep it finite: NaN rows would compare unequal
    ws = torch.rand((N // 32, k // 32), generator=g) * 4 + 0.01
    qs = torch.exp2(torch.randint(-6, 7, (8, k // 32), generator=g).float())
    return rawq.view(torch.float8_e4m3fn), qs, raw.view(torch.float8_e4m3fn), ws


t0 = time.time()
for k, block_k in GEOMETRY:
    q, qs, w, ws = dyadic(k, 1000 + k)
    ref = reference(q, qs, w, ws, k)
    outs = {m: launch(m, q, qs, w, ws, k, block_k) for m in range(1, 9)}
    for m in range(1, 9):
        out = outs[m]
        check(f"A K={k} BLOCK_K={block_k} rows={m}: equals the float64 reference at every element, no NaN left",
              torch.equal(bits(out), bits(ref[:m])) and bool(torch.isfinite(out.float()).all()))
    print(f"A K={k} done {time.time() - t0:.0f}s", flush=True)

for k, block_k in GEOMETRY:
    q, qs, w, ws = heavy(k, 2000 + k)
    full = launch(8, q, qs, w, ws, k, block_k)
    outs = {m: launch(m, q, qs, w, ws, k, block_k) for m in range(1, 9)}
    check(f"B K={k} BLOCK_K={block_k}: finite, no NaN left", all(bool(torch.isfinite(o.float()).all()) for o in outs.values()))
    for m in range(1, 9):
        check(f"B K={k} BLOCK_K={block_k} rows={m}: every token bitwise the 8-token call's", torch.equal(bits(outs[m]), bits(full[:m])))
    # a token's result must not depend on its position or its neighbours: rotate the token order
    perm = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    rotated = launch(8, q[perm], qs[perm], w, ws, k, block_k)
    check(f"B K={k} BLOCK_K={block_k}: permuted 8 tokens, results permute bitwise", torch.equal(bits(rotated), bits(full[perm])))
    for m in (3, 5, 6, 7):
        sub = launch(m, q[perm], qs[perm], w, ws, k, block_k)
        check(f"B K={k} BLOCK_K={block_k} rows={m} of the permuted order equal the permuted 8-token rows",
              torch.equal(bits(sub), bits(full[perm][:m])))
    print(f"B K={k} done {time.time() - t0:.0f}s", flush=True)

if BASELINE:
    OLD = load(BASELINE, "gemv_rows_interp_baseline")
    for k, block_k in GEOMETRY:
        q, qs, w, ws = heavy(k, 3000 + k)
        for m in (1, 2, 4, 8):
            new = launch(m, q, qs, w, ws, k, block_k)
            old = torch.full((m, N), float("nan"), dtype=torch.bfloat16)
            if m == 1:
                OLD._native_fp8_gemv_c1[(N // 4,)](q[:1].contiguous(), qs[:1].contiguous(), w, ws, old, N, k, 4, block_k)
            elif m == 2:
                OLD._native_c2_gemv[(N // 8,)](q[:2].contiguous(), qs[:2].contiguous(), w, ws, old, N, k, 8, block_k)
            else:
                OLD._native_c4_c8_gemv[(N // 8,)](q[:m].contiguous(), qs[:m].contiguous(), w, ws, old, N, k, m, 8, block_k)
            check(f"C K={k} BLOCK_K={block_k} rows={m}: baseline kernel bitwise", torch.equal(bits(new), bits(old)))
else:
    print("C skipped (no baseline file given)")

# ------------------------------------------------------------------ D: HC post
import ast

HC = load(SITE / "sglang/kernels/ops/layernorm/native_hc_post.py", "hc_post_rows_interp")


def source_function(relative, name):
    path = SITE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


mhc_post_torch = source_function("sglang/kernels/ops/layernorm/mhc.py", "_mhc_post_torch")   # the generic fallback, verbatim


def hc_inputs(rows, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, hidden, generator=g) * 2).to(torch.bfloat16)
    residual = (torch.randn(rows, 4, hidden, generator=g) * torch.exp(torch.randn(rows, 4, hidden, generator=g))).to(torch.bfloat16)
    post = torch.rand(rows, 4, generator=g) * 2
    comb = torch.rand(rows, 4, 4, generator=g) / 4
    return x, residual, post, comb


def hc_reference(x, residual, post, comb):
    """The kernel's documented expression: fp32 products, +0.0 identity, sequential combine, one bf16 rounding."""
    xf, rf = x.float(), residual.float()
    out = torch.empty(x.shape[0], 4, x.shape[1], dtype=torch.float32)
    for j in range(4):
        a = [torch.zeros(()) + comb[:, i, j].unsqueeze(-1) * rf[:, i] for i in range(4)]
        out[:, j] = post[:, j].unsqueeze(-1) * xf + (((a[0] + a[1]) + a[2]) + a[3])
    return out.to(torch.bfloat16)


def hc_run(x, residual, post, comb, hidden, block=256):
    out = torch.empty_like(residual)
    HC._hc_post_c2[(hidden // block, 4, x.shape[0])](x, residual, post, comb, out, hidden, block)
    return out


def order_pattern(rows, hidden):
    """65536, 2^-9, -65536, 2^-9 with unit weights: only the sequential order gives 2^-9 (pairwise gives 0)."""
    x = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    residual = torch.zeros(rows, 4, hidden, dtype=torch.bfloat16)
    for i, value in enumerate((65536.0, 2.0**-9, -65536.0, 2.0**-9)):
        residual[:, i] = value
    return x, residual, torch.zeros(rows, 4), torch.ones(rows, 4, 4)


# real hidden size, decode rows 1..8 (grid (20, 4, M)); reduced hidden size (grid (1, 4, M)) for the row spread: the kernel's
# per-element expression and its (t, j, h-block) indexing do not depend on H.
cases = [(rows, 5120) for rows in range(1, 9)] + [
    (rows, 256) for rows in (9, 17, 100, 511, 513, 1000, 1305, 1535, 1537, 4096)
]
first = {}
for rows, hidden in cases:
    args = hc_inputs(rows, hidden, 4000 + rows)
    got = hc_run(*args, hidden)
    check(f"D HC post M={rows} H={hidden}: bitwise the fp32 expression reference", torch.equal(bits(got), bits(hc_reference(*args))))
    fallback = mhc_post_torch(args[0], args[1], args[2].view(rows, 4, 1), args[3])
    check(f"D HC post M={rows} H={hidden}: bitwise the generic fallback _mhc_post_torch (CPU ATen)", torch.equal(bits(got), bits(fallback)))
    op = order_pattern(rows, hidden)
    check(f"D HC post M={rows} H={hidden}: sequential-order probe gives 2^-9", bool((hc_run(*op, hidden).float() == 2.0**-9).all()))
    if hidden == 5120:
        first[rows] = (args, got)
big_args, big = first[8]
for rows in range(1, 8):
    sub = tuple(t[:rows].contiguous() for t in big_args)
    check(f"D HC post M={rows} of the 8-row operands: bitwise the first rows of the 8-row call", torch.equal(bits(hc_run(*sub, 5120)), bits(big[:rows])))
perm = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
check("D HC post: permuted 8 rows, results permute bitwise", torch.equal(bits(hc_run(*[t[perm].contiguous() for t in big_args], 5120)), bits(big[perm])))
print(f"D done {time.time() - t0:.0f}s", flush=True)

print(f"native GEMV rows interpreter: {'FAIL ' + str(len(FAILS)) if FAILS else 'PASS'} in {time.time() - t0:.0f}s", flush=True)
sys.exit(1 if FAILS else 0)
