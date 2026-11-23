"""WMMA decode GEMV (CPU, no GPU): E4M3 -> fp16 bit move, policy table, dispatch guard.

1. For every finite E4M3FN code, the fp16 value of ((code & 0x7f) << 7) | ((code & 0x80) << 8) times 2**8 equals the
   torch E4M3FN value exactly (normals, subnormals, both zeros).
2. Every POLICY entry tiles its K exactly (K % (splits * 32 * groups) == 0), its N by BLOCK_N, and serves() follows the
   minimum-row column and the eight-row ceiling.
3. fp8_utils dispatches to it only behind SGLANG_DSV41_WMMA_GEMV via _dsv41_wmma_gemv_serves, before the native branches.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import torch

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
MOD = SITE / "sglang/kernels/ops/gemm/dsv41_wmma_gemv.py"
FP8 = SITE / "sglang/srt/layers/quantization/fp8_utils.py"

codes = np.arange(256, dtype=np.uint16)
finite = (codes & 0x7F) != 0x7F
bits = ((codes & 0x7F) << 7) | ((codes & 0x80) << 8)
got = bits.astype(np.uint16).view(np.float16).astype(np.float64) * 256.0
want = torch.tensor(codes.astype(np.uint8)).view(torch.float8_e4m3fn).to(torch.float64).numpy()
assert np.array_equal(np.signbit(got[finite]), np.signbit(want[finite])), "sign mismatch"
assert np.array_equal(got[finite], want[finite]), np.nonzero(got[finite] != want[finite])

tree = ast.parse(MOD.read_text(), filename=str(MOD))
keep = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("POLICY", "MAX_ROWS")]
keep += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serves"]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(MOD), "exec"), ns)
policy, serves, max_rows = ns["POLICY"], ns["serves"], ns["MAX_ROWS"]
assert max_rows == 8
for (n, k), (block_n, splits, groups, warps, stages, min_rows) in policy.items():
    assert n % block_n == 0 and n % 32 == 0, (n, block_n)
    assert k % (splits * 32 * groups) == 0, (n, k, splits, groups)
    assert 2 <= min_rows <= 8 and warps in (2, 4, 8) and stages in (1, 2)
    for rows in range(0, 12):
        assert serves((n, k), rows) == (min_rows <= rows <= 8), (n, k, rows)
assert not serves((1280, 5120), 4), "draft-only shapes stay on the native kernels"

src = FP8.read_text()
assert "def _dsv41_wmma_gemv_serves(" in src and "dsv41_wmma_gemv.enabled()" in src
first_native = src.index("output = engram_gemv(")
wmma_call = src.index("output = wmma_gemv(q_input, x_scale, weight, weight_scale)")
assert wmma_call < first_native, "WMMA branch must precede the native branches"
assert 'os.environ.get("SGLANG_DSV41_WMMA_GEMV", "0") == "1"' in MOD.read_text()
print("wmma gemv policy: ok")
