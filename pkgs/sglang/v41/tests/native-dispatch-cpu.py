"""CPU checks of the gfx1151 native norm / indexer dispatch (no GPU, no Triton launch).

* The ATen reduction-geometry model behind native_rmsnorm_decode (torch_mean_block_width)
  against hand-derived values of ATen/native/hip/Reduce.cuh.
* native_rmsnorm_decode / native_routed_add_ decline everything that is not a gfx1151 CUDA call.
* The installed ``triton_w8a8_block_fp8_linear`` routes the indexer wq_b shape to
  ``native_idx_wqb_gemv`` for 1..8 rows only when opted in and leaves every other
  call (and the existing attention opt-ins) on their previous path.
* Row-generic dispatch (patch 0047): every native projection serves 1..8 rows (1 -> C1, 2 -> C2, 3..8 ->
  ``native_c4_c8_gemv``), 9+ rows stay on the generic matmul, ``SGLANG_DSV41_NATIVE_ROWS`` selects ``legacy``
  (previous 1,2,4,8 / 1,2,3,4,8 dispatch) or an explicit list, and the ``native_gemv_rows=True`` opt-in routes any
  table shape (including the DSpark draft's wq_a, wkv and main_proj) through ``native_gemv_rows``.
"""

import ast
import math
import os
import sys
import types
from pathlib import Path

import torch

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
assert not torch.cuda.is_initialized()

# ---- reduction geometry (hand-derived from Reduce.cuh setReduceConfig) -----------------
sys.path.insert(0, str(SITE))
from sglang.kernels.ops.layernorm import native_rmsnorm_decode as nr  # noqa: E402
from sglang.kernels.ops.layernorm import native_routed_add as ra  # noqa: E402

expected_width = {
    (1, 5120): 512, (2, 5120): 256, (3, 5120): 256, (4, 5120): 128, (5, 5120): 128,
    (8, 5120): 64, (16, 5120): 32,
    (1, 1280): 256, (2, 1280): 256, (3, 1280): 256, (4, 1280): 128, (8, 1280): 64, (16, 1280): 32,
    (1, 512): 128, (2, 512): 128, (4, 512): 128, (8, 512): 64,
    (1, 128): 32, (4, 128): 32, (16, 128): 32,
}
for key, bw in expected_width.items():
    assert nr.torch_mean_block_width(*key) == bw, (key, nr.torch_mean_block_width(*key), bw)
assert nr.torch_mean_block_width(1, 100) is None  # below the vectorization threshold
assert nr.torch_mean_block_width(1, 130) is None  # not a multiple of the vector width
assert nr.torch_mean_block_width(1, 5120 * 8) is None  # would split across warps / blocks

# ---- native entry points decline off-target calls --------------------------------------
x = torch.zeros(1, 5120, dtype=torch.bfloat16)
w = torch.ones(5120, dtype=torch.bfloat16)
assert nr.native_rmsnorm_decode(x, w, 1e-20) is None  # CPU tensor
assert ra.native_routed_add_(torch.zeros(1, 5120), torch.zeros(1, 5120, dtype=torch.bfloat16)) is False
assert not torch.cuda.is_initialized()

# ---- indexer wq_b dispatch in the installed triton_w8a8_block_fp8_linear ---------------


def source_function(relative, name, namespace):
    path = SITE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 -- installed function, device stubs
    return namespace[name]


class T:
    def __init__(self, shape, dtype, contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="cuda", index=0)
        self.is_cuda = True
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def view(self, *shape):
        shape = list(shape[0] if shape and isinstance(shape[0], (list, tuple)) else shape)
        if -1 in shape:
            known = math.prod(d for d in shape if d != -1)
            shape[shape.index(-1)] = math.prod(self.shape) // known
        return T(shape, self.dtype)

    def to(self, dtype=None):
        return T(self.shape, dtype or self.dtype)

    def __iadd__(self, other):
        return self


# device equality must hold between the stubs
DEVICE = types.SimpleNamespace(type="cuda", index=0)


class Rec:
    def __init__(self, tag, log):
        self.tag, self.log = tag, log

    def __call__(self, q, qs, w, ws, *rest, **options):
        self.log.append(self.tag)
        return T((q.shape[0], w.shape[0]), torch.bfloat16)


calls = []
fake_gemv = types.ModuleType("sglang.kernels.ops.gemm.deepseek_v41_gemv")
for name in (
    "engram_gemv", "native_wob_gemv", "native_wqkv_a_gemv", "native_wqb_gemv",
    "native_c2_gemv", "native_shared_expert_gemv", "native_c4_c8_gemv", "native_idx_wqb_gemv",
    "native_gemv_rows",
):
    setattr(fake_gemv, name, Rec(name, calls))
sys.modules["sglang.kernels.ops.gemm.deepseek_v41_gemv"] = fake_gemv

props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc-")
torch_stub = types.SimpleNamespace(
    float8_e4m3fn=torch.float8_e4m3fn, bfloat16=torch.bfloat16, float16=torch.float16,
    float32=torch.float32, Tensor=T,
    version=types.SimpleNamespace(hip="test"),
    cuda=types.SimpleNamespace(get_device_properties=lambda _: props),
)


def fake_quant(x, group, scale_ue8m0=False, **_):
    rows, k = x.shape
    return (T((rows, k), torch.float8_e4m3fn), T((rows, k // group), torch.float32))


def fake_matmul(q, w, qs, ws, block, output_dtype=None):
    calls.append("generic")
    return T((q.shape[0], w.shape[0]), output_dtype)


ns = {
    "torch": torch_stub, "_is_hip": True,
    "sglang_per_token_group_quant_fp8": fake_quant,
    "per_token_group_quant_fp8": fake_quant,
    "w8a8_block_fp8_matmul_triton": fake_matmul,
    "List": list, "Optional": lambda x: x,
    # The WMMA decode GEMV (SGLANG_DSV41_WMMA_GEMV) is off by default; wmma-gemv-policy-cpu.py covers it.
    "_dsv41_wmma_gemv_serves": lambda shape, rows: False,
}
source_function("sglang/srt/layers/quantization/fp8_utils.py", "_v41_native_rows", ns)
linear = source_function("sglang/srt/layers/quantization/fp8_utils.py", "triton_w8a8_block_fp8_linear", ns)


def case(m, n=4096, k=1280, flag=None, bias=None, contiguous=True, props_arch="gfx1151:sramecc-", value=True, dtype=torch.float8_e4m3fn):
    props.gcnArchName = props_arch
    calls.clear()
    x = T((m, k), torch.bfloat16)
    w = T((n, k), dtype, contiguous)
    ws = T((n // 32, k // 32), torch.float32)
    # all stub devices compare equal because they share one SimpleNamespace
    for t in (x, w, ws):
        t.device = DEVICE
    kwargs = {flag: value} if flag else {}
    ns["triton_w8a8_block_fp8_linear"](x, w, [32, 32], ws, bias=bias, act_scale_ue8m0=True, **kwargs)
    assert len(calls) == 1, calls
    return calls[0]


ns["triton_w8a8_block_fp8_linear"] = linear
# fake_quant returns fresh stubs whose device must equal the weight's
_orig_quant = fake_quant


def quant_with_device(x, group, scale_ue8m0=False, **kw):
    q, s = _orig_quant(x, group, scale_ue8m0)
    q.device = s.device = DEVICE
    return q, s


ns["sglang_per_token_group_quant_fp8"] = quant_with_device
ns["per_token_group_quant_fp8"] = quant_with_device

for m in range(1, 9):
    assert case(m, flag="native_idx_wqb") == "native_idx_wqb_gemv", m
for m in (9, 16, 64):
    assert case(m, flag="native_idx_wqb") == "generic", m
for m in range(1, 9):
    assert case(m) == "generic", m  # not opted in
assert case(1, n=8192, flag="native_idx_wqb") == "generic"  # attention-sized weight, wrong flag
assert case(1, k=2048, n=5120, flag="native_idx_wqb") == "generic"
assert case(1, flag="native_idx_wqb", bias=T((4096,), torch.bfloat16)) == "generic"
assert case(1, flag="native_idx_wqb", contiguous=False) == "generic"
assert case(1, flag="native_idx_wqb", props_arch="gfx950") == "generic"
# existing attention opt-ins keep their entry points; three, five, six and seven rows now join four and eight
assert case(1, n=8192, flag="native_wqb_c1") == "native_wqb_gemv"
assert case(2, n=8192, flag="native_wqb_c1") == "native_c2_gemv"
for m in range(3, 9):
    assert case(m, n=8192, flag="native_wqb_c1") == "native_c4_c8_gemv", m
for m in (9, 12, 16, 512):
    assert case(m, n=8192, flag="native_wqb_c1") == "generic", m
assert case(1, n=4096, flag="native_wqb_c1") == "generic"
projections = (
    (25600, 6144, "engram_wkv_c1", True, "engram_gemv"),
    (5120, 2048, "native_wob_c1", True, "native_wob_gemv"),
    (1792, 5120, "native_wqkv_a_c1", True, "native_wqkv_a_gemv"),
    (8192, 1280, "native_wqb_c1", True, "native_wqb_gemv"),
    (1152, 5120, "native_v41_shared", "gate_up", "native_shared_expert_gemv"),
    (5120, 576, "native_v41_shared", "down", "native_shared_expert_gemv"),
)


def expected_entry(c1_name, m, shared):
    if m == 1:
        return c1_name
    if m == 2:
        return "native_shared_expert_gemv" if shared else "native_c2_gemv"
    return "native_c4_c8_gemv"


for n, k, flag, value, c1_name in projections:
    shared = flag == "native_v41_shared"
    for m in range(1, 9):
        assert case(m, n=n, k=k, flag=flag, value=value) == expected_entry(c1_name, m, shared), (n, k, m)
    for m in (9, 16, 64, 1306):
        assert case(m, n=n, k=k, flag=flag, value=value) == "generic", (n, k, m)
    assert case(4, n=n, k=k, flag=flag, value=value, bias=T((n,), torch.bfloat16)) == "generic"
    assert case(3, n=n, k=k, flag=flag, value=value, contiguous=False) == "generic"
    assert case(3, n=n, k=k, flag=flag, value=value, props_arch="gfx942") == "generic"
    assert case(3, n=n, k=k, flag=flag, value=value, dtype=torch.float8_e4m3fnuz) == "generic"

# SGLANG_DSV41_NATIVE_ROWS: legacy restores the previous dispatch, a list selects exactly those rows (the new opt-in follows the same set)
legacy_dir = {}
for n, k, flag, value, c1_name in projections:
    shared = flag == "native_v41_shared"
    legacy_dir[(n, k)] = {m: case(m, n=n, k=k, flag=flag, value=value) for m in range(1, 9)}
os.environ["SGLANG_DSV41_NATIVE_ROWS"] = "legacy"
try:
    for n, k, flag, value, c1_name in projections:
        shared = flag == "native_v41_shared"
        for m in (1, 2, 4, 8):
            assert case(m, n=n, k=k, flag=flag, value=value) == expected_entry(c1_name, m, shared), ("legacy", n, k, m)
        for m in (3, 5, 6, 7, 9):
            assert case(m, n=n, k=k, flag=flag, value=value) == "generic", ("legacy", n, k, m)
    for m in (1, 2, 3, 4, 8):
        assert case(m, flag="native_idx_wqb") == "native_idx_wqb_gemv", ("legacy idx", m)
    for m in (5, 6, 7, 9):
        assert case(m, flag="native_idx_wqb") == "generic", ("legacy idx", m)
    assert case(3, n=4096, k=1280, flag="native_gemv_rows") == "generic", "legacy: generic opt-in only serves its legacy rows"
    assert case(4, n=4096, k=1280, flag="native_gemv_rows") == "native_gemv_rows", "legacy rows of the generic opt-in"
    os.environ["SGLANG_DSV41_NATIVE_ROWS"] = "2,3-4"
    for m in range(1, 9):
        want = m in (2, 3, 4)
        assert (case(m, n=8192, flag="native_wqb_c1") != "generic") == want, ("list", m)
        assert (case(m, flag="native_idx_wqb") != "generic") == want, ("list idx", m)
    os.environ["SGLANG_DSV41_NATIVE_ROWS"] = "all"
    for m in range(1, 9):
        assert case(m, flag="native_idx_wqb") == "native_idx_wqb_gemv", m
finally:
    del os.environ["SGLANG_DSV41_NATIVE_ROWS"]

# the generic opt-in `native_gemv_rows=True` (DSpark draft wq_a / wkv / main_proj and every other table shape): exactly the
# shapes of PROJECTION_BLOCK_K, rows 1..8, the same guards as the per-shape flags
from sglang.kernels.ops.gemm.dsv41_native_rows import PROJECTION_BLOCK_K  # noqa: E402

table = sorted(PROJECTION_BLOCK_K)
assert (1280, 5120) in table and (512, 5120) in table and (5120, 15360) in table, table
assert (1536, 5120) in table, table      # the three draft stages' wkv stacked by CommitKvProj
assert len(table) == 11, table
for n, k in table:
    for m in range(1, 9):
        assert case(m, n=n, k=k, flag="native_gemv_rows") == "native_gemv_rows", (n, k, m)
    for m in (9, 16, 512):
        assert case(m, n=n, k=k, flag="native_gemv_rows") == "generic", (n, k, m)
    assert case(3, n=n, k=k, flag="native_gemv_rows", bias=T((n,), torch.bfloat16)) == "generic"
    assert case(3, n=n, k=k, flag="native_gemv_rows", contiguous=False) == "generic"
    assert case(3, n=n, k=k, flag="native_gemv_rows", props_arch="gfx950") == "generic"
    assert case(3, n=n, k=k, flag="native_gemv_rows", dtype=torch.float8_e4m3fnuz) == "generic"
    assert case(3, n=n, k=k) == "generic"                                     # unflagged
for n, k in ((2816, 2048), (512, 1024), (4100, 1280), (4096, 1296), (1280, 5121), (128, 32), (5120, 15361)):
    assert case(3, n=n, k=k, flag="native_gemv_rows") == "generic", (n, k)   # not in the table
# a shape flag keeps its own entry point when both are set
assert case(3, n=8192, flag="native_wqb_c1") == "native_c4_c8_gemv"
assert not torch.cuda.is_initialized()
print("native dispatch CPU PASS: reduction geometry table, off-target declines, rows 1..8 routing of every native projection, legacy/list switch, table opt-in (draft shapes), fallbacks")
