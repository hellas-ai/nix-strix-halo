"""CPU-only AST dispatch check for the qualified eight-row GEMV launches."""

import ast
import sys
import types
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
SITE = ROOT if (ROOT / "sglang").is_dir() else ROOT / "lib/python3.13/site-packages"
SOURCE = SITE / "sglang/kernels/ops/gemm/deepseek_v41_gemv.py"
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
names = {
    "_multirow_output_tile", "_multirow_launch", "_launch_rows", "native_c4_c8_gemv"
}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in nodes} == names

records = []


class Kernel:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            records.append((self.name, grid, args, kwargs))

        return launch


DEVICE = types.SimpleNamespace(type="cuda")


class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.device = DEVICE

    def is_contiguous(self):
        return True


torch = types.SimpleNamespace(
    float8_e4m3fn="fp8", float32="f32", bfloat16="bf16",
    version=types.SimpleNamespace(hip="test"),
    cuda=types.SimpleNamespace(
        get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx1151")
    ),
    empty=lambda shape, dtype, device: Tensor(shape, dtype),
)
scope = {
    "torch": torch,
    "c1_rows": lambda: 4,
    "_native_fp8_gemv_c1": Kernel("c1"),
    "_native_c2_gemv": Kernel("c2"),
    "_native_c4_c8_gemv": Kernel("multi"),
}
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
exec(compile(module, str(SOURCE), "exec"), scope)  # noqa: S102 -- source under test

block_k_by_shape = {
    (25600, 6144): 512, (5120, 2048): 512, (1792, 5120): 512,
    (8192, 1280): 256, (1152, 5120): 512, (5120, 576): 64,
    (4096, 1280): 256, (1280, 5120): 512, (512, 5120): 512,
    (5120, 15360): 512, (1536, 5120): 512,
}
winners = {
    (1792, 5120): (16, 512, 8, 2),
    (8192, 1280): (16, 256, 4, 1),
    (5120, 2048): (8, 512, 8, 2),
    (1152, 5120): (16, 512, 8, 3),
    (5120, 576): (4, 64, 2, 1),
}


def expected(n, k, m):
    bk = block_k_by_shape[(n, k)]
    if m == 8 and (n, k) in winners:
        return winners[(n, k)]
    return (4 if m == 8 and (n, k) == (25600, 6144) else 8, bk, 4, 1)


def check_call(n, k, m, call):
    name, grid, args, options = call
    if m == 1:
        assert name == "c1" and grid == (n // 4,)
        assert args[-2:] == (4, block_k_by_shape[(n, k)])
        assert options == {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
    elif m == 2:
        assert name == "c2" and grid == (n // 8,)
        assert args[-2:] == (8, block_k_by_shape[(n, k)])
        assert options == {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
    else:
        rows, bk, warps, stages = expected(n, k, m)
        assert name == "multi" and grid == (n // rows,)
        assert args[-3:] == (m, rows, bk), (n, k, m, args)
        assert options == {"num_warps": warps, "num_stages": stages, "enable_fp_fusion": False}
    assert "waves_per_eu" not in options


for (n, k), bk in block_k_by_shape.items():
    for m in range(1, 9):
        assert scope["_multirow_launch"](n, k, m, bk) == expected(n, k, m)
        records.clear()
        scope["_launch_rows"](None, None, None, None, None, n, k, m, bk)
        assert len(records) == 1
        check_call(n, k, m, records[0])

# The older flagged entry point uses the same per-shape policy.
for n, k in list(block_k_by_shape)[:6]:
    for m in range(3, 9):
        q = Tensor((m, k), "fp8")
        qs = Tensor((m, k // 32), "f32")
        w = Tensor((n, k), "fp8")
        ws = Tensor((n // 32, k // 32), "f32")
        records.clear()
        out = scope["native_c4_c8_gemv"](q, qs, w, ws)
        assert out.shape == (m, n) and len(records) == 1
        check_call(n, k, m, records[0])

print("M8 GEMV launch CPU PASS: 11 shapes x rows 1-8, both dispatch entries")
