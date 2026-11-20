"""Installed prefill dispatch regression with CPU tensors and device stubs.

Checks actual caller/helper guards and launch metadata without a GPU. The public
FP8 test separately checks all conversion bits and changed-data graph replay.
"""

import ast
import math
import struct
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"


def source_function(relative, name, namespace):
    path = SITE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 -- exercise installed functions with device stubs
    return namespace[name]


class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="cuda")
        self.is_cuda = True
        self.contiguous = True

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def new_empty(self, shape, dtype):
        return Tensor(shape, dtype)

    def stride(self, index):
        return math.prod(self.shape[index % len(self.shape) + 1 :])

    def view_as(self, _):
        return self


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


props = types.SimpleNamespace(gcnArchName="gfx1151:sramecc-")
torch_stub = types.SimpleNamespace(
    float8_e4m3fn=torch.float8_e4m3fn,
    bfloat16=torch.bfloat16,
    float16=torch.float16,
    float32=torch.float32,
    float=torch.float,
    int=torch.int,
    version=types.SimpleNamespace(hip="test"),
    cuda=types.SimpleNamespace(get_device_properties=lambda _: props),
    empty_like=lambda value: Tensor(value.shape, value.dtype),
)
fp8_kernel = Kernel()
fp8_globals = {
    "torch": torch_stub,
    "triton": types.SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
    "get_w8a8_block_fp8_configs": lambda *args: None,
    "get_platform": lambda: types.SimpleNamespace(is_sm90=False),
    "_is_gfx1250": False,
    "_w8a8_block_fp8_matmul": fp8_kernel,
    "select_w8a8_block_fp8_matmul_kernel": lambda *args: fp8_kernel,
}
for function in (
    "prepare_block_fp8_matmul_inputs",
    "_use_gfx1151_fp8_prefill_cast",
    "w8a8_block_fp8_matmul_triton",
):
    source_function("sglang/kernels/ops/gemm/fp8_kernel.py", function, fp8_globals)


def fp8_case(m, n, k, expected, dtype=torch.float8_e4m3fn, block=(32, 32)):
    fp8_kernel.calls.clear()
    fp8_globals["w8a8_block_fp8_matmul_triton"](
        Tensor((m, k), dtype),
        Tensor((n, k), dtype),
        Tensor((m, k // block[1]), torch.float32),
        Tensor((n // block[0], k // block[1]), torch.float32),
        list(block),
        torch.bfloat16,
    )
    assert len(fp8_kernel.calls) == 1
    grid, _, kwargs = fp8_kernel.calls[0]
    assert bool(kwargs.get("EXACT_E4M3FN_TO_FP16", False)) == expected
    assert [
        kwargs[key]
        for key in (
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        )
    ] == [64, block[0], block[1], 32, 4, 3]
    assert grid(kwargs) == (((m + 63) // 64) * (n // block[0]),)


for n, k in (
    (1792, 5120),
    (8192, 1280),
    (5120, 2048),
    (1152, 5120),
    (5120, 576),
    (25600, 6144),
    (4096, 1280),
):
    for m in (512, 1306, 1536):
        fp8_case(m, n, k, True)
for m in (1, 2, 3, 4, 8, 15):
    fp8_case(m, 5120, 2048, False)
fp8_case(512, 512, 512, False)
fp8_case(512, 5120, 2048, False, block=(64, 64))
for dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2):
    fp8_case(512, 5120, 2048, False, dtype=dtype)
props.gcnArchName = "gfx942"
fp8_case(512, 5120, 2048, False)
props.gcnArchName = "gfx1151"

# Mathematical finite-code representation, including negative zero. CPU NaN
# payload conventions differ; the device test checks canonical signed NaN bits.
raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
fp8 = raw.view(torch.float8_e4m3fn)
fp16 = fp8.to(torch.float16)
for code in range(256):
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        assert torch.isnan(fp16[code])
        assert bool(torch.signbit(fp16[code])) == bool(code & 128)
        continue
    value = (
        mantissa * 2.0**-9 if exponent == 0 else (8 + mantissa) * 2.0 ** (exponent - 10)
    )
    bits = struct.unpack("<H", struct.pack("<e", value))[0] | ((code & 128) << 8)
    assert int(fp16.view(torch.uint16)[code]) == bits
    assert struct.unpack("<e", struct.pack("<H", bits & 0x7FFF))[0] == value
finite = (raw & 127) != 127
assert torch.equal(
    fp16.view(torch.int16)[finite], fp8.float().half().view(torch.int16)[finite]
)

hc_kernel = Kernel()
hc_globals = {"torch": torch_stub, "_hc_post_c2": hc_kernel}
helper = source_function(
    "sglang/kernels/ops/layernorm/native_hc_post.py", "native_hc_post_c2", hc_globals
)
false_option = types.SimpleNamespace(get=lambda: False)
caller_globals = {
    "torch": torch_stub,
    "_is_npu": False,
    "_is_xpu": False,
    "_is_cuda": False,
    "_is_hip": True,
    "_is_gfx95_supported": False,
    "_use_aiter": False,
    "envs": types.SimpleNamespace(
        SGLANG_OPT_USE_FLASHINFER_MHC=false_option,
        SGLANG_OPT_USE_TILELANG_MHC_POST=false_option,
    ),
}
caller = source_function("sglang/srt/models/deepseek_v4.py", "hc_post", caller_globals)
model = types.SimpleNamespace(
    config=types.SimpleNamespace(model_type="deepseek_v41"),
    hc_pre_from_prev_sublayer=True,
    hc_mult=4,
    hidden_size=5120,
)
native_module = types.ModuleType("sglang.kernels.ops.layernorm.native_hc_post")
native_module.native_hc_post_c2 = helper
fallback_module = types.ModuleType("sglang.kernels.ops.layernorm.mhc")
fallback_calls = []


def fallback(*args):
    fallback_calls.append(args)
    return args[1]


fallback_module.hc_post = fallback


def hc_inputs(rows):
    return [
        Tensor((rows, 5120), torch.bfloat16),
        Tensor((rows, 4, 5120), torch.bfloat16),
        Tensor((rows, 4), torch.float32),
        Tensor((rows, 4, 4), torch.float32),
    ]


with patch.dict(
    sys.modules,
    {
        native_module.__name__: native_module,
        fallback_module.__name__: fallback_module,
    },
):
    for rows in (1, 2, 4, 512, 1306, 1536):
        hc_kernel.calls.clear()
        caller(model, *hc_inputs(rows))
        assert len(hc_kernel.calls) == 1
        grid, _, options = hc_kernel.calls[0]
        assert grid == (20, 4, rows)
        assert options == {
            "num_warps": 4,
            "num_stages": 1,
            "enable_fp_fusion": False,
            "allow_flush_denorm": False,
        }
        assert (((rows - 1) * 4 + 3) * 5120 + 19 * 256 + 255) == rows * 4 * 5120 - 1
    for rows in (3, 8, 511, 513, 1305, 1307, 1535, 1537):
        hc_kernel.calls.clear()
        caller(model, *hc_inputs(rows))
        assert not hc_kernel.calls
        try:
            helper(*hc_inputs(rows))
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported rows accepted: {rows}")
    for bad in ("dtype", "stride", "arch", "model"):
        args = hc_inputs(1536)
        if bad == "dtype":
            args[0].dtype = torch.float16
        elif bad == "stride":
            args[1].contiguous = False
        elif bad == "arch":
            props.gcnArchName = "gfx950"
        else:
            model.config.model_type = "deepseek_v4"
        hc_kernel.calls.clear()
        caller(model, *args)
        assert not hc_kernel.calls, bad
        props.gcnArchName = "gfx1151"
        model.config.model_type = "deepseek_v41"
assert len(fallback_calls) == 12
assert not torch.cuda.is_initialized()
print(
    "prefill dispatch PASS: native FP8 formats/shapes/tiles, HC caller/helper grids and fallbacks, FP16 exact finite codes; CUDA uninitialized"
)
