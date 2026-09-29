"""Exercise the installed plugin and native Metal ABI without model downloads."""

import importlib.metadata
import json

import mlx.core as mx
import vllm_metal
from vllm_metal.metal import get_ops


assert mx.metal.is_available(), "Metal device unavailable"
assert mx.distributed.is_available("jaccl"), "MLX was built without JACCL"
assert vllm_metal.register() == "vllm_metal.platform.MetalPlatform"
mx.set_default_device(mx.gpu)

# Loading the prebuilt extension and metallibs catches mismatched MLX private
# headers and unresolved dylibs that a top-level Python import cannot detect.
ops = get_ops()
slots = mx.array([1, 19, 31], dtype=mx.int64)
for dtype in (mx.float32, mx.float16, mx.bfloat16):
    keys = mx.arange(3 * 2 * 64).reshape(3, 2, 64).astype(dtype)
    values = -keys
    shape = (2, 16, 2, 64)
    actual_keys, actual_values = ops.reshape_and_cache(
        keys, values, mx.zeros(shape, dtype=dtype), mx.zeros(shape, dtype=dtype), slots
    )
    expected_keys = mx.zeros((32, 2, 64), dtype=dtype)
    expected_keys[slots] = keys
    expected_keys = expected_keys.reshape(shape)
    mx.eval(actual_keys, actual_values, expected_keys)
    assert bool(mx.array_equal(actual_keys, expected_keys)), f"key scatter: {dtype}"
    assert bool(mx.array_equal(actual_values, -expected_keys)), f"value scatter: {dtype}"

print(json.dumps({
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("vllm", "vllm-metal", "mlx", "mlx-metal", "mlx-lm")
    },
    "metal": True,
    "jaccl": True,
    "native_cache_scatter": "exact for float32, float16, bfloat16",
}, sort_keys=True))
