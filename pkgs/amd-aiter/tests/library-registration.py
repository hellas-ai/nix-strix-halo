"""Exercise CPU/fake dispatch and teardown without importing GPU-only AITER ops."""

import importlib.util
import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

spec = importlib.util.spec_from_file_location("isolated_aiter_guard", sys.argv[1])
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@guard.torch_compile_guard(mutates_args=[])
def registration_probe(x: torch.Tensor) -> torch.Tensor:
    return x + 1


op = torch.ops.aiter.registration_probe
x = torch.arange(5, dtype=torch.float32)
torch.testing.assert_close(op(x), x + 1, rtol=0, atol=0)
with FakeTensorMode():
    y = op(torch.empty(7))
    assert y.shape == (7,) and y.dtype == torch.float32

# A namespaced schema causes Library to track aiter::aiter::registration_probe.
# Dispatch still works, but Torch cannot remove that name during destruction.
guard.aiter_lib._destroy()
assert not hasattr(torch.ops.aiter, "registration_probe")
assert not torch.cuda.is_initialized()
