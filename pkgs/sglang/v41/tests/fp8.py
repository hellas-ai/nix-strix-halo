#!/usr/bin/env python3
"""Bounded zero-row/layout and UE8M0 boundary checks for the portable quant path."""

import json
import os
import sys
from pathlib import Path

os.environ['SGLANG_USE_AITER'] = '0'

import torch
from sglang.kernels.ops.quantization.fp8_kernel import (
    _run_per_token_group_quant_8bit_kernel,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear


def internal(x, q, s):
    return _run_per_token_group_quant_8bit_kernel(
        x, q, s, 32, 1e-10, -448., 448.,
        scale_ue8m0=True, fuse_silu_and_mul=False, masked_m=None)


def main():
    import inspect
    import sglang
    runtime = Path(sys.argv[1]).resolve()
    assert Path(inspect.getfile(_run_per_token_group_quant_8bit_kernel)).resolve().is_relative_to(runtime)
    assert Path(sglang.__file__).resolve().is_relative_to(runtime)
    torch.cuda.set_per_process_memory_fraction(0.03)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx1151'
    x0 = torch.empty((0, 64), device='cuda', dtype=torch.bfloat16)
    q0, s0 = sglang_per_token_group_quant_fp8(x0, 32, scale_ue8m0=True)
    assert q0.shape == (0, 64) and s0.shape == (0, 2)
    internal(x0, q0, s0)  # raw branch also has an empty-grid guard
    x = torch.zeros((2, 64), device='cuda', dtype=torch.bfloat16)
    x[0, 0] = 1.75
    x[1, 31] = -3.5
    q, s = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
    torch.testing.assert_close(s.cpu(), torch.tensor([[2**-8, 2**-42],
                                                       [2**-7, 2**-42]], dtype=torch.float32),
                               rtol=0, atol=0)
    assert float(q[0, 0].float()) == 448. and float(q[1, 31].float()) == -448.
    try:
        internal(x, torch.empty_like(q), torch.empty((2, 1), device='cuda'))
    except AssertionError:
        pass
    else:
        raise AssertionError('bad scale shape accepted')
    try:
        internal(x, torch.empty_like(q), torch.empty((2, 4), device='cuda')[:, ::2])
    except AssertionError:
        pass
    else:
        raise AssertionError('noncontiguous scale accepted')
    w = torch.ones((33, 64), device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    ws = torch.ones((2, 2), device='cuda', dtype=torch.float32) * 2**-10
    y = triton_w8a8_block_fp8_linear(x, w, [32,32], ws, act_scale_ue8m0=True)
    assert torch.isfinite(y).all() and y.shape == (2,33)
    print(json.dumps({'event': 'complete', 'empty_rows': True,
                      'power2_boundary': True, 'bad_layout_rejected': True}), flush=True)


if __name__ == '__main__':
    main()
