"""gfx1151 synthetic C8 kernel gates; not full-model or serving qualification.

Gates are declared before execution: exact equality only within the same kernel
and reduction order. WMMA and C1 GEMV intentionally use different reductions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Match the plain launcher; set before importing installed dispatch modules.
os.environ['SGLANG_GLM53_WMMA_GEMV'] = '1'
os.environ['SGLANG_GLM53_ROW_INVARIANT_FP8_GEMV'] = '1'
os.environ['SGLANG_GLM53_MOE_GEMV_MAX_ROWS'] = '16'

import torch
from sglang.kernels.ops.gemm import glm53_wmma_gemv as wmma
from sglang.kernels.ops.moe import fused_moe_triton_kernels as moe
from sglang.kernels.ops.quantization import fp8_kernel as fp8

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise SystemExit('output already exists')
if not torch.cuda.is_available():
    raise SystemExit('requires a free gfx1151 GPU')
torch.cuda.set_device(0)
arch = torch.cuda.get_device_properties(0).gcnArchName
if arch.split(':')[0] != 'gfx1151':
    raise SystemExit(f'expected gfx1151, got {arch}')
torch.manual_seed(91738)


def codes(shape):
    raw = torch.randint(0, 256, shape, device='cuda', dtype=torch.int32)
    return torch.where((raw & 127) == 127, raw - 1, raw).to(torch.uint8).view(torch.float8_e4m3fn)


def exact(label, expected, actual):
    torch.cuda.synchronize()
    if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
        raise AssertionError((label, 'nonfinite'))
    if not torch.equal(expected, actual):
        raise AssertionError((label, 'mismatch', int((expected != actual).sum())))


def graph_run(call):
    # Prime compilation before capture; capture allocations in PyTorch's graph pool.
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    return output.clone()


results = []
shapes = ((6144, 4096), (4096, 3072), (2048, 4096), (4096, 1536),
          (4096, 4096), (1024, 4096), (4096, 512))
for n, k in shapes:
    w = codes((n, k))
    ws = torch.rand((n // 128, k // 128), device='cuda') * .01
    x = codes((8, k))
    xs = torch.rand((8, k // 128), device='cuda') * .01
    kg = min(16, (k // 128) & -(k // 128))
    one = torch.empty((1, n), device='cuda', dtype=torch.bfloat16)
    fp8._gfx1151_block_fp8_gemv[(n // 2,)](
        x[0], w, one[0], xs[0], ws, n, k, 2, kg, num_warps=4, num_stages=1)
    call_one = lambda: fp8.w8a8_block_fp8_matmul_triton(x[:1], w, xs[:1], ws, [128, 128], torch.bfloat16)
    exact(('c1', n, k), one, call_one())
    exact(('c1-graph', n, k), one, graph_run(call_one))
    results.append({'kind': 'ordinary-c1-gemv', 'n': n, 'k': k, 'rows': 1, 'exact': True})
    for rows in (2, 4, 8):
        # Two-row reference uses the identical WMMA kernel and padded tile.
        reference = torch.cat([wmma.wmma_gemv(x[i:i+1].expand(2, -1).contiguous(),
                                              xs[i:i+1].expand(2, -1).contiguous(), w, ws)[:1]
                               for i in range(rows)])
        call = lambda: fp8.w8a8_block_fp8_matmul_triton(x[:rows], w, xs[:rows], ws, [128, 128], torch.bfloat16)
        direct = wmma.wmma_gemv(x[:rows], xs[:rows], w, ws)
        exact(('wmma-direct', n, k, rows), reference, direct)
        exact(('wmma-dispatch', n, k, rows), reference, call())
        exact(('wmma-graph', n, k, rows), reference, graph_run(call))
        results.append({'kind': 'ordinary-wmma', 'n': n, 'k': k, 'rows': rows, 'exact': True})

# Direct installed MoE GEMV, 8 synthetic experts instead of the full 288; the
# arithmetic uses the real TP4 geometry and actual expert-ID/route handling.
# Comparing C8 to two C4 invocations retains identical per-route reduction.
for n, k, top_k, mul in ((1024, 4096, 8, False), (4096, 512, 1, True)):
    w = codes((8, n, k))
    ws = torch.rand((8, n // 128, k // 128), device='cuda') * .01
    x = codes((8 * 8 // top_k, k))
    xs = torch.rand((8 * 8 // top_k, k // 128), device='cuda') * .01
    ids = torch.randint(0, 8, (8, 8), device='cuda', dtype=torch.int32)
    ids[0, 0] = -1
    ids[1, 1] = ids[1, 0]  # repeated route
    routes = torch.rand((8, 8), device='cuda', dtype=torch.float32)
    routes[2, 0] = 0
    kg = min(16, (k // 128) & -(k // 128))

    def run(start, count):
        xx = x[start * 8 // top_k:(start + count) * 8 // top_k]
        ss = xs[start * 8 // top_k:(start + count) * 8 // top_k]
        ii = ids[start:start + count]
        rr = routes[start:start + count]
        out = torch.empty((count * 8, n), device='cuda', dtype=torch.bfloat16)
        moe._gfx1151_fp8_moe_gemv[(n // 2, count * 8)](
            xx, w, out, ss, ws, ii, rr, n, k, ss.stride(0), ss.stride(1), top_k, mul, 2, kg,
            num_warps=4, num_stages=1)
        return out

    reference = torch.cat((run(0, 4), run(4, 4)))
    exact(('moe-c8', n, k), reference, run(0, 8))
    exact(('moe-c8-graph', n, k), reference, graph_run(lambda: run(0, 8)))
    results.append({'kind': 'direct-moe-gemv', 'n': n, 'k': k, 'rows': 8,
                    'experts': 8, 'top_k': top_k, 'mul_route': mul, 'exact': True})


def file_hash(module):
    path = Path(module.__file__).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

report = {'status': 'PASS_SYNTHETIC_KERNEL_GATES', 'target': arch, 'seed': 91738,
          'modules': {m.__name__: file_hash(m) for m in (wmma, moe, fp8)}, 'cases': results,
          'serving_qualified': False, 'moe_dispatch_qualified': False,
          'note': 'Synthetic inputs; WMMA compared only to same WMMA arithmetic, MoE direct kernel to C4.'}
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
