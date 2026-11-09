"""Require exact BF16 results against the unchanged production decode kernel."""
import argparse
import json
from pathlib import Path

import torch
from sglang.kernels.ops.quantization.fp8_kernel import (
    _gfx1151_block_fp8_gemv,
    _gfx1151_block_fp8_gemv_rows,
)

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
assert not args.out.exists()
torch.cuda.set_device(0)
torch.manual_seed(7351)


def finite_codes(shape):
    codes = torch.randint(0, 256, shape, device="cuda", dtype=torch.int32)
    codes = torch.where((codes & 127) == 127, codes - 1, codes)
    return codes.to(torch.uint8).view(torch.float8_e4m3fn)


results = []
for n,k in ((6144,4096), (4096,3072), (2048,4096), (4096,1536),
            (4096,4096), (1024,4096), (4096,512)):
    w = finite_codes((n,k))
    ws = torch.rand((n//128,k//128),device="cuda") * .01
    kg = min(16, (k//128) & -(k//128))
    for rows in (2,3,4,8,16):
        x = finite_codes((rows,k))
        xs = torch.rand((rows,k//128),device="cuda") * .01
        expected = torch.empty((rows,n),device="cuda",dtype=torch.bfloat16)
        actual = torch.empty_like(expected)
        for row in range(rows):
            _gfx1151_block_fp8_gemv[(n//2,)](
                x[row],w,expected[row],xs[row],ws,n,k,2,kg,num_warps=4,num_stages=1)
        _gfx1151_block_fp8_gemv_rows[(n//2,rows)](
            x,w,actual,xs,ws,n,k,2,kg,num_warps=4,num_stages=1)
        assert torch.isfinite(actual).all()
        assert torch.equal(expected,actual), (n,k,rows,int((expected!=actual).sum()))
        results.append(dict(n=n,k=k,rows=rows,exact=True,elements=actual.numel()))
torch.cuda.synchronize()
report = dict(status="PASS_NATIVE_EXACT_DECODE_ARITHMETIC",cases=results,
              serving_qualification=False, max_allocated=torch.cuda.max_memory_allocated())
args.out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
