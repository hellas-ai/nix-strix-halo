"""Verify multi-request WMMA keeps the ordinary decode reduction order."""
import argparse,json,os
from pathlib import Path
import torch
from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton as mm
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
os.environ['SGLANG_GLM53_ROW_INVARIANT_FP8_GEMV']='0'
torch.manual_seed(7374);torch.cuda.set_device(0)
def codes(shape):
 t=torch.randint(0,256,shape,device='cuda',dtype=torch.int32)
 return torch.where((t&127)==127,t-1,t).to(torch.uint8).view(torch.float8_e4m3fn)
cases=[]
for n,k in ((6144,4096),(4096,3072),(2048,4096),(4096,1536),(4096,4096),(1024,4096),(4096,512)):
 w=codes((n,k));ws=torch.rand(n//128,k//128,device='cuda')*.01
 for bs,width in ((2,4),(3,2),(4,2),(4,4)):
  x=codes((bs*width,k));xs=torch.rand(bs*width,k//128,device='cuda')*.01
  expected=[]
  for pos in range(width):
   rows=torch.arange(bs,device='cuda')*width+pos
   expected.append(mm(x[rows],w,xs[rows],ws,[128,128],torch.bfloat16))
  expected=torch.stack(expected,dim=1).reshape(bs*width,n)
  actual=mm(x,w,xs,ws,[128,128],torch.bfloat16)
  assert torch.equal(expected,actual),(n,k,bs,width,int((expected!=actual).sum()))
  cases.append(dict(n=n,k=k,bs=bs,width=width,exact=True))
torch.cuda.synchronize()
r=dict(status='PASS_NATIVE_MULTI_REQUEST_DECODE_MATH',cases=cases,full_model_qualified=False)
a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
