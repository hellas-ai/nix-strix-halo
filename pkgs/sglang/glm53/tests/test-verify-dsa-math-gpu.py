"""Decode-equivalent split-K geometry for multiple verification rows."""
import argparse,json
from pathlib import Path
import torch
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import triton_sparse_mla_decode_splitk as attention
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
torch.manual_seed(7371);torch.cuda.set_device(0)
cases=[]
for tail in (0,64):
    for bs in (1,2,4):
        for width in (2,4):
            for topk in (1024,2048):
                q=torch.randn(bs*width,16,512,device='cuda',dtype=torch.bfloat16)
                r=torch.randn(bs*width,16,tail,device='cuda',dtype=torch.bfloat16)
                kv=torch.randn(4096,1,512+tail,device='cuda',dtype=torch.bfloat16)
                indices=torch.randint(4096,(bs*width,1,topk),device='cuda',dtype=torch.int32)
                # Exercise different causal lengths and invalid tail slots.
                indices[:,:,-7:]=-1
                reference=[]
                for pos in range(width):
                    rows=torch.arange(bs,device='cuda')*width+pos
                    reference.append(attention(q[rows],r[rows],kv,indices[rows],.125).clone())
                expected=torch.stack(reference,dim=2).reshape(1,bs*width,16,512)
                actual=attention(q,r,kv,indices,.125,decode_batch_size=bs)
                assert torch.equal(actual,expected),(bs,width,topk,float((actual-expected).abs().max()))
                cases.append(dict(bs=bs,width=width,topk=topk,rope_dim=tail,exact=True))
torch.cuda.synchronize()
result=dict(status='PASS_NATIVE_DECODE_EQUIVALENT_DSA',cases=cases,full_model_qualified=False)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
