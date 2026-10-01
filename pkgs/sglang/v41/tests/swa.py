#!/usr/bin/env python3
"""Actual ratio0 SWA pool/backend check; --cpu-only never imports Torch/SGLang.

GPU invocation: installed-runtime/bin/sglang-python pkgs/sglang/v41/tests/swa.py RUNTIME
Use a bounded GPU test runner with an exclusive device lease.
"""
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

D, H, PAGE, ROWS = 512, 16, 128, 4096
CASES = 0
MAX_ABS = 0.0


def e4m3(code):
    sign = -1.0 if code & 128 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return float('nan')
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (1 + mantissa / 8) * 2.0**(exponent - 7)


def cpu_only():
    assert e4m3(0) == 0 and math.copysign(1, e4m3(128)) == -1
    assert e4m3(1) == 2**-9 and e4m3(56) == 1 and e4m3(126) == 448
    assert math.isnan(e4m3(127)) and e4m3(254) == -448
    print(json.dumps({'event':'cpu-only-pass','literal_fp8_constants':7,
                      'gpu_tested':False}))


def expected_bytes(values):
    # Literal E4M3FN nearest-even encoding; no production pack/dequant helper.
    blocks = values[:, :448].double().reshape(-1, 7, 64)
    exp = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1.e-8) / 448))
    scaled = (blocks / torch.exp2(exp)[..., None]).clamp(-448,448)
    levels = torch.tensor([e4m3(i) for i in range(127)], dtype=torch.float64)
    high = torch.searchsorted(levels, scaled.abs().contiguous()).clamp(0,126)
    low = (high-1).clamp_min(0)
    dl, dh = scaled.abs()-levels[low], levels[high]-scaled.abs()
    code = torch.where((dh < dl) | ((dh == dl) & ((high & 1) == 0)), high, low)
    code = (code | (torch.signbit(scaled).long() << 7)).byte().flatten(1)
    payload = torch.cat([code,values[:,448:].contiguous().view(torch.uint8)],1)
    return payload, (exp+127).byte()


def read_bytes(pool, layer, slots):
    raw = pool.get_swa_key_buffer_radix(layer).view(torch.uint8)
    p,o = slots//PAGE, slots%PAGE
    payload = raw[p[:,None],o[:,None]*576+torch.arange(576,device='cuda')].cpu()
    scales = raw[p[:,None],PAGE*576+o[:,None]*8+torch.arange(7,device='cuda')].cpu()
    return payload,scales


def literal_decode(payload,scales):
    lut = torch.tensor([e4m3(i) for i in range(256)], dtype=torch.float64)
    nope = lut[payload[:,:448].long()].reshape(-1,7,64)*torch.exp2(scales.double()-127)[...,None]
    rope = payload[:,448:].contiguous().view(torch.bfloat16).double()
    return torch.cat([nope.flatten(1),rope],1)


def make_pool():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    pool = DeepSeekV4TokenToKVPool(max_num_reqs=8,swa_size=ROWS+PAGE,c4_size=0,c128_size=0,
        c4_state_pool_size=0,c128_state_pool_size=0,page_size=256,swa_page_size=PAGE,
        dtype=torch.float8_e4m3fn,c4_state_dtype=torch.float32,c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,qk_rope_head_dim=64,indexer_head_dim=128,layer_num=2,
        device='cuda',enable_memory_saver=False,compression_ratios=[0,0],full_size=ROWS)
    assert pool.kv_layout.value == 'v4'
    assert pool.get_swa_key_buffer_radix(0).data_ptr()!=pool.get_swa_key_buffer_radix(1).data_ptr()
    return pool


def fixture(pool,n,mode):
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import DeepseekV4HipRadixBackend,DSV4AttnMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    layer_id = n%2
    slots = torch.arange(PAGE,ROWS+PAGE,device='cuda',dtype=torch.int64)
    idx = torch.zeros(n,128,device='cuda',dtype=torch.int32)
    lengths = torch.zeros(n,device='cuda',dtype=torch.int32)
    positions = torch.arange(n,device='cuda',dtype=torch.int64)
    raw = torch.arange(n,device='cuda',dtype=torch.int64)+256
    core = DSV4AttnMetadata(page_size=256,page_table=torch.zeros(n,2,device='cuda',dtype=torch.int32),
        raw_out_loc=raw,cuda_int32_kwargs={'device':'cuda','dtype':torch.int32},
        seq_lens_casual=(positions+1).int(),positions_casual=positions.int(),
        swa_page_indices=idx,swa_topk_lengths=lengths,index_topk=512,swa_out_cache_loc=raw.int())
    core.c0_flashmla_metadata = None
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.token_to_kv_pool = pool
    backend.forward_metadata = types.SimpleNamespace(core_attn_metadata=core)
    backend.mtp_enabled = False
    backend.head_dim_v = D
    backend.softmax_scale = D**-.5
    layer = types.SimpleNamespace(layer_id=layer_id,v_head_dim=D)
    batch = types.SimpleNamespace(forward_mode=ForwardMode.DECODE if mode=='decode' else ForwardMode.EXTEND,
        out_cache_loc=raw,positions=positions,req_pool_indices=torch.arange(n,device='cuda'))
    q = torch.empty(n,H,D,device='cuda',dtype=torch.bfloat16)
    live = torch.zeros(n,D,device='cuda',dtype=torch.bfloat16)
    sink = torch.empty(H,device='cuda')
    return types.SimpleNamespace(**locals())


def refresh(f,phase):
    f.values = (torch.randn(ROWS,D)*.5).bfloat16()
    # Explicit scale/rounding cases supplement dense random rows.
    f.values[0,:448] = 0
    f.values[1,:64] = torch.tensor([448.,-448.,1.,-1.,0.,-0.,2**-9,-2**-9]*8).bfloat16()
    f.pool.set_swa_key_buffer_radix_fused(f.layer_id,f.slots,f.values.cuda())
    f.q.copy_((torch.randn(f.n,H,D)*(.3 if phase!=2 else .7)).bfloat16().cuda())
    f.sink.copy_((torch.linspace(-1,1,H)+phase*.2).cuda())
    # Changed logical-to-physical page permutation, preserving in-page offsets.
    mapping = torch.randperm(ROWS//PAGE)
    logical = (torch.arange(f.n)[:,None]*129+torch.arange(128)[None,:]+phase*63)%ROWS
    indices = mapping[logical//PAGE]*PAGE+logical%PAGE+PAGE
    indices[0,:4] = torch.tensor([PAGE,2*PAGE-1,2*PAGE,ROWS+PAGE-1])
    lengths = torch.full((f.n,),128,dtype=torch.int32)
    if f.mode=='prefill':
        lengths = (torch.arange(f.n,dtype=torch.int32)+1).clamp_max(128)
    if phase==1:
        lengths = torch.tensor(([0,1,63,64,65,127,128]*((f.n+6)//7))[:f.n],dtype=torch.int32)
        indices[:,7] = -1
        indices[:,31] = -1
    elif phase==2:
        lengths = torch.tensor(([128,127,65,64,63,1,0]*((f.n+6)//7))[:f.n],dtype=torch.int32)
        indices[:,15] = -1
    # Entries beyond lengths deliberately remain valid nonzero garbage.
    f.idx.copy_(indices.cuda().int())
    f.lengths.copy_(lengths.cuda())
    f.positions.add_(phase+1)


def invoke(f):
    return f.backend._forward_attention(f.q,f.live,f.live,f.layer,f.batch,
        compress_ratio=0,save_kv_cache=False,attn_sink=f.sink)


def verify(f,out,phase):
    global CASES,MAX_ABS
    data,sf = read_bytes(f.pool,f.layer_id,f.slots)
    wanted,wanted_sf = expected_bytes(f.values)
    assert torch.equal(data,wanted), 'native SWA payload mismatch'
    assert torch.equal(sf,wanted_sf), 'native SWA scales mismatch'
    decoded = literal_decode(data,sf)
    indices,lengths = f.idx.cpu().long(),f.lengths.cpu().long()
    valid = (indices>=PAGE)&(torch.arange(128)[None,:]<lengths[:,None])
    keys = decoded[(indices-PAGE).clamp(0,ROWS-1)]
    logits = torch.einsum('nhd,nkd->nhk',f.q.cpu().double(),keys)*D**-.5
    logits.masked_fill_(~valid[:,None,:],float('-inf'))
    sink = f.sink.cpu().double()[None,:,None].expand(f.n,-1,-1)
    weights = torch.softmax(torch.cat([logits,sink],-1),-1)[...,:128]
    expected = torch.einsum('nhk,nkd->nhd',weights,keys)
    actual = out.cpu().double()
    delta = (actual-expected).abs()
    error = float(delta.max())
    rel = float(delta.norm()/expected.norm().clamp_min(1.e-20))
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.012)
    assert rel<.004, rel
    if (lengths==0).any(): assert actual[lengths==0].eq(0).all()
    CASES+=1
    MAX_ABS=max(MAX_ABS,error)
    print(json.dumps({'event':'case','tokens':f.n,'mode':f.mode,'layer':f.layer_id,'phase':phase,
                      'native_cache_bytes':'exact','max_abs':error,'relative_l2':rel}),flush=True)


def main():
    global torch
    os.environ['SGLANG_USE_AITER']='0'
    os.environ['SGLANG_HACK_FLASHMLA_BACKEND']='triton'
    os.environ['SGLANG_DSV4_KV_LAYOUT']='v4'
    os.environ['SGLANG_DSV4_COMPRESSED_KV_LAYOUT']='fp8'
    os.environ['SGLANG_DSV4_UNIFIED_KV_FP8']='0'
    import torch
    from sglang.srt.runtime_context import get_context
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import DeepseekV4HipRadixBackend
    from sglang.kernels.ops.attention.nsa_triton_decode import triton_mla_kernels_decode_fused as kernels
    runtime=Path(sys.argv[1]).resolve()
    kernel_path=Path(inspect.getfile(kernels))
    assert kernel_path.is_relative_to(runtime)
    assert Path(inspect.getfile(DeepseekV4HipRadixBackend)).is_relative_to(runtime)
    assert hasattr(kernels,'_prune_single_scope_configs'), 'pruning patch must be installed by the derivation'
    configs=kernels._prune_single_scope_configs(kernels._fused_gather_attn_dsv4_kernel.configs,{'h_q':16})
    assert len(configs)==3 and all(c.kwargs['BLOCK_H']==16 for c in configs)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1151'
    torch.cuda.set_per_process_memory_fraction(.03)
    torch.set_num_threads(4)
    torch.manual_seed(292917)
    print(json.dumps({'event':'provenance','runtime':str(runtime),'kernel_sha256':hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        'torch':torch.__version__,'hip':torch.version.hip,'scope':'actual ratio0 native SWA pool/backend; constructed metadata, not model scheduler'}),flush=True)
    with torch.inference_mode(),get_context().override_server_args(page_size=256,dsv4_attn_backend='flashmla'):
        pool=make_pool()
        for mode,n in [('decode',1),('decode',3),('decode',8),('prefill',7),('prefill',128),('prefill',256)]:
            f=fixture(pool,n,mode)
            refresh(f,0)
            started=time.monotonic()
            out=invoke(f)
            torch.cuda.synchronize()
            print(json.dumps({'event':'first-invoke','mode':mode,'tokens':n,'wall_s':time.monotonic()-started,
                'best_config':str(kernels._fused_gather_attn_dsv4_kernel.best_config)}),flush=True)
            verify(f,out,0)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): captured=invoke(f)
            for phase in (1,2):
                refresh(f,phase)
                graph.replay()
                torch.cuda.synchronize()
                verify(f,captured,phase)
            del f,out,captured,graph
            gc.collect()
    assert torch.cuda.max_memory_allocated()<512*1024**2
    print(json.dumps({'event':'complete','cases':CASES,'changed_data_graph_replays':12,'max_abs':MAX_ABS,
        'rtol':.02,'atol':.012,'relative_l2_limit':.004,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
        'gpu_peak_reserved':torch.cuda.max_memory_reserved()}),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--cpu-only']:
        cpu_only()
    else:
        main()
