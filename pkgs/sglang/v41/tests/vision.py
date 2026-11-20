#!/usr/bin/env python3
"""Compare actual BF16 V4.1 ViT+aligner weights against pinned official forward."""

import gc
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['SGLANG_USE_AITER'] = '0'

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.models.deepseek_v41_vit import Aligner as RuntimeAligner
from sglang.srt.models.deepseek_v41_vit import ViT as RuntimeViT
from sglang.srt.multimodal.deepseek_v41_image_processing import patchify_image
from sglang.test.test_utils import publish_build_topology


root = Path(sys.argv[2] if len(sys.argv) > 2 else '/mnt/ds41-model')
expected_bytes = 970_506_240
expected_tensors = 263


def official_module():
    path = root / 'inference/vision.py'
    spec = importlib.util.spec_from_file_location('official_dsv41_vision_full', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def images(args):
    yy, xx = np.indices((73, 109))
    a = np.stack([(xx*17+yy*3)%256, (yy*19)%256,
                  (xx^yy)%256, np.full_like(xx,255)], -1).astype(np.uint8)
    first = Image.fromarray(a, 'RGBA')
    b = np.stack([(xx*3)%256, (yy*11)%256, ((xx+yy)*7)%256], -1).astype(np.uint8)
    second = Image.fromarray(b, 'RGB').resize((41, 157))
    changed = first.copy()
    changed.paste((0,255,0,255), (0,0,40,40))
    result = []
    for name, image in [('rgba_landscape',first), ('rgb_portrait',second),
                        ('rgba_changed',changed)]:
        patches,h,w,lh,lw = patchify_image(image,args)
        assert (h,w) in ((32,48),(76,20))
        result.append((name, patches, h, w))
    assert not torch.equal(result[0][1],result[2][1])
    return result


def make_tower(args, official):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        holder = torch.nn.Module()
        holder.vision = official.ViT(args) if official else RuntimeViT(args)
        holder.aligner = official.Aligner(args) if official else RuntimeAligner(args)
    finally:
        torch.set_default_dtype(old_dtype)
    return holder.to('cuda').eval()


def load_checkpoint(holder, official):
    path = root / 'model-00001-of-00048.safetensors'
    params = dict(holder.named_parameters())
    loaded, bytes_seen = set(), 0
    with safe_open(path, framework='pt', device='cpu') as file, torch.no_grad():
        for original in file.keys():
            if not original.startswith(('vision.','aligner.')):
                continue
            key = original if official else original.replace('.attn.wqkv.', '.attn.qkv_proj.').replace('.attn.wo.', '.attn.proj.')
            assert key in params, (original,key)
            tensor = file.get_tensor(original)
            assert tensor.dtype == torch.bfloat16 and params[key].shape == tensor.shape
            params[key].copy_(tensor, non_blocking=False)
            loaded.add(key)
            bytes_seen += tensor.numel() * tensor.element_size()
    assert len(loaded) == expected_tensors and len(params) == expected_tensors, (len(loaded),len(params))
    assert bytes_seen == expected_bytes, bytes_seen
    print(json.dumps({'event':'weights','official':official,'tensors':len(loaded),
                      'bytes':bytes_seen}),flush=True)


def run_tower(holder, case, use_official):
    name, patches, h, w = case
    saved = []
    handles = [block.register_forward_hook(lambda _m,_x,y: saved.append(y.detach().cpu()))
               for block in holder.vision.blocks]
    try:
        with torch.inference_mode():
            if use_official:
                # The pinned reference makes cos/sin on the default device;
                # its inference program sets CUDA as default around the tower.
                torch.set_default_device('cuda')
            try:
                vit = holder.vision(patches.to('cuda'), h, w)
                aligned = holder.aligner(vit, h, w)
            finally:
                if use_official:
                    torch.set_default_device('cpu')
        assert len(saved) == 32
        return {'name':name,'blocks':saved,'vit':vit.detach().cpu(),
                'aligned':aligned.detach().cpu()}
    finally:
        for handle in handles:
            handle.remove()


def relative(a,b):
    a,b = a.float(),b.float()
    return float((a-b).norm()/b.norm().clamp(min=1))


def main():
    import inspect
    import sglang
    runtime = Path(sys.argv[1]).resolve()
    assert Path(sglang.__file__).resolve().is_relative_to(runtime)
    assert Path(inspect.getfile(RuntimeViT)).resolve().is_relative_to(runtime)
    assert root.joinpath('config.json').exists()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.03)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1151'
    args = SimpleNamespace(**json.loads((root/'inference/config.json').read_text()))
    cases = images(args)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(port)
    init_distributed_environment(world_size=1,rank=0,local_rank=0,
                                 distributed_init_method=f'tcp://127.0.0.1:{port}',
                                 backend='gloo')
    publish_build_topology(tp_size=1,mm_attention_backend='triton_attn')
    initialize_model_parallel(backend='gloo')
    try:
        official = official_module()
        ref_tower = make_tower(args,official)
        load_checkpoint(ref_tower,True)
        references = [run_tower(ref_tower,case,True) for case in cases]
        repeated=run_tower(ref_tower,cases[0],True)
        print(json.dumps({'event':'reference_repeat','name':cases[0][0],
                          'block_equal':all(torch.equal(a,b) for a,b in zip(repeated['blocks'],references[0]['blocks'])),
                          'vit_equal':torch.equal(repeated['vit'],references[0]['vit']),
                          'aligner_equal':torch.equal(repeated['aligned'],references[0]['aligned'])}),flush=True)
        del ref_tower;gc.collect();torch.cuda.empty_cache()
        tower = make_tower(args,False)
        load_checkpoint(tower,False)
        assert all(type(block.attn.qkv_backend).__name__=='ReferenceSingleImageSdpa'
                   for block in tower.vision.blocks)
        maximum = 0.
        for case,reference in zip(cases,references):
            actual=run_tower(tower,case,False)
            block_errors=[relative(a,b) for a,b in zip(actual['blocks'],reference['blocks'])]
            vit_error=relative(actual['vit'],reference['vit'])
            align_error=relative(actual['aligned'],reference['aligned'])
            block_equal=all(torch.equal(a,b) for a,b in zip(actual['blocks'],reference['blocks']))
            vit_equal=torch.equal(actual['vit'],reference['vit'])
            align_equal=torch.equal(actual['aligned'],reference['aligned'])
            maximum=max(maximum,vit_error,align_error,*block_errors)
            print(json.dumps({'event':'case','name':case[0],
                              'grid':[case[2],case[3]],
                              'blocks_equal':block_equal,
                              'vit_equal':vit_equal,
                              'aligner_equal':align_equal,
                              'block_relative_l2':block_errors,
                              'max_block_relative_l2':max(block_errors),
                              'last_block_relative_l2':block_errors[-1],
                              'vit_relative_l2':vit_error,
                              'aligner_relative_l2':align_error}),flush=True)
            assert block_equal and vit_equal and align_equal
            del actual
        print(json.dumps({'event':'complete','cases':len(cases),
                          'max_relative_l2':maximum}),flush=True)
        assert maximum < .03, maximum
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__=='__main__':
    main()
