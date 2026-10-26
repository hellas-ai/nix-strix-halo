#!/usr/bin/env python3
"""Exercise installed-style hybrid dispatch bounds without a GPU/context."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REL = 'sglang/kernels/ops/attention/nsa_triton_decode/'
source = Path(sys.argv[1]) / 'lib/python3.13/site-packages' / REL / 'triton_mla_kernels_decode_fused.py'
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_use_checkpoint_attention')
device = object()


class Tensor:
    def __init__(self, shape, dtype, *, contiguous=True, dev=device, count=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = dev
        self.contiguous = contiguous
        self.count = count

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        if self.count is not None:
            return self.count
        n = 1
        for value in self.shape:
            n *= value
        return n


ns = {'torch': SimpleNamespace(bfloat16='bf16', uint8='u8', int32='i32', float32='f32'),
      '_checkpoint_attention_enabled': True, 'BUFFER_OPS_DISABLE_THRESHOLD': 2**31,
      '_is_gfx1151_attention_device': lambda d: d is device}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), 'exec'), ns)
guard = ns['_use_checkpoint_attention']


def operands(rows, split):
    return [Tensor((rows,16,512),'bf16'), Tensor((192,256,1,528),'u8'),
            Tensor((384,128,1,528),'u8'), Tensor((rows,128),'i32'),
            Tensor((rows,512),'i32'), 256,128,1,split,
            Tensor((rows,),'i32'),Tensor((rows,),'i32'),Tensor((16,),'f32')]


count = 0
for rows, split in [(64,0),(200,0),(1536,0),(1,4),(2,4),(4,4),(8,4),(16,4)]:
    args = operands(rows,split)
    assert guard(*args)
    count += 1
    args[9:12] = [None,None,None]
    assert guard(*args)
    count += 1
for rows, split in [(0,0),(32,0),(1537,0),(3,4),(32,4),(64,4),(4,0)]:
    assert not guard(*operands(rows,split))
    count += 1
for index, value in [(6,64),(7,2),(9,None),(11,Tensor((16,),'bf16')),
                     (3,Tensor((64,128),'i64')),
                     (2,Tensor((384,128,1,528),'u8',count=2**31+1)),
                     (0,Tensor((64,16,512),'bf16',contiguous=False)),
                     (0,Tensor((64,16,512),'bf16',dev=object()))]:
    args = operands(64,0)
    args[index] = value
    assert not guard(*args), (index,value)
    count += 1
for rows, split in [(64,0),(1536,0),(1,4),(4,4),(8,4),(16,4)]:
    args = operands(rows,split)
    args[2] = Tensor((192,256,1,528),'u8')
    args[6] = 256
    assert guard(*args)
    count += 1
ns['_checkpoint_attention_enabled'] = False
assert not guard(*([None]*12)), 'default-off must not touch device/tensor metadata'

print(f'PASS {count + 1} CPU checkpoint attention dispatch cases; no torch or GPU import')
