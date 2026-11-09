#!/usr/bin/env python3
"""Exercise installed 0016 dispatch bodies with CPU tensor/kernel stubs.

No Torch, Triton or native module is imported. These checks qualify Python
selection and argument forwarding, not GPU arithmetic or model numerics.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace


def qualify(source):
    """Actual installed Python dispatch bodies; tensor/native calls are CPU stubs."""
    source = Path(source)
    trees = {name: ast.parse((source / name).read_text()) for name in (
        'kernels/ops/moe/glm53_moe_wmma.py',
        'kernels/ops/moe/fused_moe_triton_kernels.py',
        'srt/layers/moe/moe_runner/triton_utils/fused_moe.py',
    )}
    environment = {}
    namespace = {'os': SimpleNamespace(environ=environment)}
    pure = []
    for node in trees['kernels/ops/moe/glm53_moe_wmma.py'].body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ('NUM_EXPERTS', 'SHAPES', 'AUTO_BLOCK_M', 'AUTO_TILES') for t in node.targets):
            pure.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ('enabled', 'applies', 'block_m', '_tiles'):
            pure.append(node)
    exec(compile(ast.Module(pure, type_ignores=[]), '<installed0016-pure-methods>', 'exec'), namespace)
    calls = []
    module = SimpleNamespace(**{k: namespace[k] for k in ('enabled', 'applies', 'block_m', '_tiles')},
        quant_widen=lambda a: ('quantized16', 'scales32'), widen=lambda a: 'widened16',
        moe_gemm=lambda *a, **k: calls.append((a, k)))
    class Tensor:
        def __init__(self, shape, dtype='bf16'):
            self.shape, self.dtype, self.ndim = tuple(shape), dtype, len(shape)
        def stride(self, dim): return 1
        def numel(self):
            value = 1
            for n in self.shape: value *= n
            return value
        def contiguous(self): return self
    def method(tree, name, scope, stop=None):
        node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []; node.returns = None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs): arg.annotation = None
        if stop is not None:
            boundary = next(i for i, n in enumerate(node.body) if stop(n))
            node.body = node.body[:boundary + 1] + [ast.Return(ast.Constant('GENERIC_FALLBACK'))]
        exec(compile(ast.fix_missing_locations(ast.Module([node], type_ignores=[])), name, 'exec'), scope)
        return scope[name]
    scope = dict(_glm53_moe_wmma=module, _gfx1151_fp8_widen=SimpleNamespace(value=True),
        _GLM53_MOE_GEMV_MAX_ROWS=16, torch=SimpleNamespace(bfloat16='bf16', float8_e4m3fn='fp8', float32='fp32'),
        tl=SimpleNamespace(bfloat16='bf16'))
    lower = method(trees['kernels/ops/moe/fused_moe_triton_kernels.py'],
        'invoke_fused_moe_kernel', scope,
        lambda n: isinstance(n, ast.If) and '_glm53_moe_wmma.moe_gemm' in ast.unparse(n))
    align_calls = []
    scope.update(_is_hip=True, _use_aiter=False, padding_size=0,
        get_config_dtype_str=lambda **k: 'fp8',
        try_get_optimal_moe_config=lambda *a, **k: ({'BLOCK_SIZE_M': 64}, ({'BLOCK_SIZE_M': 64}, None)),
        _moe_support_tma=lambda: False,
        moe_align_block_size=lambda ids, bm, experts: align_calls.append((bm, experts)) or ('sorted', 'experts', 'padded'))
    upper = method(trees['srt/layers/moe/moe_runner/triton_utils/fused_moe.py'],
                   '_prepare_fused_moe_run', scope)
    def alignment():
        return upper(Tensor([1024,4096]), Tensor([288,1024,4096]), Tensor([288,4096,512]), Tensor([1024,8]),
            use_fp8_w8a8=True, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=[128,128])
    def invoke(n, k, mul, **changes):
        arguments = dict(A=Tensor([1024 if not mul else 8192,k]), B=Tensor([288,n,k],'fp8'), bias=None,
            C=Tensor([8192,n]), A_scale=None, B_scale=Tensor([288,n//128,k//128],'fp32'), B_zp=None,
            topk_weights=Tensor([1024,8]), topk_ids=Tensor([1024,8]), sorted_token_ids=Tensor([8192],'int32'),
            expert_ids=Tensor([288],'int32'), num_tokens_post_padded=Tensor([1],'int32'),
            mul_routed_weight=mul, top_k=1 if mul else 8, config={'BLOCK_SIZE_M':32,'GLM53_MOE_WMMA':True},
            compute_type='bf16', use_fp8_w8a8=True, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False, block_shape=[128,128])
        arguments.update(changes)
        return lower(**arguments)
    assert module.enabled() is False and not module.applies([288,1024,4096],[128,128])
    assert alignment()[0] == {'BLOCK_SIZE_M':64} and align_calls[-1] == (64,288)
    assert invoke(1024,4096,False) == 'GENERIC_FALLBACK' and calls == []
    environment['SGLANG_GLM53_MOE_WMMA'] = '0'
    assert not module.enabled()
    assert alignment()[0] == {'BLOCK_SIZE_M':64} and align_calls[-1] == (64,288)
    assert invoke(1024,4096,False) == 'GENERIC_FALLBACK' and calls == []
    environment['SGLANG_GLM53_MOE_WMMA'] = '1'
    assert alignment()[0] == {'BLOCK_SIZE_M':32,'GLM53_MOE_WMMA':True} and align_calls[-1] == (32,288)
    for n, k, mul in ((1024,4096,False),(4096,512,True)):
        assert invoke(n,k,mul) is None
        args, kwargs = calls[-1]
        assert args[0:2] == ('quantized16','scales32') and args[9:13] == (mul,1 if mul else 8,8192,32)
        assert kwargs == {'num_tokens':1024}
    assert invoke(1024,4096,False,A=Tensor([1024,4096],'fp8'),A_scale='existing_scale') is None
    assert calls[-1][0][0:2] == ('widened16','existing_scale')
    count = len(calls)
    for change in ({'topk_ids':Tensor([16,8])},{'B':Tensor([287,1024,4096],'fp8')},
                   {'B':Tensor([288,1024,4096],'bf16')},{'block_shape':[64,128]},
                   {'B_scale':None},{'B_scale':Tensor([288,8,32],'bf16')},{'bias':object()},
                   {'config':{'BLOCK_SIZE_M':64}},{'a_use_tma':True},{'c_sorted':True}):
        assert invoke(1024,4096,False,**change) == 'GENERIC_FALLBACK' and len(calls) == count
    environment['SGLANG_GLM53_MOE_WMMA'] = 'true'
    assert not module.enabled()
    return dict(status='PASS_ACTUAL_SOURCE_DISPATCH_CPU_STUBS_ONLY', default_disabled=True,
        both_projections=True, prequantized_API=True, negative_dispatch_controls=10,
        no_Torch_Triton_or_native_module_import=True, native_or_numerical_qualification=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.source), sort_keys=True))
