#!/usr/bin/env python3
"""Compile the seven fixed GLM WMMA decode shapes without a GPU.

Produces Triton's ordinary cache in TRITON_CACHE_DIR. An optional observed
cache confirms keys and reports binary equality; it does not invoke kernels.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend
from triton.runtime.cache import get_cache_key
from triton.runtime.jit import create_function_from_signature
from triton._C.libtriton import get_cache_invalidating_env_vars
from sglang.kernels.ops.gemm import glm53_wmma_gemv as wmma

p = argparse.ArgumentParser()
p.add_argument('--observed-cache', type=Path)
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
if a.report.exists():
    raise SystemExit('report exists')
cache = Path(os.environ['TRITON_CACHE_DIR']).absolute()
cache.mkdir(parents=True, exist_ok=True)
target = GPUTarget('hip', 'gfx1151', 32)
backend = make_backend(target)
env = get_cache_invalidating_env_vars()
if env:
    raise SystemExit(f'non-default Triton invalidating environment: {env}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(fn, args, launch):
    # The binder reproduces Triton's pointer specialization without GPU tensors.
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, raw_options = binder(*args, **launch)
    options, signature, constexprs, attrs = fn._pack_args(backend, raw_options, bound, specialization, raw_options)
    src = ASTSource(fn, signature, constexprs, attrs)
    key = get_cache_key(src, backend, options, env)
    name = base64.b32encode(hashlib.sha256(key.encode()).digest()).decode().rstrip('=')
    if a.observed_cache and not (a.observed_cache / name / (fn.__name__ + '.hsaco')).is_file():
        raise SystemExit(f'key missing from observed cache: {fn.__name__} {name}')
    triton.compile(src, target=target, options=launch)
    binary = cache / name / (fn.__name__ + '.hsaco')
    if not binary.is_file():
        raise SystemExit(f'compiler did not create expected cache: {name}')
    observed = a.observed_cache / name / binary.name if a.observed_cache else None
    return {'kernel': fn.__name__, 'key': name, 'hsaco_sha256': sha(binary),
            'observed_sha256': sha(observed) if observed else None,
            'observed_binary_equal': sha(binary) == sha(observed) if observed else None}


# CPU allocations provide the exact alignment and pointer-range attributes
# represented by the validated GPU case. Reject if allocator violates them.
u8 = torch.zeros(8, dtype=torch.uint8)
f32 = torch.zeros(8, dtype=torch.float32)
bf16 = torch.zeros(8, dtype=torch.bfloat16)
for tensor in (u8, f32, bf16):
    if tensor.data_ptr() % 16 or tensor.untyped_storage().size() > 2**31 - 1:
        raise SystemExit('CPU specialization attributes do not match expected GPU inputs')
rows = []
for n, k in ((6144,4096),(4096,3072),(2048,4096),(4096,1536),
             (4096,4096),(1024,4096),(4096,512)):
    splits = max(1, k // 1024)
    rows.append(build(wmma._wmma_fp8_gemv_partial,
                      [u8,f32,u8,f32,f32,8,n,k,k//splits,16,64,4],
                      {'num_warps':4,'num_stages':2}))
    rows[-1].update(n=n,k=k)
    rows.append(build(wmma._sum_partials, [f32,bf16,8,n,splits,16,1024],
                      {'num_warps':4}))
    rows[-1].update(n=n,k=k)
report = {'target': vars(target), 'compiler_environment': env,
          'module': str(Path(wmma.__file__).resolve()), 'module_sha256': sha(Path(wmma.__file__)),
          'observed_cache': str(a.observed_cache) if a.observed_cache else None,
          'rows': rows, 'gpu_execution_tested': False}
a.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
