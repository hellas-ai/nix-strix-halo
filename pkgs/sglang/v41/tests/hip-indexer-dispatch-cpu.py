#!/usr/bin/env python3
"""CPU-only check of the single-request eligibility proof used by the HIP block indexer (0087) and that the prebuilt library exists.

  hip-indexer-dispatch-cpu.py RUNTIME_ROOT
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(sys.argv[1]) / "lib/python3.13/site-packages/sglang"
blob = (root / "kernels/ops/attention/dsv4/libdsv41_indexer_block.so").read_bytes()
assert blob[:4] == b"\x7fELF" and b"dsv41_fp4_indexer" in blob, "prebuilt library missing or lacks dsv41_fp4_indexer"

path = root / "srt/layers/attention/dsv4/indexer.py"
tree = ast.parse(path.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_single_request_extend_rows_eligible")
assert not any(isinstance(n, ast.Attribute) and n.attr in {"item", "tolist", "cpu", "numpy"} for n in ast.walk(fn)), "guard must not read GPU values"
EXTEND = object()
scope = {"ForwardMode": NS(EXTEND=EXTEND)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])), str(path), "exec"), scope)
guard = scope[fn.name]


def batch(**kw):
    b = NS(forward_mode=EXTEND, batch_size=1, req_pool_indices=NS(shape=(1,)), _original_forward_mode=None, tbo_parent_token_range=None)
    for k, v in kw.items():
        setattr(b, k, v)
    return b


meta = NS(use_prefill_cuda_graph=False)
count = 0
assert guard(batch(), meta); count += 1
for key, value in (("forward_mode", object()), ("batch_size", 2), ("req_pool_indices", NS(shape=(2,))), ("_original_forward_mode", EXTEND), ("tbo_parent_token_range", (0, 8))):
    assert not guard(batch(**{key: value}), meta), key
    count += 1
assert not guard(batch(), NS(use_prefill_cuda_graph=True)); count += 1
print(f"PASS {count} CPU HIP indexer eligibility cases; library present")
