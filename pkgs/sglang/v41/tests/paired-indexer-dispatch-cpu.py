#!/usr/bin/env python3
"""Fail-closed CPU test for the actual V4.1 C1/C2 serving seam."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace as NS


ROOT = Path(sys.argv[1]) / "lib/python3.13/site-packages/sglang"
PATH = ROOT / "srt/layers/attention/dsv4/low_ratio_backend_hip.py"
tree = ast.parse(PATH.read_text())
fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
          and node.name == "_single_request_low_ratio_full_chunk_pair_eligible")
forbidden = {"item", "tolist", "cpu", "numpy"}
assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden
               for node in ast.walk(fn)), "guard must not read GPU values"
module = ast.Module(body=[ast.ImportFrom(module="__future__",
                                         names=[ast.alias(name="annotations")],
                                         level=0), fn], type_ignores=[])
scope = {}
exec(compile(ast.fix_missing_locations(module), str(PATH), "exec"), scope)
guard = scope[fn.name]


def tensor(shape=None, n=None):
    return NS(shape=shape, numel=lambda: n)


def valid(ratio=1):
    batch = NS(forward_mode=NS(name="EXTEND"), batch_size=1,
               req_pool_indices=tensor(shape=(1,)),
               _original_forward_mode=None, tbo_parent_token_range=None,
               extend_seq_lens_cpu=[1536], extend_num_tokens=1536,
               input_ids=tensor(n=1536))
    meta = NS(late_layer_tail=None)
    idx = NS(compress_ratio=ratio, page_table=tensor(shape=(1536, 832)),
             compressed_seq_lens=tensor(shape=(1536,)),
             use_prefill_cuda_graph=False)
    return batch, meta, idx


def run(batch=None, meta=None, idx=None, *, ratio=1, rows=slice(0, 1536),
        query_rows=1536, q_rows=1536, seq_lens_cpu=None,
        extend_lens_cpu=None):
    defaults = valid(ratio)
    batch, meta, idx = (batch or defaults[0], meta or defaults[1], idx or defaults[2])
    return guard(batch, meta, idx, ratio, query_rows, q_rows, rows,
                 [50688] if seq_lens_cpu is None else seq_lens_cpu,
                 [1536] if extend_lens_cpu is None else extend_lens_cpu)


assert run(ratio=1) and run(ratio=2)
for key, value in (
    ("forward_mode", NS(name="MIXED")), ("forward_mode", NS(name="TARGET_VERIFY")),
    ("forward_mode", NS(name="DECODE")), ("batch_size", 2),
    ("req_pool_indices", tensor(shape=(2,))),
    ("_original_forward_mode", NS(name="EXTEND")),
    ("tbo_parent_token_range", (0, 1536)),
    ("extend_seq_lens_cpu", [1535]), ("extend_seq_lens_cpu", [768, 768]),
    ("extend_num_tokens", 1535), ("input_ids", tensor(n=1535)),
):
    batch, meta, idx = valid()
    setattr(batch, key, value)
    assert not run(batch, meta, idx), key
for key, value in (("late_layer_tail", NS(extend_seq_lens_cpu=[1536])),):
    batch, meta, idx = valid()
    setattr(meta, key, value)
    assert not run(batch, meta, idx), key
for key, value in (
    ("compress_ratio", 4), ("page_table", tensor(shape=(1, 832))),
    ("compressed_seq_lens", tensor(shape=(1,))),
    ("use_prefill_cuda_graph", True),
):
    batch, meta, idx = valid()
    setattr(idx, key, value)
    assert not run(batch, meta, idx), key
for change in (
    {"ratio": 4}, {"query_rows": 330, "rows": slice(0, 330)},
    {"query_rows": 1535}, {"q_rows": 1535}, {"rows": slice(1, 1536)},
    {"rows": slice(0, 1535)}, {"rows": slice(0, 1536, 2)},
    {"seq_lens_cpu": [1535, 50688]}, {"extend_lens_cpu": [768, 768]},
):
    assert not run(**change), change

outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "low_ratio_index_topk_hip_extend")
score = next(node for node in ast.walk(outer) if isinstance(node, ast.FunctionDef)
             and node.name == "score_rows")
calls = [node for node in ast.walk(score) if isinstance(node, ast.Call)
         and isinstance(node.func, ast.Name)
         and node.func.id == "aiter_fp4_paged_mqa_logits"]
assert len(calls) == 1
assert any(kw.arg == "paired_single_request" and isinstance(kw.value, ast.Name)
           and kw.value.id == "paired_single_request" for kw in calls[0].keywords)

dispatch_path = ROOT / "kernels/ops/attention/dsv4/fp4_indexer_gfx1151.py"
dispatch_tree = ast.parse(dispatch_path.read_text())
dispatch_fn = next(node for node in dispatch_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "paged_fp4_logits")
assign = next(node for node in ast.walk(dispatch_fn) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "paired_prefill"
                      for target in node.targets))
dispatch_expr = compile(ast.Expression(assign.value), str(dispatch_path), "eval")


def dispatch(*, rows=1536, heads=32, paired=True, width=28672,
             dtype="float32", contiguous=True, device="cuda", gfx=True,
             enabled=True, parallel=True):
    out = NS(shape=(rows, width), dtype=dtype, is_contiguous=lambda: contiguous)
    q = NS(device=NS(type=device))
    env = {"SGLANG_DSV41_PAIRED_INDEXER_K": "1" if enabled else "0",
           "SGLANG_DSV41_PARALLEL_INDEXER_SCORE": "1" if parallel else "0"}
    return bool(eval(dispatch_expr, {
        "os": NS(environ=env), "torch": NS(float32="float32"),
        "q": q, "out": out, "rows": rows, "heads": heads,
        "paired_single_request": paired, "use_gfx1151_indexer": lambda _: gfx,
    }))


for width in (12288, 20480, 28672, 53248, 32768, 65536):
    assert dispatch(width=width), width
for change in (
    {"rows": 330}, {"rows": 4}, {"heads": 16}, {"paired": False},
    {"width": 8192}, {"width": 12800}, {"width": 69632},
    {"dtype": "float16"}, {"contiguous": False}, {"device": "cpu"},
    {"gfx": False}, {"enabled": False}, {"parallel": False},
):
    assert not dispatch(**change), change
print("PASS: low-ratio C1/C2 full-chunk guard and explicit bool propagation")
