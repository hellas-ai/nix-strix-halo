"""Actual-source tests for 0051; CPU only, no real GPU/service use."""
import ast
import os
import sys
from enum import Enum, IntEnum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

import sglang
import torch
from sglang.srt.layers.attention import graph_variants as variants

root = Path(sglang.__file__).parent
torch.set_num_threads(2)
torch.manual_seed(53)
assert not torch.cuda.is_initialized()


def mode(decode=True):
    return SimpleNamespace(is_decode_or_idle=lambda: decode,
                           is_target_verify=lambda: not decode,
                           is_draft_extend_v2=lambda: False)


def batch(lengths, decode=True):
    return SimpleNamespace(seq_lens_cpu=lengths, forward_mode=mode(decode),
                           seq_lens=SimpleNamespace(max=lambda: (_ for _ in ()).throw(AssertionError("D2H forbidden"))))


policy = variants.GlmKpoolGraphVariants(2048)
for lengths, expected in (([2048], "dense"), ([2049], "sparse16k"),
                          ([16384], "sparse16k"), ([16385], "sparse"),
                          ([7000, 16385, 3], "sparse"), ([7000, 15000, 1, 2], "sparse16k")):
    assert policy.select(batch(torch.tensor(lengths))) == expected
assert policy.select(batch(None)) == "sparse"
assert policy.select(batch(torch.empty(0, dtype=torch.int64))) == "sparse"
assert policy.select(batch(torch.tensor([5000]), decode=False)) == "sparse"
assert policy.select(batch(SimpleNamespace(device=SimpleNamespace(type="cuda")))) == "sparse"

# Exercise actual factory default OFF and exact model gate with mocked platform/config predicates.
model = SimpleNamespace(model_type="glm5_next_text", index_kpool=4)
modules = {
    "sglang.srt.configs.model_config": SimpleNamespace(get_dsa_index_topk=lambda c: 2048,
                                                       is_deepseek_dsa=lambda c: True),
    "sglang.srt.utils": SimpleNamespace(is_hip=lambda: True),
}
with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
    os.environ.pop("SGLANG_GLM53_INDEXER_16K_GRAPH", None)
    assert type(variants.create_attention_graph_variants(model)) is variants.DsaGraphVariants
    os.environ["SGLANG_GLM53_INDEXER_16K_GRAPH"] = "1"
    # Fail closed unless the calling runner explicitly permits plain capture.
    assert type(variants.create_attention_graph_variants(model)) is variants.DsaGraphVariants
    assert type(variants.create_attention_graph_variants(model, allow_glm_short=True)) is variants.GlmKpoolGraphVariants
    assert type(variants.create_attention_graph_variants(SimpleNamespace(text_config=model), allow_glm_short=True)) is variants.GlmKpoolGraphVariants
    assert type(variants.create_attention_graph_variants(SimpleNamespace(model_type="deepseek_v4", index_kpool=4), allow_glm_short=True)) is variants.DsaGraphVariants


# Execute the actual constructor capture-mode and factory-call statements with
# the actual source enums; importing the full runner would initialize GPU modules.
enum_ns = {"Enum": Enum, "IntEnum": IntEnum, "auto": auto, "CustomSpecAlgo": object}
for path, name in (("srt/speculative/spec_info.py", "SpeculativeAlgorithm"),
                   ("srt/model_executor/forward_batch_info.py", "ForwardMode")):
    tree = ast.parse((root / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module([future, node], [])), path, "exec"), enum_ns)
runner_path = "srt/model_executor/runner/decode_cuda_graph_runner.py"
runner_tree = ast.parse((root / runner_path).read_text())
runner_class = next(n for n in runner_tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeCudaGraphRunner")
init = next(n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
def self_assignment(node, name):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr == name for t in targets)
first = next(i for i, n in enumerate(init.body) if self_assignment(n, "capture_forward_mode"))
last = next(i for i, n in enumerate(init.body) if self_assignment(n, "attention_graph_variants"))
actual_capture_setup = compile(ast.Module(init.body[first:last + 1], []), runner_path, "exec")
algorithm = enum_ns["SpeculativeAlgorithm"]
cases = [(algorithm.NONE, False, False), (algorithm.NONE, True, False),
         (algorithm.NONE, False, True)]
cases += [(a, False, False) for a in algorithm if a != algorithm.NONE]
cases += [(algorithm.DFLASH, True, False), (algorithm.DSPARK, True, False)]
with patch.dict(sys.modules, modules), patch.dict(os.environ, {"SGLANG_GLM53_INDEXER_16K_GRAPH": "1"}):
    for spec_algorithm, draft, dllm in cases:
        self = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=draft),
                               return_hidden_states_mode=None, is_dllm=dllm,
                               speculative_num_draft_tokens=4)
        model_runner = SimpleNamespace(spec_algorithm=spec_algorithm, is_draft_worker=draft,
            model_config=SimpleNamespace(hf_config=model), decode_num_tokens_per_req=lambda **kw: 4)
        self.model_runner = model_runner
        env = dict(enum_ns, self=self, model_runner=model_runner, Optional=Optional,
                   AttentionGraphVariants=variants.AttentionGraphVariants,
                   create_attention_graph_variants=variants.create_attention_graph_variants)
        exec(actual_capture_setup, env)
        labels = self.attention_graph_variants.capture_labels
        assert ("sparse16k" in labels) == (spec_algorithm == algorithm.NONE and not draft and not dllm), (spec_algorithm, draft, dllm, labels)


def source_functions(path, names, namespace):
    tree = ast.parse((root / path).read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for n in nodes:
        n.decorator_list = []
    exec(compile(ast.Module(nodes, []), path, "exec"), namespace)
    return namespace


ns = source_functions("srt/layers/attention/dsa/kpool_fp8_index.py",
                      {"_topk_from_pooled_history_logits_hip", "build_pooled_page_table_64"},
                      {"torch": torch, "BLOCK_SIZE_K": 64})
topk = ns["_topk_from_pooled_history_logits_hip"]
# Importing runner_utils/__init__ initializes GPU-oriented MoE modules. Use
# the actual capture setter/getter source in a lightweight module instead.
capture_path = "sglang.srt.model_executor.runner_utils.capture_mode"
capture = ModuleType(capture_path)
capture.Optional = Optional
capture._capture_attention_variant = None
source_functions("srt/model_executor/runner_utils/capture_mode.py",
                 {"_set_capture_attention_variant", "get_capture_attention_variant"}, capture.__dict__)
sys.modules[capture_path] = capture
_set_capture_attention_variant = capture._set_capture_attention_variant
ns = source_functions("srt/layers/attention/dsa_backend.py", {"_transform_table_1_to_real"}, ns)
raw_table = torch.arange(131076, dtype=torch.int32)[None, :]
real = ns["_transform_table_1_to_real"](SimpleNamespace(real_page_size=64), raw_table)
pooled = ns["build_pooled_page_table_64"](real, 4)
assert real.shape == (1, 2049) and pooled.shape == (1, 513)
full_cols = pooled.shape[1] * 64
assert full_cols == 32832

# Actual paged-indexer method: fixed crop only during short capture, preserving shared pointers and row padding.
calls = []


def fake_mqa(q, cache, weights, context_lens, pages, schedule, width, **kw):
    calls.append((width, pages.shape, pages.data_ptr(), pages.stride()))
    return torch.zeros(q.shape[0], width)


ns = source_functions("srt/layers/attention/dsa/dsa_indexer_kpool.py", {"_get_topk_paged"},
    {"torch": torch, "TYPE_CHECKING": False, "ForwardBatch": object, "BaseIndexerMetadata": object,
     "get_token_to_kv_pool": lambda: SimpleNamespace(page_size=64),
     "get_is_capture_mode": lambda: True,
     "is_cuda": lambda: False,
     "mqa_logits": SimpleNamespace(fp8_paged_mqa_logits=fake_mqa)})
lens = torch.tensor([14003, 3, 0])
pages = pooled.expand(3, -1).contiguous()
fake_self = SimpleNamespace(index_kpool=4,
    _get_index_k_read_buffer=lambda p, l: torch.zeros(1, 64 * 132, dtype=torch.uint8),
    _get_kpool_decode_metadata=lambda *a, **k: (lens // 4, (lens // 4).view(-1, 1), pages, None),
    _should_use_tilelang_paged_mqa_logits=lambda q: False,
    _kpool_fused_topk_mapping=lambda m: (None, None, None),
    _topk_from_kpool_logits=lambda logits, groups, **kw: topk(logits, groups, 4, 2048, **kw))
meta = SimpleNamespace(get_page_table_64=lambda: real.expand(3, -1), get_seqlens_int32=lambda: lens)
fb = batch(lens)
for label, width in (("sparse16k", 4096), ("sparse", 32832), (None, 32832)):
    _set_capture_attention_variant(label)
    out = ns["_get_topk_paged"](fake_self, fb, 3, torch.zeros(4, 32, 128), torch.zeros(4, 32, 1), meta)
    assert calls[-1][0] == width
    assert calls[-1][2] == pages.data_ptr()
    assert calls[-1][3] == pages.stride()
    assert out.shape == (4, 2051) and (out[3] == -1).all()
ns["get_is_capture_mode"] = lambda: False
_set_capture_attention_variant("sparse16k")
ns["_get_topk_paged"](fake_self, fb, 3, torch.zeros(3, 32, 128), torch.zeros(3, 32, 1), meta)
assert calls[-1][0] == 32832
_set_capture_attention_variant(None)

# Actual stable HIP selection: ties, causal future poison, exceptional logits, tail, batch changes and slot reuse.
for lens in ([14003], [2048], [2049], [7], [15000, 10, 6, 4], [1, 16384, 0], [9999]):
    rows = len(lens)
    seq = torch.tensor(lens)
    groups = seq // 4
    logits = torch.round(torch.randn(rows, full_cols) * 8) / 8
    for r, g in enumerate(groups.tolist()):
        logits[r, g:] = 9e20
    pages = torch.arange(131076).expand(rows, -1) + torch.arange(rows)[:, None] * 200000
    kw = dict(page_table=pages, seq_lens=seq, out_rows=4 if rows == 3 else None)
    a = topk(logits, groups, 4, 2048, **kw)
    b = topk(logits[:, :4096], groups, 4, 2048, **kw)
    assert torch.equal(a, b), lens
    assert a.shape[1] == 2051
    if rows == 3:
        assert (a[3] == -1).all()
for value in (float("nan"), float("inf"), -float("inf")):
    logits = torch.zeros(1, full_cols)
    logits[0, :3500] = value
    kw = dict(seq_lens=torch.tensor([14003]))
    assert torch.equal(topk(logits, torch.tensor([3500]), 4, 2048, **kw),
                       topk(logits[:, :4096], torch.tensor([3500]), 4, 2048, **kw))
assert not torch.cuda.is_initialized()
print("PASS: default-OFF GLM-only CPU selection; actual plain/speculative/draft/DLLM capture eligibility; fixed capture crop shares pointers; stable HIP topk/tails/NaN/padding/slot reuse identical")
