"""Exercise actual host accounting across adaptive widths and overlap results."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Union

import torch
import sglang

root = Path(os.environ.get("GLM53_TEST_METRICS_ROOT", Path(sglang.__file__).parent))
source = root / "srt/speculative/adaptive_spec_params.py"
tree = ast.parse(source.read_text())
names = {"glm53_chain_width_metrics_enabled", "glm53_chain_num_proposed_drafts"}
ns = {"os": os, "GLM53_MTP_VERIFY_WIDTHS_ENV": "SGLANG_GLM53_MTP_VERIFY_WIDTHS"}
exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], []), str(source), "exec"), ns)
enabled, count = (ns[n] for n in ("glm53_chain_width_metrics_enabled", "glm53_chain_num_proposed_drafts"))
for name in ("SGLANG_GLM53_MTP_VERIFY_WIDTHS", "SGLANG_GLM53_DFLASH_VERIFY_WIDTHS"):
    os.environ.pop(name, None)
assert not enabled() and count(8, 16, 24) == 24
os.environ["SGLANG_GLM53_MTP_VERIFY_WIDTHS"] = "4,4,2,2"
assert enabled() and count(8, 16, 24) == 8 and count(8, 8, 24) == 0

source = root / "srt/managers/scheduler_components/batch_result_processor.py"
tree = ast.parse(source.read_text())
functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in {"_get_speculative_output_stride", "_resolve_spec_v2_tokens"}]
ns = {"GenerationBatchResult": object, "ScheduleBatch": object, "List": list,
      "glm53_chain_width_metrics_enabled": enabled}
exec(compile(ast.Module(functions, []), str(source), "exec"), ns)
req = SimpleNamespace(is_retracted=False, finished=lambda: False, grammar=None,
                      kv=SimpleNamespace(kv_committed_len=0), spec_verify_ct=0,
                      spec_num_correct_drafts=0, spec_num_cap_tokens=0,
                      spec_num_block_accept_tokens=0)
accepted_hist, cap_hist = [], []
req.update_spec_correct_drafts_histogram = accepted_hist.append
req.update_spec_cap_lens_histogram = cap_hist.append
processor = SimpleNamespace(model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
                            advance_grammar_fsm=lambda *a: None)
for width, accepted in ((4, 2), (2, 1), (4, 3), (1, 0), (2, 0), (4, 1)):
    result = SimpleNamespace(next_token_ids=torch.arange(width),
        accept_lens=torch.tensor([accepted + 1]), speculative_output_stride=None,
        speculative_num_draft_tokens=width, num_non_draft_tokens_per_req=1,
        block_accept_lens=None, cap_lens=None)
    out = ns["_resolve_spec_v2_tokens"](processor, result, SimpleNamespace(reqs=[req]))
    assert out == [list(range(accepted + 1))] and result.num_cap_tokens == width
assert req.spec_verify_ct == 6 and req.spec_num_cap_tokens == 17
assert cap_hist == [4, 2, 4, 1, 2, 4] and req.spec_num_correct_drafts == 7
assert count(req.spec_verify_ct, req.spec_num_cap_tokens, 18) == 11
# An explicit ragged cap is authoritative, even if the output allocation is wider.
result.cap_lens = torch.tensor([2])
ns["_resolve_spec_v2_tokens"](processor, result, SimpleNamespace(reqs=[req]))
assert result.num_cap_tokens == 2 and cap_hist[-1] == 2
os.environ.pop("SGLANG_GLM53_MTP_VERIFY_WIDTHS")
os.environ["SGLANG_GLM53_DFLASH_VERIFY_WIDTHS"] = "8,4,4,2"
assert count(3, 8 + 4 + 2, 21) == 11
source = root / "srt/managers/tokenizer_manager.py"
tree = ast.parse(source.read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_calculate_spec_decoding_metrics")
ns = {"Dict": Dict, "Any": Any, "Union": Union, "BatchStrOutput": object,
      "BatchEmbeddingOutput": object, "BatchTokenIDOutput": object,
      "get_spec": lambda: SimpleNamespace(speculative_num_draft_tokens=8),
      "glm53_chain_num_proposed_drafts": count, "glm53_chain_width_metrics_enabled": enabled,
      "_ragged_verify_cap_accept": lambda: False}
exec(compile(ast.Module([method], []), str(source), "exec"), ns)
recv = SimpleNamespace(spec_verify_ct=[3], spec_num_correct_drafts=[6],
                       spec_num_cap_tokens=[14], completion_tokens=[9],
                       spec_correct_drafts_histogram=None)
meta = {}
ns[method.name](None, meta, recv, 0)
assert meta["spec_num_proposed_drafts"] == 11 and meta["spec_accept_rate"] == 6 / 11
recv.spec_num_cap_tokens, recv.spec_num_correct_drafts, recv.completion_tokens = [3], [0], [3]
ns[method.name](None, meta, recv, 0)
assert meta["spec_verify_ct"] == 3 and meta["spec_num_proposed_drafts"] == 0 and meta["spec_accept_length"] == 1
print("PASS: actual-width proposal accounting across 4/2/4/1/2/4, stable fallback, DFlash widths, explicit caps")
