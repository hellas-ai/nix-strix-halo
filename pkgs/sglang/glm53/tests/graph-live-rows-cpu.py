"""Replay the actual graph buffer registry across changing request counts."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import sglang
from sglang.srt.model_executor import forward_batch_info as fbi

torch.set_num_threads(1)
root = Path(sglang.__file__).parent
path = Path(os.environ.get("GLM53_TEST_REGISTRY_SOURCE", root / "srt/model_executor/cuda_graph_buffer_registry.py"))
spec = importlib.util.spec_from_file_location("graph_registry_test", path)
registry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = registry
spec.loader.exec_module(registry)
parallel = SimpleNamespace(moe_ep_size=1, attn_tp_size=4, attn_tp_rank=0)
fbi.get_parallel = lambda: parallel

for width in (1, 2, 4, 8):
    for sharded in (False, True):
        for rank in range(4):
            parallel.attn_tp_rank = rank
            scalar = torch.tensor([4 * width], dtype=torch.int32)
            base = registry.build_decode_registry(
                device=torch.device("cpu"), max_bs=4, max_num_token=4 * width,
                seq_len_fill_value=1, cache_loc_dtype=torch.int64, share_pool=False,
            )
            backing = {name: base.get_slot(name).buffer for name in base.slot_names()}
            backing["num_token_non_padded"] = scalar
            graph = registry.build_decode_registry(
                device=torch.device("cpu"), max_bs=4, max_num_token=4 * width,
                seq_len_fill_value=1, cache_loc_dtype=torch.int64,
                enable_num_token_non_padded=True, allow_missing_num_token_non_padded=True,
                attn_tp_sharded_fn=lambda n: sharded, require_gathered_buffer=True,
                share_pool=False, source=SimpleNamespace(**backing),
            )
            assert graph.get_slot("num_token_non_padded").buffer.data_ptr() == scalar.data_ptr()
            for active in (4, 3, 1, 0, 2, 4, 3):
                rows, padded = active * width, 4 * width
                fb = SimpleNamespace(
                    global_num_token_non_padded=None, num_token_non_padded=None,
                    input_ids=torch.arange(rows), positions=torch.arange(rows),
                    req_pool_indices=torch.arange(active), out_cache_loc=torch.arange(rows),
                    seq_lens=torch.ones(active, dtype=torch.int32), seq_lens_cpu=torch.ones(active, dtype=torch.int32),
                )
                graph.fill_from(fb, raw_bs=active, padded_bs=4, raw_num_tokens=rows, padded_num_tokens=padded)
                expected = min(max(rows - rank * width, 0), width) if sharded else rows
                assert scalar.item() == expected, (width, active, rank, sharded, scalar)
                # Communication geometry remains padded while routing uses live rows.
                assert graph.get_slot("global_num_tokens_gpu").buffer.item() == padded
                assert torch.all(graph.get_slot("positions").buffer[rows:padded] == 0)
                assert scalar.data_ptr() == graph.get_slot("num_token_non_padded").buffer.data_ptr()
            # Preserve an explicit local count; use the invariant global GPU
            # count when present rather than inferring from current FB shape.
            fb.num_token_non_padded = torch.tensor([1], dtype=torch.int32)
            graph.fill_from(fb, raw_bs=3, padded_bs=4, raw_num_tokens=3 * width, padded_num_tokens=4 * width)
            assert scalar.item() == 1
            fb.global_num_token_non_padded = torch.tensor(2 * width, dtype=torch.int32)
            graph.fill_from(fb, raw_bs=3, padded_bs=4, raw_num_tokens=3 * width, padded_num_tokens=4 * width)
            expected = min(max(2 * width - rank * width, 0), width) if sharded else 2 * width
            assert scalar.item() == expected

# The opt-in is restricted to HIP EP1 and the runner excludes ragged layouts.
runner = Path(os.environ.get("GLM53_TEST_RUNNER_SOURCE", root / "srt/model_executor/runner/decode_cuda_graph_runner.py"))
tree = ast.parse(runner.read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_glm53_live_rows_enabled")
ns = {"torch": torch, "os": os, "get_parallel": lambda: parallel}
exec(compile(ast.Module([helper], []), str(runner), "exec"), ns)
os.environ["SGLANG_GLM53_LIVE_ROWS"] = "0"
assert not ns["_glm53_live_rows_enabled"]()
os.environ["SGLANG_GLM53_LIVE_ROWS"] = "1"
assert ns["_glm53_live_rows_enabled"]() == (torch.version.hip is not None)
parallel.moe_ep_size = 2
assert not ns["_glm53_live_rows_enabled"]()

draft = Path(os.environ.get("GLM53_TEST_DRAFT_GRAPH_SOURCE", root / "srt/speculative/eagle_draft_cuda_graph_runner.py"))
tree = ast.parse(draft.read_text())
refresh = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_glm53_set_num_token_non_padded")
ns = {"compute_local_num_token_non_padded_cpu": fbi.compute_local_num_token_non_padded_cpu}
exec(compile(ast.Module([refresh], []), str(draft), "exec"), ns)
for sharded in (False, True):
    for rank in range(4):
        parallel.attn_tp_rank = rank
        scalar = torch.tensor([4], dtype=torch.int32)
        draft_runner = SimpleNamespace(_glm53_num_token_non_padded=scalar,
            model_runner=SimpleNamespace(attn_tp_sequence_sharded=lambda n: sharded))
        address = scalar.data_ptr()
        for rows in (4, 3, 1, 0, 2, 4):
            ns["_glm53_set_num_token_non_padded"](draft_runner, rows, 4)
            expected = min(max(rows - rank, 0), 1) if sharded else rows
            assert scalar.item() == expected and scalar.data_ptr() == address
draft_runner._glm53_num_token_non_padded = None
ns["_glm53_set_num_token_non_padded"](draft_runner, 3, 4)
print("PASS: live graph rows across arrivals, widths1/2/4/8, replicated/sharded TP4, stable scalar pointers, padded communication geometry")
