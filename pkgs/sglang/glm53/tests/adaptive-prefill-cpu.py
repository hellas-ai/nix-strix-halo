"""CPU tests of the installed scheduler/KDA checkpoint methods, without HIP.

The AST loader executes the actual installed methods with small CPU fixtures;
it does not reimplement their boundary arithmetic. Pass site-packages or a
runtime root. No GPU initialization or model loading is required.
"""

import ast
import importlib.util
import math
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS

import torch

root = Path(sys.argv.pop(1))
if (root / "lib/python3.13/site-packages").exists():
    root /= "lib/python3.13/site-packages"
package = root / "sglang/srt"
spec = importlib.util.spec_from_file_location(
    "adaptive_policy", package / "arg_groups/glm53_adaptive_prefill.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

# 0063 adds a default-off guard to the actual cache donation method. Keep its
# real installed module in the AST namespace rather than replacing the guard.
geometry = None
geometry_path = package / "managers/glm53_prefill_geometry_trace.py"
if geometry_path.exists():
    spec = importlib.util.spec_from_file_location("installed_glm53_geometry", geometry_path)
    geometry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = geometry
    spec.loader.exec_module(geometry)
    assert not geometry.ENABLED, "Adaptive CPU fixture requires the geometry trace disabled"


def method(path, name, namespace):
    tree = ast.parse((package / path).read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, name
    node = found[0]
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(package / path), "exec"), namespace)
    return namespace[name]


class AdaptivePrefill(unittest.TestCase):
    def setUp(self):
        policy.ENABLED = True

    def tearDown(self):
        policy.ENABLED = False

    def test_local_budgets_and_static_ceiling(self):
        self.assertEqual(policy.buffer_ceiling(1024), 4096)
        self.assertEqual(policy.buffer_ceiling(8192), 8192)
        for running, waiting, continuation, expected in [
            (0, 1, False, 4096), (0, 0, True, 4096),
            (1, 1, False, 1024), (1, 0, True, 1024),
            (0, 1, True, 1024), (0, 3, False, 1024),
        ]:
            self.assertEqual(policy.batch_budget(1024, running, waiting, continuation), expected)
        policy.ENABLED = False
        self.assertEqual(policy.buffer_ceiling(1024), 1024)
        self.assertIsNone(policy.batch_budget(None, 0, 1, False))
        policy.validate({"unsupported scope": False})

    def test_scheduler_uses_local_budget_without_mutating_startup_grid(self):
        tree = ast.parse((package / "managers/scheduler.py").read_text())
        raw = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw")
        assignment = next(n for n in raw.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "batch_budget")
        for running, waiting, continuing, expected in [(0, 0, True, 4096), (1, 0, True, 1024), (0, 2, False, 1024)]:
            scheduler = NS(chunked_prefill_size=1024, waiting_queue=[None] * waiting,
                           chunked_req=object() if continuing else None)
            ns = {"glm53_adaptive_prefill": policy, "self": scheduler, "running_bs": running}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), "scheduler-local-budget", "exec"), ns)
            self.assertEqual(ns["chunked_prefill_size"], expected)
            self.assertEqual(scheduler.chunked_prefill_size, 1024)

    def test_actual_scheduler_scope_guard(self):
        tree = ast.parse((package / "managers/scheduler.py").read_text())
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "init_chunked_prefill")
        guard = next(n for n in init.body if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                     and n.test.attr == "ENABLED")
        model = NS(hf_text_config=NS(model_type="glm5_next_text", mamba_chunk_size=64))
        execution = NS(graph=NS(cuda_graph_config=NS(prefill=NS(backend="disabled"))),
                       mamba=NS(enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False),
                       moe=NS(moe_runner_backend="triton", enable_eplb=False, expert_distribution_recorder_mode=None))
        scheduler = NS(model_config=model, chunked_prefill_size=1024, max_prefill_tokens=16384,
                       ps=NS(tp_size=4, pp_size=1, dp_size=1, attn_cp_size=1, attn_dcp_size=1, moe_ep_size=1),
                       is_hybrid_ssm=True, is_hybrid_swa=False, disable_radix_cache=False,
                       tree_cache=NS(page_size=1024), spec_algorithm=NS(is_none=lambda: True),
                       is_mixed_chunk=False, enable_hierarchical_cache=False,
                       tp_worker=NS(model_runner=NS(linear_attn_backends=NS(prefill=NS(is_triton=lambda: True)))))
        ns = {"self": scheduler, "glm53_adaptive_prefill": policy,
              "get_exec": lambda: execution, "get_schedule": lambda: NS(enable_dynamic_chunking=False),
              "get_disagg": lambda: NS(disaggregation_mode="null"), "TAIL_MERGE_ROWS": 0,
              "logger": NS(info=lambda *args: None)}
        code = compile(ast.Module(body=[guard], type_ignores=[]), "scheduler-scope", "exec")
        exec(code, ns)
        for target, field, bad in [(scheduler, "chunked_prefill_size", 4096),
                                   (scheduler.tree_cache, "page_size", 4096),
                                   (scheduler.ps, "pp_size", 2),
                                   (scheduler, "max_prefill_tokens", 1024),
                                   (execution.mamba, "enable_mamba_extra_buffer", False),
                                   (execution.graph.cuda_graph_config.prefill, "backend", "breakable")]:
            old = getattr(target, field)
            setattr(target, field, bad)
            with self.assertRaises(ValueError):
                exec(code, ns)
            setattr(target, field, old)
        ns["TAIL_MERGE_ROWS"] = 1
        with self.assertRaises(ValueError):
            exec(code, ns)
        policy.ENABLED = False
        # Disabled branch must not even evaluate experimental scope fields.
        ns["self"] = object()
        exec(code, ns)

    def test_installed_buffer_hook_preserves_provider_and_default(self):
        ns = {"buffer_ceiling": policy.buffer_ceiling, "_prefill_buffer_ceiling_fn": None}
        ceiling = method("arg_groups/prefill_buffer_ceiling.py", "prefill_buffer_ceiling_of", ns)
        self.assertEqual(ceiling(object(), 1024), 4096)
        ns["_prefill_buffer_ceiling_fn"] = lambda record, default: 8192
        self.assertEqual(ceiling(object(), 1024), 8192)
        policy.ENABLED = False
        self.assertEqual(ceiling(object(), 1024), 8192)

    def test_config_and_published_activation_reserve_agree(self):
        schedule = NS(chunked_prefill_size=1024, max_prefill_tokens=16384, max_running_requests=4)
        cfg = NS(**vars(schedule), disaggregation_mode="null", tp_size=4, pp_size=1)
        config_fn = method("arg_groups/overrides.py", "pre_capture_activation_reserve_mb_of", {"buffer_ceiling": policy.buffer_ceiling})
        live_fn = method("runtime_context.py", "pre_capture_activation_reserve_mb", {
            "buffer_ceiling": policy.buffer_ceiling, "get_schedule": lambda: schedule,
            "get_parallel": lambda: NS(tp_size=4, pp_size=1),
            "get_disagg": lambda: NS(disaggregation_mode="null"),
        })
        for enabled, expected in [(False, 4096), (True, 7168)]:
            policy.ENABLED = enabled
            self.assertEqual(config_fn(cfg, 32000), expected)
            self.assertEqual(live_fn(32000), expected)
            self.assertEqual(config_fn(cfg, 128000), 10240)
            self.assertEqual(live_fn(128000), 10240)

    def checkpoint(self, prefix, length, branch=None):
        entry = namedtuple("Track", "track_mask track_index track_seqlen")
        ns = {"glm53_adaptive_prefill": policy, "mamba_cache_chunk_size": lambda: 64,
              "mamba_checkpoint_grid": lambda page: math.lcm(64, page),
              "get_exec": lambda: NS(mamba=NS(enable_mamba_extra_buffer_lazy=False)),
              "_MambaRadixCacheV2TrackEntry": entry}
        fn = method("managers/schedule_batch.py", "_mamba_radix_cache_v2_req_prepare_for_extend", ns)
        req = NS(prefix_indices=range(prefix), extend_range=NS(length=length),
                 mamba_branching_seqlen=branch,
                 kv=NS(mamba_ping_pong_track_buffer=torch.tensor([10, 11]),
                       mamba_next_track_idx=0, mamba_last_track_seqlen=None))
        batch = NS(tree_cache=NS(page_size=1024),
                   model_config=NS(hf_text_config=NS(mamba_chunk_size=64)),
                   req_to_token_pool=NS(get_mamba_ping_pong_other_idx=lambda i: 1-i))
        return req, fn(batch, req)

    def test_exact_final_and_latest_real_interior_checkpoint(self):
        for prefix, length, depth, track in [
            (0, 4096, 4096, 4096), (4096, 1024, 5120, 5120),
            (5120, 4096, 9216, 9216), (94208, 3484, 97280, 97281),
            (8192, 3136, 11264, 11265), (8192, 3100, 11264, 11292),
            (576, 1024, 1024, 1025), (1600, 600, 2048, 2049),
            (576, 448, 1024, 1024), (1600, 448, 2048, 2048),
        ]:
            req, result = self.checkpoint(prefix, length)
            self.assertTrue(result.track_mask)
            self.assertEqual(req.kv.mamba_last_track_seqlen, depth)
            self.assertEqual(result.track_seqlen, track)
            self.assertEqual(req.kv.mamba_next_track_idx, 1)
        req, result = self.checkpoint(114688, 314)
        self.assertFalse(result.track_mask)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        self.assertEqual(req.kv.mamba_next_track_idx, 0)
        req, result = self.checkpoint(8192, 4096, 10240)
        self.assertEqual(req.kv.mamba_last_track_seqlen, 10240)
        self.assertEqual(result.track_seqlen, 10241)
        req, result = self.checkpoint(0, 576)
        self.assertFalse(result.track_mask)
        req, result = self.checkpoint(1536, 127)
        self.assertFalse(result.track_mask)

    def test_checkpoint_rejects_unallocated_or_unaligned_geometry(self):
        for prefix, length, branch in [(1, 4096, None), (0, 4097, None), (0, 4096, 64)]:
            with self.assertRaises(AssertionError):
                self.checkpoint(prefix, length, branch)

    def test_kda_and_conv_use_the_same_actual_depth_and_batch_row(self):
        # Final4K, interior tail, untracked suffix, then a partial-prefix row.
        fb = NS(extend_seq_lens=torch.tensor([4096, 3484, 314, 1024]),
                extend_prefix_lens=torch.tensor([0, 94208, 114688, 576]),
                mamba_track_mask=torch.tensor([True, True, False, True]),
                mamba_track_indices=torch.tensor([10, 11, 12, 13]),
                mamba_track_seqlens=torch.tensor([4096, 97281, -1, 1025]))
        aligned = method("model_executor/forward_batch_info.py", "mamba_track_aligned_lens", {"mamba_cache_chunk_size": lambda: 64})
        fb.mamba_track_aligned_lens = lambda: aligned(fb)
        backend = NS(mamba_chunk_size=64, device="cpu", conv_states_shape=(100, 3))
        ssm_fn = method("layers/attention/hybrid_linear_attn_backend.py", "_init_track_ssm_indices", {"torch": torch, "Mamba2AttnBackend": type("Other", (), {})})
        indices = ssm_fn(backend, torch.tensor([1, 2, 3, 4]), fb)
        self.assertEqual(indices[0].tolist(), [-1, 48, -1, 7])
        self.assertEqual(indices[2].tolist(), [11, 13])
        self.assertEqual(indices[3].tolist(), [1, 3])
        self.assertEqual(indices[4].tolist(), [1])
        self.assertEqual(indices[5].tolist(), [10])
        conv_fn = method("layers/attention/hybrid_linear_attn_backend.py", "_init_track_conv_indices", {"torch": torch})
        conv = conv_fn(backend, torch.tensor([0, 4096, 7580, 7894, 8918]), fb)
        self.assertEqual(conv.tolist(), [[4093, 4094, 4095], [7165, 7166, 7167], [8339, 8340, 8341]])
        copy_fn = method("layers/attention/hybrid_linear_attn_backend.py", "_track_mamba_state_extend", {})
        states = torch.arange(20, dtype=torch.float32).view(20, 1)
        metadata = NS(has_mamba_track_mask=True, track_ssm_h_src=indices[1],
                      track_ssm_h_dst=indices[2], track_ssm_h_batch_src=indices[3],
                      track_ssm_final_src=indices[4], track_ssm_final_dst=indices[5],
                      track_ssm_recompute_dst=None)
        # Deliberately different low-precision h: explicit FP32 batch-row snapshot wins.
        copy_fn(backend, fb, torch.full((1, 200, 1), -7, dtype=torch.bfloat16), states,
                metadata, h_track_buf=torch.tensor([[101.25], [202.5], [303.75], [404.25]]))
        self.assertEqual(states[10].item(), 1)
        self.assertEqual(states[11].item(), 202.5)
        self.assertEqual(states[13].item(), 404.25)

    def test_actual_cache_donates_only_saved_depth_and_keeps_continuation_row(self):
        # Actual tail metadata: 4K prefix + 3484 rows, checkpoint at 7168.
        req, entry = self.checkpoint(4096, 3484)
        req.kv.req_pool_idx = 1
        req.kv.cache_protected_len = 4096
        req.get_fill_ids = lambda: list(range(7580))
        req.extra_key = "adaptive-fixture"
        req.cache_salt = None
        req.last_node = object()
        rows = torch.arange(7580, dtype=torch.int32).expand(2, -1).clone()
        mapping = torch.zeros((2, 2), dtype=torch.int64)
        pool = NS(req_to_token=rows, mamba_pool=NS(),
                  req_index_to_mamba_ping_pong_track_buffer_mapping=mapping,
                  get_mamba_ping_pong_keep_idx=lambda r: r.kv.mamba_last_track_idx,
                  write=lambda index, values: rows.__setitem__(index, values.to(rows.dtype)))
        setter = method("mem_cache/memory_pool.py", "set_mamba_ping_pong_slot", {})
        donate = method("mem_cache/memory_pool.py", "donate_mamba_ping_pong_slot", {"_MAMBA_DEBUG_ASSERTS": True})
        pool.set_mamba_ping_pong_slot = lambda *args: setter(pool, *args)
        pool.donate_mamba_ping_pong_slot = lambda *args: donate(pool, *args)
        inserted = []
        node = NS(mamba_value=torch.tensor([entry.track_index]))
        def insert(params):
            inserted.append(params)
            return NS(prefix_len=len(params.value), mamba_exist=False)
        tree = NS(req_to_token_pool=pool, enable_mamba_extra_buffer=True, disable=False,
                  int8_ckpt_pool=None, page_size=1024, mamba_cache_chunk_size=64,
                  _alloc_mamba_slot=lambda: torch.tensor([99]), insert=insert,
                  match_prefix=lambda params: NS(device_indices=torch.arange(7168), last_device_node=node),
                  dec_lock_ref=lambda node: None, inc_lock_ref=lambda node: None)
        namespace = {
            "torch": torch, "InsertParams": NS, "MatchPrefixParams": NS,
            "RadixKey": lambda tokens, extra, **kwargs: NS(tokens=tokens, extra=extra, **kwargs),
        }
        if geometry is not None:
            namespace["glm53_geometry"] = geometry
        cache = method("mem_cache/mamba_radix_cache.py", "cache_unfinished_req", namespace)
        cache(tree, req, chunked=True)
        self.assertEqual(len(inserted), 1)
        self.assertEqual(len(inserted[0].key.tokens), 7168)
        self.assertEqual(len(inserted[0].value), 7168)
        self.assertEqual(inserted[0].mamba_value.tolist(), [10])
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [99, 11])
        self.assertEqual(mapping[1].tolist(), [99, 11])
        self.assertEqual(req.kv.cache_protected_len, 7168)
        self.assertEqual(len(req.prefix_indices), 7580)
        self.assertEqual(req.kv.req_pool_idx, 1)


unittest.main(verbosity=2)
