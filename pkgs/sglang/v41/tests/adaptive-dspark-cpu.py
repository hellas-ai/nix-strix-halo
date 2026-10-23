"""Exercise installed adaptive DSpark control flow with CPU tensors/device stubs.

No model, accelerator, collective or graph is initialized. This tests state
ownership and scheduling; GPU numerical/performance qualification is separate.
"""

import ast
import copy
import dataclasses
import logging
import sys
import time
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(sys.argv[1])
SRT = ROOT / "lib/python3.13/site-packages/sglang/srt"
if not SRT.exists():
    SRT = ROOT / "python/sglang/srt"  # pre-build source check
SOURCE = SRT / "speculative/dspark_components/dspark_worker_v2.py"
N = types.SimpleNamespace


class Bag(N):
    @contextmanager
    def override(self, **values):
        saved = {key: getattr(self, key) for key in values}
        self.__dict__.update(values)
        try:
            yield self
        finally:
            self.__dict__.update(saved)


class Toggle:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class Mode:
    def __init__(self, mode):
        self.mode = mode

    def is_extend(self):
        return self.mode == "prefill"

    def is_idle(self):
        return self.mode == "idle"


def with_phase(config, phase, **changes):
    result = copy.copy(config)
    result.decode = N(**(vars(config.decode) | changes))
    return result


# Compile the installed helpers and complete worker class, preserving its method
# bodies. Only imports, base worker initialization and GPU factories are stubbed.
module = types.ModuleType("installed_adaptive_dspark")
sys.modules[module.__name__] = module
ns = vars(module)
ns.update(
    torch=torch,
    time=time,
    get_available_gpu_memory=lambda *args: 20,
    dataclass=dataclasses.dataclass,
    contextmanager=contextmanager,
    nullcontext=nullcontext,
    BaseSpecWorker=object,
    TpModelWorker=object,
    logger=logging.getLogger(__name__),
    DraftBlockResult=N,
    VerifyWindow=N,
    Backend=N(FULL="full"),
    Phase=N(DECODE="decode"),
    with_phase=with_phase,
)
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
names = {
    "_VerifyState",
    "_adaptive_verify_width",
    "_slice_draft_block",
    "_slice_verify_window",
    "_decode_capture_config",
    "DSparkWorkerV2",
}
selected = [node for node in tree.body if getattr(node, "name", None) in names]
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    )
                ]
                + selected,
                type_ignores=[],
            )
        ),
        str(SOURCE),
        "exec",
    ),
    ns,
)
Worker = ns["DSparkWorkerV2"]
State = ns["_VerifyState"]


class AdaptiveTest(unittest.TestCase):
    def setUp(self):
        self.spec = Bag(speculative_num_draft_tokens=4)
        self.graph = Bag(
            cuda_graph_config=N(decode=N(bs=[1, 2], max_bs=2, backend="full")),
            cuda_graph_bs_decode=[1, 2],
        )
        self.parallel = N(enable_dp_attention=False, attn_cp_size=1, tp_size=4)
        self.schedule = N(disable_overlap_schedule=True, max_running_requests=4)
        self.envs = N(
            SGLANG_RAGGED_VERIFY_MODE=Toggle("static"),
            SGLANG_DSPARK_FOLDED_PROPOSAL=Toggle(True),
        )
        ns.update(
            get_spec=lambda: self.spec,
            get_exec=lambda: N(graph=self.graph),
            get_parallel=lambda: self.parallel,
            get_schedule=lambda: self.schedule,
            envs=self.envs,
            is_cuda_alike=lambda: True,
            draft_pp_context=nullcontext,
            build_block_pos_offsets=lambda length, device: torch.arange(length),
            DSparkVerifyPlanner=lambda **kw: N(
                **kw,
                mode_value="static",
                is_compact_mode=False,
                is_verify_all=True,
                carries_confidence=False,
            ),
            TargetHiddenKvInjector=lambda **kw: N(**kw),
            CommitInjectCtx=lambda **kw: N(**kw),
            DsparkVerifyEpilogue=self.epilogue,
            TargetVerifyExecutor=lambda **kw: N(**kw),
            DsparkStepObservers=lambda **kw: N(**kw),
        )
        self.worker = Worker.__new__(Worker)
        w = self.worker
        w._additional_graph_memory_usage = {}
        w._additional_graph_time_usage = {}
        w._target_worker = N()
        w.target_worker = w._target_worker
        w.model_runner = N(
            model_config=N(hf_text_config=N(model_type="deepseek_v41")),
            pp_size=1,
            tp_rank=0,
            capture_tail_hooks=[],
            init_new_workspace=False,
            attn_backend=N(width=4),
            decode_cuda_graph_runner=N(width=4, engram_graph_prestage=object()),
        )
        w._target_worker.model_runner = w.model_runner
        w.device = "cpu"
        w.gpu_id = 0
        w.gamma = 3
        w.verify_num_draft_tokens = 4
        w._adaptive_verify = True
        w._hosts_draft = True
        w._draft_is_moe = True
        w._is_pd_prefill = False
        w._draft_dp_context_enabled = False
        w._decode_graph_allowed = True
        w._simulate_acc_len = 0
        w._tp_sync = N(available_memory_gb=lambda *args, **kw: 20)
        w._draft_graph_group = object()
        w.draft_model = N(sample_from_anchor=True)
        w.draft_model_runner = N(attn_backend=object())
        w._base_capture_tail_hooks = [object()]
        state = w._build_verify_state(4, 2)
        w._verify_states = {4: state}
        w._apply_verify_state(state)
        self.captures = []
        self.fail_capture = False

        def backend(**kw):
            w.model_runner.init_new_workspace = True
            return N(width=self.spec.speculative_num_draft_tokens)

        w.model_runner._get_attention_backend = backend

        def capture(runner, *, attn_backend, speculative_num_draft_tokens):
            width = speculative_num_draft_tokens
            buckets = list(self.graph.cuda_graph_config.decode.bs)
            self.assertEqual(self.graph.cuda_graph_bs_decode, buckets)
            self.assertEqual(self.spec.speculative_num_draft_tokens, width)
            self.assertIs(runner.attn_backend, attn_backend)
            self.assertEqual(attn_backend.width, width)
            self.assertEqual(
                runner.capture_tail_hooks,
                w._base_capture_tail_hooks + [w._verify_epilogue.capture_hook],
            )
            self.assertEqual(w._verify_epilogue.verify_num_draft_tokens, width)
            self.assertLessEqual(max(buckets) * width, 8)
            self.captures.append((width, buckets))
            if self.fail_capture:
                raise RuntimeError("injected capture failure")
            return N(width=width, engram_graph_prestage=object())

        ns["DecodeCudaGraphRunner"] = capture

    @staticmethod
    def epilogue(**kw):
        return N(**kw, capture_hook=object(), draft_tokens_buf=object())

    def test_policy_and_guards(self):
        self.assertEqual(
            [ns["_adaptive_verify_width"](bs) for bs in range(5)], [4, 4, 4, 2, 2]
        )
        for bs in (-1, 5, 8):
            with self.assertRaises(ValueError):
                ns["_adaptive_verify_width"](bs)
        self.worker._validate_adaptive_verify()
        for obj, attr, value in (
            (self.worker, "gamma", 1),
            (self.parallel, "enable_dp_attention", True),
            (self.parallel, "tp_size", 1),
            (self.parallel, "attn_cp_size", 2),
            (self.schedule, "disable_overlap_schedule", False),
            (self.schedule, "max_running_requests", 2),
            (self.graph.cuda_graph_config.decode, "bs", [1, 2, 4]),
            (self.envs.SGLANG_RAGGED_VERIFY_MODE, "value", "compact"),
        ):
            old = getattr(obj, attr)
            setattr(obj, attr, value)
            with self.assertRaises(ValueError):
                self.worker._validate_adaptive_verify()
            setattr(obj, attr, old)

    def test_width_factories_and_graph_ownership(self):
        w = self.worker
        w._capture_adaptive_verify()
        self.assertEqual(self.captures, [(2, [1, 2, 4])])
        self.assertEqual(w._additional_graph_memory_usage, {"target_verify": 0})
        self.assertGreaterEqual(w._additional_graph_time_usage["target_verify"], 0)
        states = w._verify_states
        self.assertIsNot(
            states[2].graph_runner.engram_graph_prestage,
            states[4].graph_runner.engram_graph_prestage,
        )
        for width in (2, 4, 2, 4):
            state = states[width]
            w._apply_verify_state(state)
            self.assertIs(w.model_runner.decode_cuda_graph_runner, state.graph_runner)
            self.assertIs(w.model_runner.attn_backend, state.attn_backend)
            self.assertIs(w._verify_executor.kv_injector, state.injector)
            self.assertIs(w._verify_epilogue.commit_ctx.kv_injector, state.injector)
            self.assertEqual(w._verify_executor.gamma, width - 1)
            self.assertEqual(w._verify_planner.gamma, width - 1)
            self.assertEqual(w._observers.gamma, width - 1)
            self.assertEqual(w._verify_epilogue.max_bs, 2 if width == 4 else 4)
            self.assertEqual(w.speculative_num_draft_tokens, width)
            self.assertEqual(w.gamma, 3)
            self.assertEqual(self.spec.speculative_num_draft_tokens, 4)
        self.assertEqual(
            w.spec_v2_attn_backends,
            (
                states[4].attn_backend,
                states[2].attn_backend,
                w.draft_model_runner.attn_backend,
            ),
        )

    def test_capture_failure_restores_all_state(self):
        w = self.worker
        before = (
            w.model_runner.attn_backend,
            w.model_runner.decode_cuda_graph_runner,
            w.model_runner.capture_tail_hooks,
            self.graph.cuda_graph_config,
        )
        self.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            w._capture_adaptive_verify()
        self.assertEqual(
            (
                w.model_runner.attn_backend,
                w.model_runner.decode_cuda_graph_runner,
                w.model_runner.capture_tail_hooks,
                self.graph.cuda_graph_config,
            ),
            before,
        )
        self.assertFalse(w.model_runner.init_new_workspace)
        self.assertEqual(self.spec.speculative_num_draft_tokens, 4)
        self.assertEqual(self.graph.cuda_graph_bs_decode, [1, 2])
        self.assertEqual(list(w._verify_states), [4])
        self.assertEqual(w.verify_num_draft_tokens, 4)

    def test_draft_capture_keeps_gamma3_and_capacity4(self):
        w = self.worker
        ns["SpecTpSyncSite"] = N(DSPARK_MEM=object())
        w._proposer = N(attach_draft_sampler=lambda sampler: None)
        w._maybe_build_draft_sampler = lambda **kw: N(capture_hook=object())
        w.draft_model_runner.capture_tail_hooks = []
        ns["make_draft_sampler_capture_hook"] = lambda sampler: sampler.capture_hook

        def draft_capture(*, capture_decode_cuda_graph):
            self.assertTrue(capture_decode_cuda_graph)
            self.assertEqual(self.graph.cuda_graph_config.decode.bs, [1, 2, 4])
            self.assertEqual(self.spec.speculative_num_draft_tokens, 4)
            self.assertEqual(w.gamma, 3)

        w._draft_worker = N(init_cuda_graphs=draft_capture)
        w.init_cuda_graphs()
        self.assertEqual(self.graph.cuda_graph_config.decode.bs, [1, 2])
        self.assertEqual(w.verify_num_draft_tokens, 4)

    def test_low_memory_fails_instead_of_silent_eager(self):
        ns["SpecTpSyncSite"] = N(DSPARK_MEM=object())
        self.worker._tp_sync.available_memory_gb = lambda *args, **kw: 0.5
        with self.assertRaisesRegex(RuntimeError, "insufficient memory"):
            self.worker.init_cuda_graphs()

    def test_sampler_does_not_alias_verify_buffer(self):
        w = self.worker
        calls = []
        ns["maybe_build_draft_sampler"] = lambda **kw: calls.append(kw) or N()
        with ns["_decode_capture_config"]([1, 2, 4]):
            w._maybe_build_draft_sampler(available_memory_gb=20)
        self.assertEqual((calls[-1]["gamma"], calls[-1]["max_bs"]), (3, 4))
        self.assertIsNone(calls[-1]["out"])
        w._adaptive_verify = False
        w._maybe_build_draft_sampler(available_memory_gb=20)
        self.assertIs(calls[-1]["out"], w._verify_epilogue.draft_tokens_buf)

    def test_proposal_and_window_slicing(self):
        for bs in (1, 2, 3, 4):
            ids = torch.arange(bs * 3).reshape(bs, 3)
            logits = torch.arange(bs * 3 * 7).reshape(bs, 3, 7)
            block = N(
                draft_tokens=ids,
                corrected_logits=logits,
                greedy_mask=torch.arange(bs) % 2 == 0,
                temperatures=torch.ones(bs),
            )
            for width in (2, 4):
                sliced = ns["_slice_draft_block"](block, width)
                torch.testing.assert_close(sliced.draft_tokens, ids[:, : width - 1])
                torch.testing.assert_close(
                    sliced.corrected_logits, logits[:, : width - 1]
                )
                self.assertTrue(sliced.draft_tokens.is_contiguous())
                self.assertTrue(sliced.corrected_logits.is_contiguous())
                self.assertIs(sliced.greedy_mask, block.greedy_mask)
                self.assertIs(sliced.temperatures, block.temperatures)
            block.corrected_logits = None
            self.assertIsNone(ns["_slice_draft_block"](block, 2).corrected_logits)
            self.assertEqual(block.draft_tokens.shape, (bs, 3))
            locations = torch.arange(bs * 4).reshape(bs, 4)
            window = N(
                positions_2d=locations + 100,
                verify_cache_loc=locations.flatten(),
                verify_cache_loc_2d=locations,
            )
            self.assertIs(ns["_slice_verify_window"](window, 4), window)
            narrow = ns["_slice_verify_window"](window, 2)
            torch.testing.assert_close(
                narrow.verify_cache_loc, locations[:, :2].flatten()
            )
            torch.testing.assert_close(narrow.positions_2d, locations[:, :2] + 100)
            self.assertTrue(narrow.positions_2d.is_contiguous())
            self.assertEqual(window.positions_2d.shape, (bs, 4))

    def test_batch_transitions_scope_width_and_restore_maximum(self):
        w = self.worker
        w._capture_adaptive_verify()
        for state in w._verify_states.values():
            state.planner.note_non_decode_step = lambda: None
            state.observers.note_prefill_step = lambda: None

        def forward(*args):
            self.assertEqual(
                w.verify_num_draft_tokens, self.spec.speculative_num_draft_tokens
            )
            return w.verify_num_draft_tokens

        w._forward_decode = forward
        w._forward_prefill = forward
        for mode, bs, expected in [
            ("decode", 1, 4),
            ("decode", 3, 2),
            ("decode", 4, 2),
            ("prefill", 4, 4),
            ("decode", 2, 4),
            ("idle", 0, 4),
        ]:
            batch = N(
                forward_mode=Mode(mode),
                is_extend_in_batch=False,
                seq_lens=torch.zeros(bs, dtype=torch.int64),
            )
            self.assertEqual(w.forward_batch_generation(batch), expected)
            self.assertEqual(self.spec.speculative_num_draft_tokens, 4)

        def failing_forward(*args):
            raise RuntimeError("injected forward failure")

        w._forward_decode = failing_forward
        with self.assertRaisesRegex(RuntimeError, "injected forward"):
            w.forward_batch_generation(
                N(
                    forward_mode=Mode("decode"),
                    is_extend_in_batch=False,
                    seq_lens=torch.zeros(3, dtype=torch.int64),
                )
            )
        self.assertEqual(self.spec.speculative_num_draft_tokens, 4)

    def test_decode_uses_full_draft_window_then_narrows_verify(self):
        # Run the actual decode entry through proposal and verification planning.
        # Stop before kernels/collectives; these require GPU qualification.
        w = self.worker
        w._capture_adaptive_verify()
        w._apply_verify_state(w._verify_states[2])
        w._draft_block_pos_offsets = torch.arange(4)
        w.model_runner.model = object()
        ns["DFlashDraftInputV2"] = type("DraftInput", (), {})
        ns["InfoSegment"] = N(DRAFT=0)
        w._observers.begin_step = lambda: None
        w._observers.segment = lambda segment: nullcontext()
        bs = 3
        batch = N(
            spec_info=ns["DFlashDraftInputV2"](),
            forward_mode=Mode("decode"),
            seq_lens=torch.arange(bs),
            sampling_info=None,
            req_pool_indices=torch.arange(bs),
        )

        def alloc_window(**kw):
            self.assertEqual(kw["verify_num_draft_tokens"], 4)
            self.assertEqual(kw["block_pos_offsets"].shape, (4,))
            loc = torch.arange(bs * 4).reshape(bs, 4)
            return N(
                positions_2d=loc + 100,
                verify_cache_loc=loc.flatten(),
                verify_cache_loc_2d=loc,
            )

        ns["alloc_verify_window"] = alloc_window

        def propose(**kw):
            self.assertEqual(self.spec.speculative_num_draft_tokens, 4)
            self.assertEqual(kw["verify_window"].positions_2d.shape, (bs, 4))
            return N(
                draft_block_ids=torch.ones(bs, 3, dtype=torch.int64),
                draft_block=N(
                    draft_tokens=torch.arange(bs * 3).reshape(bs, 3),
                    corrected_logits=None,
                    greedy_mask=torch.ones(bs, dtype=torch.bool),
                    temperatures=torch.ones(bs),
                ),
                confidence=None,
                draft_hidden=None,
                confidence_tap=None,
            )

        w._proposer = N(propose=propose)

        def confidence(**kw):
            self.assertEqual(self.spec.speculative_num_draft_tokens, 2)
            torch.testing.assert_close(
                kw["draft_tokens"], torch.tensor([[0], [3], [6]])
            )
            raise RuntimeError("reached narrow verification")

        w._verify_planner.compute_confidence_tensor = confidence
        with (
            patch.object(torch.Tensor, "record_stream", lambda *args: None),
            patch.object(
                torch, "get_device_module", lambda *args: N(current_stream=lambda: None)
            ),
            self.spec.override(speculative_num_draft_tokens=2),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached narrow verification"):
                w._forward_decode(batch, None)
        self.assertEqual(self.spec.speculative_num_draft_tokens, 4)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
