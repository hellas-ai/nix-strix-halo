"""Offline qualification of HIP sampling without acquiring a campaign GPU.

Run with the built runtime's sglang-python and TRITON_INTERPRET=1. Tests execute
the installed proposal/target transforms and rejection kernel, not a copy of
their implementations. Meta tensors exercise capture device placement.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

os.environ["TRITON_INTERPRET"] = "1"
import torch
import sglang

torch.set_num_threads(1)
root = Path(sglang.__file__).parent


def functions(path, names, namespace):
    mod = ast.parse((root / path).read_text())
    selected = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(selected, []), str(path), "exec"), namespace)
    return namespace


# GPU capture metadata can be validated without a GPU by using meta tensors.
graph = ast.parse((root / "srt/speculative/eagle_draft_cuda_graph_runner.py").read_text())
metadata = next(
    n for n in ast.walk(graph)
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "sampling_info" for t in n.targets)
)
env = {"torch": torch, "SamplingBatchInfo": SimpleNamespace, "num_seqs": 3,
       "self": SimpleNamespace(temperatures=torch.ones(3, 1, device="meta"),
                               model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=8)))}
exec(compile(ast.Module([metadata], []), "graph metadata", "exec"), env)
si = env["sampling_info"]
assert {si.temperatures.device.type, si.top_ps.device.type, si.top_ks.device.type, si.min_ps.device.type} == {"meta"}

spec = functions("srt/speculative/spec_utils.py", ["glm53_draft_temperatures", "glm53_draft_topk", "fast_sample", "sample_draft_proposal"],
                 {"torch": torch, "_is_hip": False,
                  "fast_topk": lambda probs, k, dim: torch.topk(probs, k, dim=dim),
                  "envs": SimpleNamespace(SGLANG_OPT_USE_GUMBEL_SAMPLE=SimpleNamespace(get=lambda: True))})
assert spec["glm53_draft_temperatures"](si).device.type == "meta"
mixed = SimpleNamespace(temperatures=torch.tensor([[1.0], [0.7], [1.0]]), top_ks=torch.tensor([1, 100, 1]))
assert torch.equal(spec["glm53_draft_temperatures"](mixed), torch.tensor([[1e-5], [0.7], [1e-5]]))

# Deliberately divergent rank RNG: the broadcast replaces a non-root proposal
# before its probability is gathered. The returned q remains the actual draw q.
calls = []
fake_group = SimpleNamespace(world_size=4, broadcast=lambda token, src: (calls.append((token.clone(), src)), token.fill_(2)))
sys.modules["sglang.srt.distributed"] = SimpleNamespace(get_tp_group=lambda: fake_group)
spec["_is_hip"] = True
q, qx, x = spec["sample_draft_proposal"](torch.tensor([[1., 2., 3.]], dtype=torch.bfloat16), torch.ones(1, 1))
assert calls and calls[0][1] == 0 and x.item() == 2
assert q.dtype == torch.float32 and torch.equal(qx, q[:, 2:3])
before = len(calls)
probs = torch.tensor([[.7, .2, .1]])
qx, x = spec["glm53_draft_topk"](probs, 1)
assert len(calls) == before + 1 and x.item() == 2 and qx.item() == probs[0, 2].item()
qx, x = spec["glm53_draft_topk"](probs, 2)
assert len(calls) == before + 1 and torch.equal(x, torch.tensor([[0, 1]]))
spec["_is_hip"] = False
qx, x = spec["glm53_draft_topk"](probs, 1)
assert len(calls) == before + 1 and x.item() == 0
worker = (root / "srt/speculative/eagle_worker_v2.py").read_text()
assert worker.count("glm53_draft_topk(probs, self.topk)") == 3
assert "fast_topk(probs, self.topk, dim=-1)" not in worker

target = functions("srt/speculative/eagle_utils.py", ["_glm53_hip_chain_target_probs"],
                   {"torch": torch, "SamplingBatchInfo": object})["_glm53_hip_chain_target_probs"]
sampler = functions("srt/layers/sampler.py", ["top_k_top_p_min_p_sampling_from_probs_torch"],
                    {"torch": torch, "Optional": Optional})["top_k_top_p_min_p_sampling_from_probs_torch"]
torch.manual_seed(43)
width = 4
logits = torch.randn(12, 31) * 3
si = SimpleNamespace(temperatures=torch.tensor([[1.0], [0.7], [1.0]]), top_ks=torch.tensor([31, 5, 1]),
                     top_ps=torch.tensor([0.95, 0.9, 1.0]), min_ps=torch.tensor([0., 0.15, 0.]),
                     need_top_k_sampling=True, need_top_p_sampling=True, need_min_p_sampling=True)
p = target(logits, si, width)
plain = torch.softmax(logits / si.temperatures.repeat_interleave(width, 0), -1)
_, weights, order, _ = sampler(plain, si.top_ks.repeat_interleave(width), si.top_ps.repeat_interleave(width),
                              si.min_ps.repeat_interleave(width), True, None, None, return_filtered_probs=True)
reference = torch.zeros_like(p).scatter_(1, order.long(), weights / weights.sum(-1, keepdim=True))
torch.testing.assert_close(p, reference)
assert torch.all(p[8:].amax(-1) == 1)

sys.modules["sglang.srt.speculative.eagle_utils"] = SimpleNamespace(_glm53_hip_chain_target_probs=target)
dflash = functions("srt/speculative/dflash_utils.py", ["build_dflash_verify_target_probs"],
                   {"torch": torch, "Any": object, "Optional": Optional, "is_hip": lambda: True})
torch.testing.assert_close(dflash["build_dflash_verify_target_probs"](
    next_token_logits=logits, sampling_info=si, draft_token_num=width, bs=3), p.view(3, width, -1))

# Run the actual Triton kernel on CPU. Analytically integrate its first output
# over every draft choice and accept/reject branch; the result must be p.
file = Path(os.environ.get("GLM53_TEST_SAMPLING_SOURCE", root / "kernels/ops/speculative/reject_sampling.py"))
module_spec = importlib.util.spec_from_file_location("reject_sampling_cpu", file)
kernel = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(kernel)


def verify(candidates, probs, drafts, coins, finals):
    bs, w = candidates.shape
    retrieve = torch.arange(bs * w, dtype=torch.int32).view(bs, w)
    pred = torch.full((bs * w,), -1, dtype=torch.int32)
    index = torch.full((bs, w), -1, dtype=torch.int32)
    count = torch.empty(bs, dtype=torch.int32)
    kernel.chain_speculative_sampling_triton(pred, index, count, candidates, retrieve, retrieve, retrieve,
                                             coins, finals, probs, drafts, 1., 1., True)
    return pred.view(bs, w), index, count


for proposal in (torch.tensor([0., 1., 0., 0.]), torch.tensor([0.1, 0.4, 0.2, 0.3])):
    goal = torch.tensor([0.3, 0.1, 0.4, 0.2])
    cases, mass = [], []
    for x in range(4):
        qx = float(proposal[x])
        if not qx:
            continue
        accept = min(1., float(goal[x]) / qx)
        cases.append((x, accept / 2, 0.5, x))
        mass.append(qx * accept)
        if accept < 1:
            residual = (goal - proposal).clamp_min(0)
            residual /= residual.sum()
            prefix = 0.
            for token, probability in enumerate(residual.tolist()):
                if probability:
                    cases.append((x, (1 + accept) / 2, prefix + probability / 2, token))
                    mass.append(qx * (1 - accept) * probability)
                prefix += probability
    bs = len(cases)
    pred, _, _ = verify(torch.tensor([[0, x] for x, _, _, _ in cases]), goal.expand(bs, 2, 4).clone(),
                         proposal.expand(bs, 1, 4).clone(), torch.tensor([[c, 0] for _, c, _, _ in cases]),
                         torch.tensor([f for _, _, f, _ in cases]))
    expected = torch.tensor([token for _, _, _, token in cases])
    assert torch.equal(pred[:, 0], expected)
    emitted = torch.zeros(4).scatter_add_(0, pred[:, 0].long(), torch.tensor(mass))
    torch.testing.assert_close(emitted, goal)

# Vocab crosses a kernel tile; all drafts accepted and width-one bonus-only.
for w in (1, 2, 4, 8):
    vocab = 8193
    probs = torch.zeros(2, w, vocab)
    probs[:, :, 8192] = 1
    proposals = torch.zeros(2, max(w - 1, 1), vocab)
    proposals[:, :, 8192] = 1
    pred, idx, count = verify(torch.full((2, w), 8192, dtype=torch.int64), probs, proposals,
                              torch.full((2, w), 0.5), torch.full((2,), 0.9))
    assert torch.all(count == w - 1) and torch.all(pred == 8192)
    assert torch.equal(idx, torch.arange(2 * w, dtype=torch.int32).view(2, w))

# The parallel kernel sees deterministic exponential fixtures here, permitting
# exact checks of both acceptance indices and its residual/bonus winner. The
# exponential-max identity above gives the distribution for arbitrary weights.
for w in (1, 2, 4, 8):
    bs, vocab = 4, 8193
    p = torch.zeros(bs, w, vocab)
    p[:, :, 5] = .3
    p[:, :, 4097] = .7
    q = torch.zeros(bs, max(w - 1, 1), vocab)
    q[:, :, 5] = .8
    q[:, :, 4097] = .2
    candidates = torch.full((bs, w), 5, dtype=torch.int64)
    coins = torch.tensor([[.9] * w, [.1] * w, [.9] * w, [.1] * w])
    retrieve = torch.arange(bs * w, dtype=torch.int32).view(bs, w)
    pred = torch.full((bs * w,), -1, dtype=torch.int32)
    index = torch.full((bs, w), -1, dtype=torch.int32)
    count = torch.empty(bs, dtype=torch.int32)
    state = torch.get_rng_state()
    noise = torch.empty(bs, vocab).exponential_().clamp_min(torch.finfo(torch.float32).tiny)
    torch.set_rng_state(state)
    kernel.glm53_parallel_chain_speculative_sampling_triton(pred, index, count, candidates, retrieve, coins, p, q)
    expected_count = torch.tensor([0, w - 1, 0, w - 1], dtype=torch.int32)
    assert torch.equal(count, expected_count)
    for b, n in enumerate(expected_count.tolist()):
        weight = p[b, n] if n == w - 1 else (p[b, n] - q[b, n]).clamp_min(0)
        expected = int((weight / noise[b]).argmax())
        assert int(pred[b * w + n]) == expected
        assert torch.equal(index[b, :n + 1], retrieve[b, :n + 1])
        assert torch.all(index[b, n + 1:] == -1)
        assert torch.all(pred[b * w:b * w + n] == 5)

# Basic statistical check of the exponential-max identity with a broad support,
# separately from the deterministic tiled-kernel fixtures above.
torch.manual_seed(45)
weight = torch.tensor([.3, .1, .4, .2])
draws = (weight / torch.empty(100000, 4).exponential_()).argmax(-1)
observed = torch.bincount(draws, minlength=4).float() / len(draws)
assert torch.max(torch.abs(observed - weight)) < .006, observed

# Seeded requests preserve the original verifier and its given coins.
os.environ["SGLANG_GLM53_HIP_PARALLEL_SPEC_SAMPLING"] = "1"
pred, index, count = verify(torch.tensor([[0, 1]]), torch.tensor([[[.3, .1, .4, .2]]]).expand(1, 2, 4).clone(),
                            torch.tensor([[[0., 1., 0., 0.]]]), torch.tensor([[.9, .5]]), torch.tensor([.5]))
# verify() invokes the optional unseeded dispatcher, so exercise the explicit
# seeded fallback separately with known CDF output (residual token 2).
kernel.chain_speculative_sampling_triton(pred.flatten(), index, count, torch.tensor([[0, 1]]),
    torch.tensor([[0, 1]], dtype=torch.int32), None, None, torch.tensor([[.9, .5]]), torch.tensor([.5]),
    torch.tensor([[[.3, .1, .4, .2]]]).expand(1, 2, 4).clone(), torch.tensor([[[0., 1., 0., 0.]]]),
    1., 1., True, sampling_seed=torch.tensor([7]))
assert int(pred[0, 0]) == 2 and int(count[0]) == 0

# Returned logprobs use the target logits at the committed row and token, never
# the proposal q or the residual draw law. SGLang reports temperature-adjusted,
# unfiltered model logprobs; target filtering changes the sampling law only.
logprob = functions("srt/layers/logprob_processor.py", ["compute_spec_logprobs"],
    {"torch": torch, "ScheduleBatch": object, "LogitsProcessorOutput": object,
     "Optional": Optional,
     "envs": SimpleNamespace(SGLANG_RETURN_ORIGINAL_LOGPROB=SimpleNamespace(get=lambda: False))})
logits = torch.randn(8, 11)
tokens = torch.tensor([7, 2, 5, 1, 0, 3, 9, 4])
batch = SimpleNamespace(seq_lens=[1, 2], top_logprobs_nums=None, token_ids_logprobs=None,
    sampling_info=SimpleNamespace(is_all_greedy=False, temperatures=torch.tensor([[1.], [.7]])))
idx = torch.tensor([[0, 1, -1, -1], [4, 5, 6, 7]])
out = SimpleNamespace(next_token_logits=logits)
logprob["compute_spec_logprobs"](batch, out, tokens, accept_index=idx)
expected = torch.log_softmax(logits / batch.sampling_info.temperatures.repeat_interleave(4, 0), -1)
torch.testing.assert_close(out.next_token_logprobs[0, :2], expected[:2].gather(1, tokens[:2, None]).flatten())
torch.testing.assert_close(out.next_token_logprobs[1], expected[4:].gather(1, tokens[4:, None]).flatten())
logprob["compute_spec_logprobs"](batch, out, tokens, chain_stride=4)
torch.testing.assert_close(out.next_token_logprobs, expected.gather(1, tokens[:, None]).view(2, 4))
print("PASS: capture devices, mixed greedy rows, HIP TP sync, NEXTN/DFlash filters, exact rejection mass, widths 1/2/4/8, parallel tiles, seed fallback, committed target logprobs")
