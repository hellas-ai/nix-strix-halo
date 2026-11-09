"""GPU qualification/microbenchmark; run only under campaign cluster.lock.

The owner must restore production and verify chat inside the same lock. This
script only allocates tensors and benchmarks kernels; it never changes units.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--draws", type=int, default=100000)
parser.add_argument("--iters", type=int, default=100)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(45)
device = "cuda"


def buffers(candidates):
    bs, width = candidates.shape
    return (
        torch.zeros(bs * width, dtype=torch.int32, device=device),
        torch.full((bs, width), -1, dtype=torch.int32, device=device),
        torch.empty(bs, dtype=torch.int32, device=device),
        torch.arange(bs * width, dtype=torch.int32, device=device).view(bs, width),
    )


def run(out, candidates, coins, finals, p, q, parallel, seed=None):
    os.environ["SGLANG_GLM53_HIP_PARALLEL_SPEC_SAMPLING"] = str(int(parallel))
    pred, index, count, retrieve = out
    chain_speculative_sampling_triton(
        pred, index, count, candidates, retrieve, None, None,
        coins, finals, p, q, 1., 1., True, sampling_seed=seed,
    )


report = {"torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(),
          "draws": args.draws, "statistical": [], "timing": []}
goal = torch.tensor([.3, .1, .4, .2], device=device)
for proposal in ([0., 1., 0., 0.], [.1, .4, .2, .3], [.3, .1, .4, .2]):
    q0 = torch.tensor(proposal, device=device)
    chosen = torch.multinomial(q0.expand(args.draws, 4), 1)
    candidates = torch.cat((torch.zeros_like(chosen), chosen), 1)
    p = goal.expand(args.draws, 2, 4).contiguous()
    q = q0.expand(args.draws, 1, 4).contiguous()
    coins = torch.rand(args.draws, 2, device=device)
    finals = torch.rand(args.draws, device=device)
    expected_accept = torch.minimum(goal, q0).sum().item()
    counts = []
    for parallel in (False, True):
        out = buffers(candidates)
        run(out, candidates, coins, finals, p, q, parallel)
        torch.cuda.synchronize()
        pred, _, count, _ = out
        counts.append(count.clone())
        emitted = torch.bincount(pred.view(-1, 2)[:, 0].long(), minlength=4).float() / args.draws
        observed = emitted.cpu().tolist()
        tolerance = 6 * torch.sqrt(goal * (1 - goal) / args.draws) + 2 / args.draws
        assert torch.all((emitted - goal).abs() <= tolerance), (parallel, proposal, observed)
        measured_accept = count.float().mean().item()
        assert abs(measured_accept - expected_accept) <= 6 * ((expected_accept * (1 - expected_accept) / args.draws) ** .5) + 2 / args.draws
        report["statistical"].append({"parallel": parallel, "q": proposal, "target": goal.cpu().tolist(),
                                      "emitted": observed, "acceptance": measured_accept})
    assert torch.equal(*counts)

for bs in (1, 2, 4):
    for width in (1, 2, 4, 8):
        vocab = 154880
        p0 = torch.softmax(torch.randn(bs, width, vocab, device=device) * 3, -1)
        q = p0[:, :max(width - 1, 1)].contiguous()
        if width > 1:
            q = .9 * q + .1 * torch.softmax(torch.randn_like(q) * 3, -1)
        proposals = torch.multinomial(q.reshape(-1, vocab), 1).view(bs, -1)
        candidates = torch.zeros(bs, width, dtype=torch.int64, device=device)
        if width > 1:
            candidates[:, 1:] = proposals
        coins = torch.rand(bs, width, device=device)
        finals = torch.rand(bs, device=device)
        out = buffers(candidates)
        measurements, counts = {}, []
        for parallel in (False, True):
            for _ in range(5):
                run(out, candidates, coins, finals, p0, q, parallel)
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iters):
                run(out, candidates, coins, finals, p0, q, parallel)
            end.record()
            end.synchronize()
            measurements["parallel_ms" if parallel else "classic_ms"] = begin.elapsed_time(end) / args.iters
            counts.append(out[2].clone())
        assert torch.equal(*counts)
        result = {"bs": bs, "width": width, **measurements, "accepted_drafts": counts[-1].cpu().tolist()}
        result["speedup"] = result["classic_ms"] / result["parallel_ms"]
        report["timing"].append(result)
        print(json.dumps(result), flush=True)

# A seeded request must keep the original CDF sampling trace even with the
# opt-in enabled. This covers the actual dispatch and compiled baseline.
out = buffers(candidates)
run(out, candidates, coins, finals, p0, q, False)
expected = tuple(t.clone() for t in out[:3])
run(out, candidates, coins, finals, p0, q, True, seed=torch.tensor([7], device=device))
assert all(torch.equal(a, b) for a, b in zip(expected, out))
report["seed_fallback"] = "PASS"
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print("PASS: distribution, acceptance, seeded fallback; timing saved to", args.output)
