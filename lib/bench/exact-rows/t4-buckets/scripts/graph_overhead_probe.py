#!/usr/bin/env python3
"""What an extra decode graph costs on gfx1151, measured without the model.  Run ONLY through session_t4.sh.

A real DS4.1 decode graph is ~4.5-5.7k kernel nodes (K1 report: 5.4-5.7k before K1/K2).  Its memory has two parts:
  (a) per-graph executable bookkeeping (kernel nodes + arguments, allocated by the HIP runtime, outside torch's allocator), and
  (b) the activation pool, which graphs captured one after another share (the runner captures largest bucket first).
(a) is a property of the node count and argument sizes, not of the weights, so it is measured here with synthetic chains of
Triton and ATen kernels on tiny tensors.  (b) needs the real model: it is bounded by the largest bucket's activations and is
read from the 'Capture ... graph end ... mem usage' journal lines (graph_logs.py capture).

Measures, per number of resident graphs G:  free-memory delta per capture (HIP mem_get_info, MiB), torch reserved delta, capture
seconds, replay microseconds (first / last graph) - once with one shared pool and once with a private pool per graph.
Budget: < 8 GB, one GPU, no collectives, no model.  Refuses to start without the sentinel.
"""
import argparse
import json
import os
import statistics
import sys
import time

SENTINEL = "/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH"

try:
    import torch  # before triton: libtriton's HIP constructor needs torch's rocprofiler tool library loaded first
    import triton
    import triton.language as tl

    @triton.jit
    def node(X, Y, P0, P1, P2, P3, n0, n1, n2, n3, n4, n5, n6, n7, BLOCK: tl.constexpr):
        # Twelve runtime arguments: the scale of the real native kernels' argument lists (GEMV/HC/K1).
        offs = tl.arange(0, BLOCK)
        tl.store(Y + offs, tl.load(X + offs) + 1.0)
except ImportError:  # --selftest on a host without torch/triton
    torch = node = None


def summarize(deltas_mib):
    """Mean marginal cost of graphs 2..G (the first graph also pays one-time runtime setup)."""
    rest = deltas_mib[1:]
    return {
        "first_mib": deltas_mib[0] if deltas_mib else None,
        "marginal_mean_mib": statistics.fmean(rest) if rest else None,
        "marginal_max_mib": max(rest) if rest else None,
        "total_mib": sum(deltas_mib),
    }


def selftest():
    s = summarize([12.0, 2.0, 2.0, 4.0])
    assert s["first_mib"] == 12.0 and abs(s["marginal_mean_mib"] - 8 / 3) < 1e-9 and s["marginal_max_mib"] == 4.0
    assert s["total_mib"] == 20.0
    assert summarize([5.0])["marginal_mean_mib"] is None
    print("selftest ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, default=5000, help="kernel nodes per graph")
    parser.add_argument("--graphs", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--out", default=None, help="write JSON here")
    parser.add_argument("--selftest", action="store_true", help="check the reporting arithmetic only (no GPU)")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not os.path.exists(SENTINEL):
        raise SystemExit("GPU-FREE-FOR-MICROBENCH sentinel absent: refusing to touch the GPU")

    torch.cuda.set_per_process_memory_fraction(min(0.99, 8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
    dev = torch.device("cuda")
    a = torch.zeros(1024, device=dev, dtype=torch.float32)
    b = torch.zeros(1024, device=dev, dtype=torch.float32)
    p = [torch.zeros(1, device=dev, dtype=torch.float32) for _ in range(4)]

    def chain(nodes):
        for i in range(nodes):
            if i % 3 == 0:
                node[(1,)](a, b, p[0], p[1], p[2], p[3], 1, 2, 3, 4, 5, 6, 7, 8, BLOCK=1024, num_warps=1)
            elif i % 3 == 1:
                b.add_(1.0)
            else:
                a.mul_(1.0)

    def free_mib():
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info()[0] / 2**20

    def run(num_graphs, shared):
        chain(8)
        torch.cuda.synchronize()          # compile + warm every kernel before capture
        graphs, deltas, reserved, capture_s = [], [], [], []
        pool = torch.cuda.graph_pool_handle() if shared else None
        for _ in range(num_graphs):
            before, res0 = free_mib(), torch.cuda.memory_reserved() / 2**20
            graph = torch.cuda.CUDAGraph()
            t0 = time.perf_counter()
            with torch.cuda.graph(graph, pool=pool if shared else torch.cuda.graph_pool_handle()):
                chain(args.nodes)
            torch.cuda.synchronize()
            capture_s.append(time.perf_counter() - t0)
            deltas.append(before - free_mib())
            reserved.append(torch.cuda.memory_reserved() / 2**20 - res0)
            graphs.append(graph)

        def replay_us(graph):
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(args.replays):
                graph.replay()
            ev1.record()
            ev1.synchronize()
            return ev0.elapsed_time(ev1) * 1000.0 / args.replays

        result = {
            "graphs": num_graphs, "shared_pool": shared, "nodes": args.nodes,
            "free_delta_mib": deltas, "reserved_delta_mib": reserved, "capture_s": capture_s,
            "free_delta": summarize(deltas), "reserved_delta": summarize(reserved),
            "replay_us_first": replay_us(graphs[0]), "replay_us_last": replay_us(graphs[-1]),
        }
        del graphs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return result

    results = []
    for shared in (True, False):
        for g in args.graphs:
            r = run(g, shared)
            results.append(r)
            print(json.dumps({k: r[k] for k in ("graphs", "shared_pool", "nodes", "free_delta", "reserved_delta",
                                                  "replay_us_first", "replay_us_last")}), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    sys.exit(main())
