"""GPU microbench + bitwise check of the live-row mask of the MXFP4 decode chain (rank-0 TP4 geometry, 384 experts).

Run ONLY through session_t1.sh (sentinel + idle checks); one GPU, <8 GB.

For every (M, live) case (M = padded graph rows, live = real rows; dead rows routed randomly as the router would route
the zero-hidden padded rows):
  1. bitwise: masked chain (live-row tensor given) vs unmasked chain, rows < live of every stage identical; dead rows zero.
  2. graph safety: ONE captured graph of the masked chain, replayed with the device live-row value changed between replays
     (M, M-1, ..., 1, M): outputs follow the device value; the value is not baked into the capture.
  3. timing (CUDA-graph replay, distinct weights/routes/activations per iteration, per layer incl. launch gaps):
        padded   = unmasked chain at M rows (what the C3 -> C4 padding costs today)
        masked   = masked chain, `live` live rows of M
        unpadded = unmasked chain at `live` rows (the shape the batch really has; what exact buckets would run)
     with distinct experts, effective GB/s over unique expert bytes and % of 242.25 GB/s.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
os.environ.setdefault("SGLANG_USE_AITER", "0")
import torch

import benchlib as B
import fixtures as F_

STREAM_GBPS = B.STREAM_GBPS


def moe_bytes(ids):
    return int(torch.unique(ids[ids >= 0]).numel()) * (B.GATE_UP_BYTES + B.DOWN_BYTES)


def bits(t):
    return t.contiguous().view(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1:1,2:1,4:1,4:2,4:3,8:5,8:6,8:7,8:4,8:3")
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dead-routing", choices=["random", "fixed"], default="random",
                    help="random: dead rows route like live ones; fixed: every dead row uses the same 6 experts at every layer/step")
    ap.add_argument("--policy", choices=["live", "wmma"], default="live")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    B.require_sentinel()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1151"
    B.check_budget(8.0)
    os.environ["SGLANG_DSV41_MXFP4_GATE_UP_ROWS"] = a.policy
    import sglang.kernels.ops.moe.dsv41_mxfp4_decode as P

    dev = "cuda"
    w13, w2, s13, s2 = F_.synth_layer(dev)
    s13e, s2e = s13.view(torch.float8_e8m0fnu), s2.view(torch.float8_e8m0fnu)

    def chain(x, w, ids, live_t):
        return P.decode_routed_experts(x, w13, w2, s13e, s2e, w, ids, 1.5, 10, live_rows=live_t)

    rows_out, failures = [], []
    for case in a.cases.split(","):
        M, live = (int(v) for v in case.split(":"))
        assert 1 <= live <= M <= 16
        n = a.iters
        routes = [F_.make_routes(M, dev, seed=700 + i, routing="random") for i in range(n)]
        ids_l = [r[0] for r in routes]
        w_l = [r[1] for r in routes]
        if a.dead_routing == "fixed":
            fixed = F_.make_routes(M, dev, seed=999, routing="random")[0][live:]
            for ids in ids_l:
                ids[live:] = fixed
        xs = [F_.realistic_activations(M, dev, seed=21 + i) for i in range(n)]
        live_t = torch.full((1,), live, dtype=torch.int32, device=dev)
        full_t = torch.full((1,), M, dtype=torch.int32, device=dev)

        # 1. bitwise
        mism = dead_nonzero = 0
        for i in range(n):
            ref = chain(xs[i], w_l[i], ids_l[i], None)
            got = chain(xs[i], w_l[i], ids_l[i], live_t)
            mism += int((bits(ref[:live]) != bits(got[:live])).sum())
            dead_nonzero += int((got[live:] != 0).sum())
            # the all-live tensor must reproduce the unmasked chain exactly (mask is the identity when nothing is padded)
            same = chain(xs[i], w_l[i], ids_l[i], full_t)
            mism += int((bits(ref) != bits(same)).sum())
        torch.cuda.synchronize()

        # 2. graph safety: capture once with the device value = M, replay with other values
        graph_bad = 0
        x0, w0, i0 = xs[0], w_l[0], ids_l[0]
        gt = torch.full((1,), M, dtype=torch.int32, device=dev)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            chain(x0, w0, i0, gt)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            gout = chain(x0, w0, i0, gt)
        ref_full = chain(x0, w0, i0, None)
        for value in list(range(M, 0, -1)) + [M]:
            gt.fill_(value)
            g.replay()
            torch.cuda.synchronize()
            graph_bad += int((bits(gout[:value]) != bits(ref_full[:value])).sum())
            graph_bad += int((gout[value:] != 0).sum())

        # 3. timing
        def padded(i):
            chain(xs[i], w_l[i], ids_l[i], None)

        def masked(i):
            chain(xs[i], w_l[i], ids_l[i], live_t)

        xs_l = [x[:live].contiguous() for x in xs]
        ids_live = [t[:live].contiguous() for t in ids_l]
        w_live = [t[:live].contiguous() for t in w_l]

        def unpadded(i):
            chain(xs_l[i], w_live[i], ids_live[i], None)

        t_pad, _ = B.time_graph(padded, n, reps=a.reps)
        t_msk, _ = B.time_graph(masked, n, reps=a.reps)
        t_unp, _ = B.time_graph(unpadded, n, reps=a.reps)
        med = lambda v: sorted(v)[len(v) // 2]
        uniq_pad = sum(moe_bytes(i) for i in ids_l) / n / (B.GATE_UP_BYTES + B.DOWN_BYTES)
        uniq_live = sum(moe_bytes(i[:live]) for i in ids_l) / n / (B.GATE_UP_BYTES + B.DOWN_BYTES)
        r = dict(M=M, live=live, dead_routing=a.dead_routing, policy=a.policy,
                 distinct_experts_padded=uniq_pad, distinct_experts_live=uniq_live,
                 padded_us=med(t_pad), masked_us=med(t_msk), unpadded_us=med(t_unp),
                 masked_saving_us_per_layer=med(t_pad) - med(t_msk),
                 masked_saving_ms_per_step_est=(med(t_pad) - med(t_msk)) * 40 / 1000.0,
                 masked_gbps=uniq_live * (B.GATE_UP_BYTES + B.DOWN_BYTES) / (med(t_msk) * 1e-6) / 1e9,
                 padded_gbps=uniq_pad * (B.GATE_UP_BYTES + B.DOWN_BYTES) / (med(t_pad) * 1e-6) / 1e9,
                 unpadded_gbps=uniq_live * (B.GATE_UP_BYTES + B.DOWN_BYTES) / (med(t_unp) * 1e-6) / 1e9,
                 mismatched_fp32_live_rows=mism, dead_nonzero=dead_nonzero, graph_replay_bad=graph_bad)
        for k in ("masked_gbps", "padded_gbps", "unpadded_gbps"):
            r[k.replace("gbps", "pct_of_242")] = 100.0 * r[k] / STREAM_GBPS
        print(json.dumps(r), flush=True)
        rows_out.append(r)
        if mism or dead_nonzero or graph_bad:
            failures.append(case)
    if a.out:
        json.dump(rows_out, open(a.out, "w"), indent=1)
    if failures:
        print("FAILED cases:", failures)
        sys.exit(1)
    print("all bitwise / graph-safety checks passed")


if __name__ == "__main__":
    main()
