"""Shared GPU benchmark helpers (graph timing, kernel-duration profiling, baseline stage wrappers).

RUN ONLY AFTER the sentinel K2-moe parent dir ../GPU-FREE-FOR-MICROBENCH exists; strix-2 only; <8 GB; do not take /tmp/ds41-gpu.lock.
"""
import os, sys, time, json, statistics
import torch

SENTINEL = "/mnt/Home/src/nixos-config/artifacts/claude-handover-20261002/GPU-FREE-FOR-MICROBENCH"
STREAM_GBPS = 242.253

def require_sentinel():
    if not os.path.exists(SENTINEL):
        raise SystemExit("GPU-FREE-FOR-MICROBENCH sentinel absent: refusing to touch the GPU")

def check_budget(max_gb=8.0):
    torch.cuda.set_per_process_memory_fraction(min(0.99, max_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory))

def time_graph(launch, n_iter, reps=5, warmup=2):
    """launch(i) must enqueue iteration i (i in [0,n_iter)). Returns list of per-iteration microseconds over reps (graph replay,
    includes inter-kernel gaps)."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(min(n_iter, 3)): launch(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(n_iter): launch(i)
    torch.cuda.synchronize()
    for _ in range(warmup): g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / n_iter)
    return out, g

def kernel_durations(graph, n_iter, name_filter=None):
    """Replay `graph` under torch.profiler and return {kernel_name: (count, mean_us)} using device-side durations."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay(); torch.cuda.synchronize()
    res = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            nm = ev.name
            if name_filter and not name_filter(nm): continue
            dt = getattr(ev, 'device_time', None)
            if dt is None: dt = ev.cuda_time
            c, t = res.get(nm, (0, 0.0)); res[nm] = (c + 1, t + dt)  # microseconds
    return {k: (c, t / c) for k, (c, t) in res.items()}

def unique_expert_bytes(ids, per_expert_bytes):
    return per_expert_bytes * int(torch.unique(ids).numel())

GATE_UP_BYTES = 1152 * 2560 + 1152 * 160     # per expert, packed + scales
DOWN_BYTES = 5120 * 288 + 5120 * 18

def gbps(bytes_, us):
    return bytes_ / (us * 1e-6) / 1e9
