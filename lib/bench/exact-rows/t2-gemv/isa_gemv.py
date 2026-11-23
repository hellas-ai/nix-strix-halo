"""Compile-only gfx1151 ISA probe of the native FP8 GEMV kernels (no GPU).

usage: run_cpu.sh isa_gemv.py [--module PATH] [--out DIR] [--tokens 1,2,3,...] [--dump]
  --module  deepseek_v41_gemv.py to load (default: the overlay's); use it to compare a baseline tree with the patched one
  --dump    write every .amdgcn next to the table (for diffing baseline vs new ISA)
Prints one row per (shape, tokens): VGPR / SGPR / scratch bytes / spills / LDS / occupancy / loop instruction counts.
"""
import argparse, hashlib, importlib.util, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcompile import compile_gfx1151, isa_stats

# (name, N, K, BLOCK_K) of the seven target projections + three draft ones (BLOCK_K as dispatched by deepseek_v41_gemv.py / fp8_utils.py).
SHAPES = (
    ("engram_wkv", 25600, 6144, 512), ("wo_b", 5120, 2048, 512), ("wqkv_a", 1792, 5120, 512), ("wq_b", 8192, 1280, 256),
    ("shared_gate_up", 1152, 5120, 512), ("shared_down", 5120, 576, 64), ("idx_wq_b", 4096, 1280, 256),
    # DSpark draft stages' unflagged linears (explicit table of dsv41_native_rows.PROJECTION_BLOCK_K)
    ("draft_wq_a", 1280, 5120, 512), ("draft_wkv", 512, 5120, 512), ("draft_main_proj", 5120, 15360, 512),
)

C1_ROWS = 4

def load(path):
    spec = importlib.util.spec_from_file_location("gemv_probe_" + hashlib.md5(path.encode()).hexdigest()[:6], path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def specs(kernel, n, k, tokens, rows, block_k):
    base = {"Q": ('ptr', 'fp8e4nv'), "QS": ('ptr', 'fp32'), "W": ('ptr', 'fp8e4nv'), "WS": ('ptr', 'fp32'), "Y": ('ptr', 'bf16'),
            "N": ('const', n), "K": ('const', k), "ROWS": ('const', rows), "BLOCK_K": ('const', block_k)}
    if "TOKENS" in kernel.arg_names:
        base["TOKENS"] = ('const', tokens)
    return base

def loop_stats(asm):
    """Instruction histogram of the largest innermost loop (the K loop)."""
    lines = asm.splitlines()
    best = None
    for i, l in enumerate(lines):
        if 'Loop Header' not in l:
            continue
        lab = l.split(':')[0]
        end = None
        for j in range(len(lines) - 1, i, -1):
            if re.search(r's_cbranch\w*\s+' + re.escape(lab) + r'\b', lines[j]):
                end = j; break
        if end is None:
            continue
        hist = {}
        for ln in lines[i + 1:end + 1]:
            t = ln.strip()
            if not t or t.startswith(('.', ';', '//', '@')): continue
            op = t.split()[0]
            if op.endswith(':'): continue
            hist[op] = hist.get(op, 0) + 1
        n = sum(hist.values())
        if best is None or n > best[1]:
            best = (hist, n)
    if best is None:
        return dict(loop_instr=0, valu=0, salu=0, vmem=0, lds=0)
    h, n = best
    valu = sum(c for op, c in h.items() if op.startswith('v_'))
    return dict(loop_instr=n, valu=valu, salu=sum(c for op, c in h.items() if op.startswith('s_')),
                vmem=sum(c for op, c in h.items() if op.startswith(('global_', 'buffer_', 'flat_'))),
                lds=sum(c for op, c in h.items() if op.startswith('ds_')))

def kernel_for(mod, tokens):
    if tokens == 1:
        return mod._native_fp8_gemv_c1, C1_ROWS, "c1"
    if tokens == 2:
        return mod._native_c2_gemv, 8, "c2"
    return mod._native_c4_c8_gemv, 8, "c4c8"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--shapes", default=None, help="comma list of shape names")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--bk", type=int, default=None, help="override BLOCK_K of every selected shape (exploration only)")
    ap.add_argument("--c1-rows", type=int, default=4, help="output rows per program of the 1-token kernel")
    args = ap.parse_args()
    global C1_ROWS
    C1_ROWS = args.c1_rows
    if args.module is None:
        import sglang.kernels.ops.gemm.deepseek_v41_gemv as mod
    else:
        mod = load(args.module)
    toks = [int(t) for t in args.tokens.split(",")]
    want = set(args.shapes.split(",")) if args.shapes else None
    rows_out = []
    for name, n, k, bk in SHAPES:
        if want and name not in want:
            continue
        if args.bk:
            bk = args.bk
        for t in toks:
            kern, out_rows, tag = kernel_for(mod, t)
            if tag == "c4c8" and t < 3:
                continue
            try:
                ck = compile_gfx1151(kern, specs(kern, n, k, t, out_rows, bk),
                                     dict(num_warps=args.warps, num_stages=1, enable_fp_fusion=False))
            except Exception as e:  # report, keep going
                rows_out.append(dict(shape=name, tokens=t, kernel=tag, error=str(e)[:200])); continue
            asm = ck.asm['amdgcn']
            st = isa_stats(ck)
            spills = len(re.findall(r'scratch_store|buffer_store.*offen|; Spill', asm))
            rec = dict(shape=name, N=n, K=k, BLOCK_K=bk, tokens=t, kernel=tag, out_rows=out_rows,
                       vgpr=st.get('NumVgprs'), sgpr=(re.search(r'\.sgpr_count:\s+(\d+)', asm) or [None, None])[1], scratch=st.get('ScratchSize'),
                       lds=st.get('LDSByteSize'), occ=st.get('Occupancy'), lines=len(asm.splitlines()),
                       sha=hashlib.sha1(re.sub(r'^\s*(\.loc|\.file|;.*)$', '', asm, flags=re.M).encode()).hexdigest()[:12])
            rec.update(loop_stats(asm))
            rows_out.append(rec)
            if args.dump and args.out:
                os.makedirs(args.out, exist_ok=True)
                open(os.path.join(args.out, f"{name}_t{t}_{tag}.amdgcn"), "w").write(asm)
    print("| shape | tokens | kernel | BLOCK_K | VGPR | SGPR | scratch B | LDS B | occ | loop instr | loop VALU | loop VMEM | asm sha |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_out:
        if 'error' in r:
            print(f"| {r['shape']} | {r['tokens']} | {r['kernel']} | ERROR {r['error']} |"); continue
        print(f"| {r['shape']} | {r['tokens']} | {r['kernel']} | {r['BLOCK_K']} | {r['vgpr']} | {r['sgpr']} | {r['scratch']} | {r['lds']} | {r['occ']} | {r['loop_instr']} | {r['valu']} | {r['vmem']} | {r['sha']} |")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json.dump(rows_out, open(os.path.join(args.out, "isa.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
