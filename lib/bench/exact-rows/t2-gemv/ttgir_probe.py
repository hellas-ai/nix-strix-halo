"""Reduction-structure check from the TTGIR: every tt.reduce of every token, in every TOKENS instantiation, has the same operand
shape and layout encoding (which fixes the in-thread / cross-lane reduction order of the AMD lowering).
usage: run_cpu.sh ttgir_probe.py [shape ...]"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sglang.kernels.ops.gemm.deepseek_v41_gemv as mod
from isa_gemv import specs, kernel_for, SHAPES
from tcompile import compile_gfx1151

def reductions(tt):
    layouts = dict(re.findall(r'^(#blocked\d*) = (#ttg\.blocked<\{.*?\}>)', tt, flags=re.M))
    out = []
    pos = 0
    while True:
        m = re.search(r'"tt\.reduce"\(%[\w]+\) <\{axis = (\d+) : i32\}> \(\{', tt[pos:])
        if not m:
            break
        start = pos + m.end()
        end = tt.index('}) : (', start)
        sig = re.match(r'\}\) : \((tensor<[^)]*>)\) -> ', tt[end:]).group(1)
        resolved = re.sub(r'#blocked\d*', lambda g: layouts[g.group(0)], sig)
        out.append((m.group(1), resolved))
        pos = end
    return out

want = set(sys.argv[1:])
allsig = {}
for name, n, k, bk in SHAPES:
    if want and name not in want:
        continue
    for t in range(1, 9):
        kern, rows, tag = kernel_for(mod, t)
        ck = compile_gfx1151(kern, specs(kern, n, k, t, rows, bk), dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
        red = reductions(ck.asm['ttgir'])
        sigs = sorted(set(red))
        allsig[(name, t)] = (len(red), sigs)
    ref = allsig[(name, 8)][1]
    for t in range(1, 9):
        cnt, sigs = allsig[(name, t)]
        same8 = (sigs == ref)
        print(f"{name:15} TOKENS={t} {kernel_for(mod, t)[2]:5}: {cnt:2d} reduces (= 2 per token: {cnt == 2 * t}); distinct signatures {len(sigs)}; identical to TOKENS=8's: {same8}")
    if name == SHAPES[0][0] or want:
        for ax, sg in ref:
            print(f"      axis={ax} {sg}")
