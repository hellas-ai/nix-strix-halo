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

