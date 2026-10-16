"""Compile-only helper: build a Triton kernel for gfx1151 without a GPU, mimicking runtime specialisation.

spec: dict arg_name -> one of
   ('ptr', dtype)          pointer, 16B aligned, 32-bit buffer range     (tt.divisibility=16, tt.pointer_range=32)
   ('i32',)                 plain int
   ('i32d',)                int divisible by 16
   ('const', value)         constexpr / specialised-to-1 int
"""
import re
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

def compile_gfx1151(jitfn, spec, options=None, arch="gfx1151"):
    fn = jitfn
    names = list(fn.arg_names)
    sig, consts, attrs = {}, {}, {}
    for i, n in enumerate(names):
        if n not in spec:
            raise KeyError(f"no spec for arg {n}")
        s = spec[n]
        kind = s[0]
        if kind == 'ptr':
            sig[n] = "*" + s[1]
            attrs[(i,)] = [["tt.divisibility", 16], ["tt.pointer_range", 32]]
        elif kind == 'i32':
            sig[n] = "i32"
        elif kind == 'i32d':
            sig[n] = "i32"
            attrs[(i,)] = [["tt.divisibility", 16]]
        elif kind in ('fp32', 'i64'):
            sig[n] = kind
        elif kind == 'const':
            sig[n] = "constexpr"
            consts[n] = s[1]
        else:
            raise ValueError(kind)
    src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
    return triton.compile(src, target=GPUTarget("hip", arch, 32), options=dict(options or {}))

def isa_stats(ck):
    asm = ck.asm['amdgcn']
    out = {}
    for m in re.finditer(r'; (NumVgprs|ScratchSize|Occupancy|NumSgprs|LDSByteSize|VGPRBlocks|SGPRBlocks): (\d+)', asm):
        out[m.group(1)] = int(m.group(2))
    return out

def loop_hist(asm, label=None):
    """Histogram of the loop body containing the most WMMA/VALU work: (hist, label, n_instrs) or None."""
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
        hist, n = {}, 0
        for ln in lines[i + 1:end + 1]:
            t = ln.strip()
            if not t or t.startswith(('.', ';', '//', '@')): continue
            op = t.split()[0]
            if op.endswith(':'): continue
            hist[op] = hist.get(op, 0) + 1; n += 1
        w = sum(c for op, c in hist.items() if op.startswith('v_wmma'))
        key = (w, n)
        if w > 0 and (best is None or key < best[0] or best[0][0] == 0):
            pass
        # prefer the innermost loop with WMMA: smallest n among loops with wmma>0
        if w > 0 and (best is None or n < best[3]):
            best = (key, hist, lab, n)
    if best is None:
        return None
    return best[1], best[2], best[3]

def classify(hist):
    v = sum(c for op, c in hist.items() if op.startswith('v_') and not op.startswith('v_wmma'))
    w = sum(c for op, c in hist.items() if op.startswith('v_wmma'))
    s = sum(c for op, c in hist.items() if op.startswith('s_'))
    d = sum(c for op, c in hist.items() if op.startswith('ds_'))
    g = sum(c for op, c in hist.items() if op.startswith(('buffer_', 'global_', 'flat_')))
    return dict(valu=v, wmma=w, salu_misc=s, lds=d, vmem=g)
