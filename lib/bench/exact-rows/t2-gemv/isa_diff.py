"""Compare two ISA dump dirs instruction-for-instruction (ignores .loc/.file/DWARF/comments, kernel metadata kept for VGPR counts)."""
import re, sys, os, glob
def stream(path):
    out = []
    for l in open(path):
        t = l.split(';')[0].rstrip()
        if not t.strip():
            continue
        if re.match(r'^\s+\.(loc|file|cfi|section|p2align|type|size|globl|amdgcn|amdhsa|end_amdhsa|set|weak|long|quad|short|byte|ascii|asciz|text|data|rodata|ident|protected|addrsig|offset|uleb|sleb)', t) and not t.strip().startswith('.set'):
            continue
        if t.lstrip().startswith(('.', '-')) and not t.startswith('.LBB'):
            continue
        out.append(re.sub(r'\s+', ' ', t.strip()))
    return out
a, b = sys.argv[1], sys.argv[2]
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a, '*.amdgcn')))
bad = 0
for n in names:
    pb = os.path.join(b, n)
    if not os.path.exists(pb):
        continue
    sa, sb = stream(os.path.join(a, n)), stream(pb)
    same = sa == sb
    bad += (not same)
    print(f"{n}: {'IDENTICAL' if same else 'DIFFERS'} ({len(sa)} vs {len(sb)} instruction/label lines)")
    if not same:
        import difflib
        for d in list(difflib.unified_diff(sa, sb, lineterm='', n=0))[:12]:
            print('   ', d)
print('DIFFERING FILES:', bad)
