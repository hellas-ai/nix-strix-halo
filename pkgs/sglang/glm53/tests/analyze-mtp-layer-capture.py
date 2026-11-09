"""Compare the first logical token of bounded eager target captures."""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
assert not args.out.exists()
sites = ("entry", "attn_input", "kda_projection", "kda_core", "dsa_projection", "dsa_prepared", "dsa_core", "dsa_v_absorb", "attn_output", "mlp_input", "mlp_output", "exit")
rows = []
for rank in range(4):
    for layer in range(45):
        for site in sites:
            if site.startswith("dsa_") and layer % 4 != 3:
                continue
            name = f"rank{rank}-layer{layer:02d}-{site}.pt"
            paths = [args.root / arm / "capture" / name for arm in ("plain-a", "mtp", "plain-b")]
            if not any(path.exists() for path in paths):
                assert site.startswith(("kda_", "dsa_"))
                continue
            a, candidate, b = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
            for value in (candidate, b):
                assert int(a["positions"].flatten()[0]) == int(value["positions"].flatten()[0])
                assert int(a["input_ids"].flatten()[0]) == int(value["input_ids"].flatten()[0])
                assert a["tensors"].keys() == value["tensors"].keys()
            for tensor_name, tensor in a["tensors"].items():
                if tensor is None:
                    assert candidate["tensors"][tensor_name] is None and b["tensors"][tensor_name] is None
                    continue
                first = lambda t: t[:, 0] if site == "kda_core" else t[0]
                x, y, z = [first(v["tensors"][tensor_name]) for v in (a, candidate, b)]
                assert x.shape == y.shape == z.shape
                assert all(torch.isfinite(t).all() for t in (x,y,z))
                delta = (x.float() - y.float()).abs()
                rows.append(dict(rank=rank, layer=layer, site=site, tensor=tensor_name,
                    shape=list(x.shape), control_exact=torch.equal(x,z),
                    candidate_exact=torch.equal(x,y), different_elements=int((x != y).sum()),
                    max_abs=float(delta.max()) if delta.numel() else 0.0, mean_abs=float(delta.mean()) if delta.numel() else 0.0))
first_different = []
for rank in range(4):
    first_different.append(next((r for r in rows if r["rank"] == rank and not r["candidate_exact"]), None))
report = dict(kind="eager-first-target-token-layer-localization", rows=rows,
    controls_exact=all(r["control_exact"] for r in rows), first_different=first_different,
    numerical_gate_pass=False, performance_qualification=False)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))
