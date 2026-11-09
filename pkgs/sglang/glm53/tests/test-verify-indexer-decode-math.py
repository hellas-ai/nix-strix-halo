"""Check request-major reconstruction and unchanged fallback gate dispatch."""
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
tree = ast.parse(args.source.read_text())
method = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == '_get_verify_logits_head_gate')
namespace = {'torch': torch, 'is_hip': lambda: True}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(args.source), 'exec'), namespace)
dispatch = namespace[method.name]
calls = []

def gate(x, scale):
    assert x.is_contiguous() and scale.is_contiguous()
    calls.append((x.clone(), scale.clone()))
    return x[:, :2].unsqueeze(-1) * scale

owner = SimpleNamespace(_get_logits_head_gate=gate)
for batch in (1, 2, 3, 4):
    for width in (1, 2, 4):
        rows = batch * width
        x = torch.arange(rows * 5, dtype=torch.float32).reshape(rows, 5)
        scales = torch.arange(1, rows + 1, dtype=torch.float32).reshape(rows, 1, 1)
        expected = gate(x, scales)
        for enabled in ('0', '1'):
            calls.clear()
            os.environ['SGLANG_GLM53_VERIFY_DECODE_MATH'] = enabled
            actual = dispatch(owner, x, scales, width)
            assert torch.equal(expected, actual), (batch, width, enabled)
            assert len(calls) == (width if enabled == '1' else 1)
            if enabled == '1':
                for step, (observed_x, observeded_scale) in enumerate(calls):
                    assert torch.equal(observed_x, x[step::width])
                    assert torch.equal(observeded_scale, scales[step::width])

namespace['is_hip'] = lambda: False
calls.clear()
assert torch.equal(dispatch(owner, x, scales, width), expected)
assert len(calls) == 1
print('PASS C1-C4 request-major gate ordering, contiguous decode-sized calls, widths 1/2/4, disabled and non-HIP paths')
