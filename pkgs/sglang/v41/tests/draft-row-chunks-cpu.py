"""DSpark draft row chunks (CPU, no GPU): `_draft_row_chunks` serves 9..16 rows as chunks of at most eight rows.

Loads the installed helper by AST and drives it with a row-independent stand-in for the block-FP8 linear. Checks that
1..8 and 17+ rows pass through as one call with the original arguments, that 9..16 rows become chunks of at most eight
rows with the matching rows of a pre-quantised input scale, that the result equals the one-call result row for row, and
that a bias or a non-2-D input keeps the single call. Also checks that the stage only installs it behind
SGLANG_DSV41_DSPARK_DRAFT_ROW_CHUNKS=1."""

import ast
import sys
from pathlib import Path

import torch

SITE = Path(sys.argv[1]).resolve() / "lib/python3.13/site-packages"
SRC = SITE / "sglang/srt/models/deepseek_v4_dspark.py"
text = SRC.read_text()
tree = ast.parse(text, filename=str(SRC))
nodes = [
    n
    for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name == "_draft_row_chunks")
    or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_DRAFT_NATIVE_ROWS")
]
assert len(nodes) == 2, "installed draft model lacks _draft_row_chunks"
namespace = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), "exec"), namespace)
chunks = namespace["_draft_row_chunks"]
assert 'os.environ.get("SGLANG_DSV41_DSPARK_DRAFT_ROW_CHUNKS", "0") == "1"' in text

calls = []


def block_linear(input, weight, block_size, weight_scale, input_scale=None, bias=None):
    calls.append((input.shape[0], None if input_scale is None else input_scale.shape[0], bias is not None))
    out = input.float() @ weight.float().t()
    if input_scale is not None:
        out = out * input_scale[:, :1]
    if bias is not None:
        out = out + bias
    return out.to(torch.bfloat16)


weight = torch.randn(24, 64)
for rows in range(1, 21):
    for scaled in (False, True):
        x = torch.randn(rows, 64)
        scale = torch.rand(rows, 2) + 0.5 if scaled else None
        calls.clear()
        got = chunks(block_linear)(
            input=x, weight=weight, block_size=[32, 32], weight_scale=None, input_scale=scale, bias=None
        )
        calls_made = list(calls)
        calls.clear()
        want = block_linear(x, weight, [32, 32], None, scale, None)
        assert torch.equal(got, want), (rows, scaled)
        if 8 < rows <= 16:
            sizes = [c[0] for c in calls_made]
            assert sizes == [8, rows - 8], (rows, sizes)
            if scaled:
                assert [c[1] for c in calls_made] == sizes, (rows, calls_made)
        else:
            assert [c[0] for c in calls_made] == [rows], (rows, calls_made)

calls.clear()
bias = torch.randn(24)
chunks(block_linear)(input=torch.randn(12, 64), weight=weight, block_size=[32, 32], weight_scale=None, bias=bias)
assert calls == [(12, None, True)], calls

calls.clear()
x3 = torch.randn(2, 6, 64)
try:
    chunks(block_linear)(input=x3, weight=weight, block_size=[32, 32], weight_scale=None)
except RuntimeError:
    pass
assert [c[0] for c in calls] == [2], calls
print("draft row chunks: ok")
