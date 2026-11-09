"""Exercise the installed scheduler's cadence control without a GPU.

Validating the real method prevents a rejected multi-key update from partially
changing the scheduling cadence or reported configuration.
"""
import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

if len(sys.argv) > 1:
    source = Path(sys.argv[1])
else:
    import sglang
    source = Path(sglang.__file__).parent / "srt/managers/scheduler.py"

tree = ast.parse(source.read_text())
klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
names = {"set_internal_state", "_should_defer_prefill", "_arm_prefill_decode_interval"}
methods = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(methods) == len(names)
overrides = []
ns = {"logging": logging, "logger": logging.getLogger(__name__), "Optional": Optional,
      "ScheduleBatch": object, "SetInternalStateReq": object,
      "SetInternalStateReqOutput": SimpleNamespace,
      "get_context": lambda: SimpleNamespace(override=lambda **kw: overrides.append(kw))}
exec(compile(ast.Module(methods, []), str(source), "exec"), ns)


def scheduler(idle=True):
    return SimpleNamespace(
        prefill_decode_interval=0, _prefill_decode_interval_remaining=0,
        is_fully_idle=lambda: idle, require_mlp_sync=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        metrics_reporter=SimpleNamespace(spec_total_num_accept_tokens=5, spec_total_num_forward_ct=2),
    )


def update(state, args):
    return ns["set_internal_state"](state, SimpleNamespace(server_args=args)).updated


for value in (-1, 1.5, True, "4", None):
    state = scheduler()
    assert not update(state, {"prefill_decode_interval": value})
    assert state.prefill_decode_interval == 0 and not overrides
state = scheduler(idle=False)
assert not update(state, {"prefill_decode_interval": 1}) and not overrides
state = scheduler()
assert not update(state, {"prefill_decode_interval": 4, "unknown_option": 1})
assert state.prefill_decode_interval == 0 and not overrides
assert state.metrics_reporter.spec_total_num_accept_tokens == 5

for value in (1, 4, 8, 0):
    state._prefill_decode_interval_remaining = 19
    assert update(state, {"prefill_decode_interval": value})
    assert state.prefill_decode_interval == value
    assert state._prefill_decode_interval_remaining == 0
    assert overrides[-1] == {"source": "update_server_args", "prefill_decode_interval": value}
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
    ns["_arm_prefill_decode_interval"](state, batch)
    assert [ns["_should_defer_prefill"](state) for _ in range(value + 1)] == [True] * value + [False]
    batch.forward_mode.is_extend = lambda: False
    ns["_arm_prefill_decode_interval"](state, batch)
    assert not ns["_should_defer_prefill"](state)

# DP scheduling uses the synchronized extend flag to arm every rank together.
state.require_mlp_sync = True
state.prefill_decode_interval = 4
batch.is_extend_in_batch = True
ns["_arm_prefill_decode_interval"](state, batch)
assert state._prefill_decode_interval_remaining == 4
print("PASS: idle cadence updates, atomic rejection, readback override, local and DP scheduling")
