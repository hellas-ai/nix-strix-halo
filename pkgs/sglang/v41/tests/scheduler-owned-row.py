"""CPU-only regression for a continuing chunk's already-owned request row.

Execute the installed scheduler's allocation method and outer prefill queue
gate from AST. The fixture starts after add_chunked_req has selected the owned
continuation, and advances can_run_list as add_one_req would. No GPU import.
"""

import ast
import sys
import types
from pathlib import Path


source = next(
    (Path(sys.argv[1]) / "lib").glob(
        "python*/site-packages/sglang/srt/managers/scheduler.py"
    )
)
module = ast.parse(source.read_text())
scheduler = next(
    node for node in module.body
    if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
)
alloc = next(
    node for node in scheduler.body
    if isinstance(node, ast.FunctionDef) and node.name == "get_num_allocatable_reqs"
)
prefill = next(
    node for node in scheduler.body
    if isinstance(node, ast.FunctionDef) and node.name == "_get_new_batch_prefill_raw"
)
loop = next(
    node for node in ast.walk(prefill)
    if isinstance(node, ast.For)
    and isinstance(node.target, ast.Name)
    and node.target.id == "req"
    and isinstance(node.iter, ast.Attribute)
    and node.iter.attr == "waiting_queue"
)
start = next(
    i for i, node in enumerate(loop.body)
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "running_bs"
            for target in node.targets)
)
end = next(
    i for i, node in enumerate(loop.body[start:], start)
    if isinstance(node, ast.If)
    and isinstance(node.test, ast.Attribute)
    and node.test.attr == "batch_is_full"
)
gate = loop.body[start : end + 1]


class StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
        return self.generic_visit(node)


def parse_body(body):
    return ast.parse(body).body


# The two copies of the real method nodes are placed in a tiny CPU harness;
# only the result recording and admitted-list append are test scaffolding.
harness = ast.FunctionDef(
    name="outer_gate",
    args=ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name) for name in
              ("self", "running_batch", "adder", "continuing_chunk")],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
    ),
    body=parse_body("admitted = []") + [
        ast.For(
            target=ast.Name(id="req", ctx=ast.Store()),
            iter=ast.parse("self.waiting_queue", mode="eval").body,
            body=gate + parse_body(
                "admitted.append(req)\n"
                "adder.can_run_list.append(req)"
            ),
            orelse=[],
        ),
    ] + parse_body("return len(admitted), running_batch.batch_is_full"),
    decorator_list=[],
)
program = ast.fix_missing_locations(ast.Module(
    body=[StripAnnotations().visit(alloc), harness], type_ignores=[]
))
globals_ = {
    "get_parallel": lambda: types.SimpleNamespace(pp_max_micro_batch_size=2),
    "DisaggregationMode": types.SimpleNamespace(PREFILL="prefill"),
}
exec(compile(program, str(source), "exec"), globals_)  # noqa: S102


def candidate(beam=False):
    return types.SimpleNamespace(
        token_indices_to_pool=None,
        beam_group=types.SimpleNamespace(beam_width=2) if beam else None,
    )


def case(*, free, pp, queue, expected, leader_beam=False, owned=True,
         pending=0, disagg="decode"):
    globals_["get_parallel"] = lambda: types.SimpleNamespace(
        pp_max_micro_batch_size=pp
    )
    long = candidate(leader_beam)
    long.kv = types.SimpleNamespace(holds_kv=owned)
    batch = types.SimpleNamespace(reqs=[], batch_is_full=False)
    adder = types.SimpleNamespace(can_run_list=[long])
    sched = types.SimpleNamespace(
        waiting_queue=[candidate(beam) for beam in queue],
        req_to_token_pool=types.SimpleNamespace(available_size=lambda: free),
        beam_coordinator=types.SimpleNamespace(
            pending_member_rows=lambda _: pending
        ),
        running_batch=batch,
        disaggregation_mode=disagg,
        enable_priority_preemption=False,
    )
    sched.get_num_allocatable_reqs = types.MethodType(
        globals_["get_num_allocatable_reqs"], sched
    )
    actual = globals_["outer_gate"](sched, batch, adder, long)
    assert actual == expected, (free, pp, queue, leader_beam, actual, expected)


# Two usable rows: the continuing chunk owns one and exactly one is free.
case(free=1, pp=2, queue=[False], expected=(1, False))
# The credit does not add a PP slot or grant a fresh row from a full pool.
case(free=1, pp=1, queue=[False], expected=(0, True))
case(free=0, pp=2, queue=[False], expected=(0, True))
case(free=1, pp=2, queue=[False], owned=False, expected=(0, True))
# Future beam members must retain their unspawned row reservation.
case(free=1, pp=2, queue=[False], leader_beam=True, expected=(0, True))
case(free=2, pp=3, queue=[True, False], expected=(0, True))
case(free=2, pp=3, queue=[False, True], expected=(1, True))
case(free=1, pp=2, queue=[False], pending=1, expected=(0, True))
# Disaggregated prefill keeps its additional transfer-queue constraint.
case(free=1, pp=2, queue=[False], disagg="prefill", expected=(0, True))
print("scheduler owned-row admission: 9 CPU source-path cases passed")
