"""Verify cadence control and HTTP readback using actual runtime methods.

Run with an installed runtime containing 0050. The HTTP top-level config is
the tokenizer's startup record; scheduler-owned internal_states are live.
No GPU, service, or real HTTP requests are used.
"""
import ast
import asyncio
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import sglang

root = Path(sglang.__file__).parent


def extract(path, names, namespace):
    tree = ast.parse((root / path).read_text())
    methods = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert len(methods) == len(names), [node.name for node in methods]
    for node in methods:
        node.decorator_list = []
    exec(compile(ast.Module(methods, []), str(root / path), "exec"), namespace)
    return namespace


startup = {"prefill_decode_interval": 1, "max_total_tokens": 196608}
args = SimpleNamespace(resolved_dict=lambda: dict(startup), launch_command="sglang serve --prefill-decode-interval 1")
context = SimpleNamespace(server_args=args, _overrides_log=[])
context_ns = extract("srt/runtime_context.py", {"resolved_server_args_dict"}, {})
context.resolved_server_args_dict = MethodType(context_ns["resolved_server_args_dict"], context)
context.override = lambda source, **fields: context._overrides_log.append((source, fields))
parallel = SimpleNamespace(enable_dp_attention=False, dp_size=1, tp_size=4, pp_size=1)
ns = {"logging": logging, "logger": logging.getLogger(__name__),
      "SetInternalStateReq": object, "GetInternalStateReq": object,
      "SetInternalStateReqOutput": SimpleNamespace, "GetInternalStateReqOutput": SimpleNamespace,
      "get_context": lambda: context, "get_parallel": lambda: parallel,
      "compute_world_size": lambda **kw: 4,
      "build_memory_usage": lambda **kw: {"token_capacity": kw["token_capacity"]},
      "get_exec": lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None)),
      "envs": SimpleNamespace(SGLANG_EXPOSE_OWN_ENV_VARS=SimpleNamespace(get=lambda: False)),
      "msgspec_to_builtins": lambda value: value, "RECORD_STEP_TIME": False}
extract("srt/managers/scheduler.py", {"set_internal_state", "get_internal_state"}, ns)
state = SimpleNamespace(prefill_decode_interval=1, _prefill_decode_interval_remaining=7,
    is_fully_idle=lambda: True, require_mlp_sync=False,
    spec_algorithm=SimpleNamespace(is_none=lambda: True, is_dspark=lambda: False),
    metrics_reporter=SimpleNamespace(last_gen_throughput=0., spec_total_num_forward_ct=0,
                                    spec_total_num_accept_tokens=0),
    draft_worker=None, startup_time=2., max_total_num_tokens=131072,
    max_running_requests=4, swa_tokens_per_layer=None, startup_available_gpu_memory_gb=25.,
    tp_worker=SimpleNamespace(model_runner=SimpleNamespace(weight_load_mem_usage=75.), graph_memory_usage={}),
    token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: SimpleNamespace(mem_usage=2.)))


def current():
    return ns["get_internal_state"](state, object()).internal_state


async def internal_states():
    return [current()]


http_ns = {"List": List, "Dict": Dict, "Any": Any, "msgspec_to_builtins": lambda value: value,
    "describe_kv_events_publisher": lambda args: None, "__version__": "test",
    "_global_state": SimpleNamespace(tokenizer_manager=SimpleNamespace(server_args=args,
        startup_time=2., get_internal_state=internal_states), scheduler_info={"max_total_num_tokens": 131072})}
extract("srt/entrypoints/http_server.py", {"server_info"}, http_ns)

for value in (0, 4, 1):
    result = ns["set_internal_state"](state, SimpleNamespace(server_args={"prefill_decode_interval": value}))
    assert result.updated and state.prefill_decode_interval == value
    assert state._prefill_decode_interval_remaining == 0
    info = asyncio.run(http_ns["server_info"]())
    assert info["prefill_decode_interval"] == 1  # Pristine tokenizer launch record.
    assert info["max_total_tokens"] == 196608  # Requested token cap.
    assert info["max_total_num_tokens"] == 131072  # Resolved scheduler handshake.
    assert [row["prefill_decode_interval"] for row in info["internal_states"]] == [value]
    assert info["internal_states"][0]["memory_usage"]["token_capacity"] == 131072

before = current()
state.is_fully_idle = lambda: False
assert not ns["set_internal_state"](state, SimpleNamespace(server_args={"prefill_decode_interval": 8})).updated
assert current() == before and state.prefill_decode_interval == 1
state.is_fully_idle = lambda: True
for value in (-1, 1.5, True, "4", None):
    assert not ns["set_internal_state"](state, SimpleNamespace(server_args={"prefill_decode_interval": value})).updated
    assert current() == before
assert not ns["set_internal_state"](state, SimpleNamespace(server_args={"prefill_decode_interval": 4, "unsupported": 1})).updated
assert current() == before
print("PASS: actual idle cadence update→context overlay→scheduler readback→HTTP nested live state; launch/current/resolved values stay distinct")
