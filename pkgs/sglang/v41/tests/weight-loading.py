"""CPU contract checks for V4's bounded copy queue, using the exact source AST.

The complete model module has GPU imports. Extract only its pure iterator and
configuration statements; preserve the actual existing copy-submission helper.
Use actual runtime configuration publication and CPU-only Torch availability.
"""
import ast
import concurrent.futures
import gc
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import torch
from sglang.srt.runtime_context import get_context, get_model, reset_context
from sglang.srt.server_args import ServerArgs

assert not torch.cuda.is_available(), 'CPU-only check requires hidden GPU devices'
runtime = Path(sys.argv[1]).resolve()
candidate_path = runtime / 'lib/python3.13/site-packages/sglang/srt/models/deepseek_v4.py'
model_ast = ast.parse(candidate_path.read_text())
helper = next(n for n in model_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_bounded_weight_iterator')
model_class = next(n for n in model_ast.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV4ForCausalLM')
load = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == 'load_weights')
start = next(i for i,n in enumerate(load.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'extra_config')
config_statements = load.body[start:start+4]
assert isinstance(config_statements[-1], ast.Assign) and config_statements[-1].targets[0].id == 'num_threads'
async_assignment = next(n for n in ast.walk(load) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'use_async_loading')
assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_bounded_weight_iterator' for n in ast.walk(load))
utils_path = runtime / 'lib/python3.13/site-packages/sglang/srt/model_loader/utils.py'
utils_ast = ast.parse(utils_path.read_text())
utils = [n for n in utils_ast.body if isinstance(n, ast.FunctionDef) and n.name in ('should_async_load', 'maybe_executor_submit')]
assert len(utils) == 2
namespace = dict(concurrent=concurrent, torch=torch, json=json, get_model=get_model)
future_import = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
module = ast.fix_missing_locations(ast.Module(body=[future_import, helper, *utils], type_ignores=[]))
exec(compile(module, str(candidate_path), 'exec'), namespace)
bounded = namespace['_bounded_weight_iterator']
submit = namespace['maybe_executor_submit']
async_expr = compile(ast.Expression(async_assignment.value), str(candidate_path), 'eval')
cases = 0


def options(extra_config):
    get_context().set_server_args(ServerArgs(model_path='dummy', model_loader_extra_config=extra_config))
    scope = namespace.copy()
    exec(compile(ast.Module(body=config_statements, type_ignores=[]), str(candidate_path), 'exec'), scope)
    return scope['async_enabled'], scope['num_threads']


assert options('{}') == (True, 8)
assert options('{"enable_multithread_load":false}') == (False, 8)
assert options('{"num_threads":2}') == (True, 2)
assert options({'enable_multithread_load': False, 'num_threads': 3}) == (False, 3)
cases += 4


class Buffer:
    def __init__(self, value, *, borrowed=False):
        self.value = value
        self.data = bytearray([value] * 4096)
        self.device = SimpleNamespace(type='cpu')
        self._sglang_runai_streamer_tensor = borrowed


def consume(source, callback, *, asynchronous=True, limit=2, errors=None):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=limit) as executor:
            futures = []
            for name, loaded_weight in bounded(source, futures, max_pending=limit):
                use_async = eval(async_expr, namespace, dict(async_enabled=asynchronous, loaded_weight=loaded_weight))
                submit(executor=executor, futures=futures, use_async=use_async,
                       func=callback, func_args=(loaded_weight,))
                assert len(futures) <= limit
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except BaseException as exc:
        if errors is None:
            raise
        errors.append((type(exc).__name__, str(exc)))


def owned_case(fail=False):
    refs, produced, copied, errors = [], [], [], []
    gates = [threading.Event() for _ in range(7)]
    entered = [threading.Event() for _ in range(7)]
    created = [threading.Event() for _ in range(7)]

    def source():
        for i in range(7):
            item = Buffer(i)
            refs.append(weakref.ref(item))
            produced.append(i)
            created[i].set()
            yield str(i), item

    def copy(item):
        i = item.value
        entered[i].set()
        assert gates[i].wait(5), 'callback timed out'
        if fail and i == 0:
            raise RuntimeError('independent-copy-failure')
        assert item.data == bytearray([i] * 4096)
        copied.append(i)

    thread = threading.Thread(target=consume, args=(source(), copy), kwargs=dict(errors=errors))
    thread.start()
    assert entered[0].wait(5) and entered[1].wait(5)
    assert not created[2].wait(0.1), 'source advanced beyond in-flight bound'
    gc.collect()
    assert refs[0]() is not None and refs[1]() is not None
    gates[0].set()
    if fail:
        assert not created[2].wait(0.1), 'failure must stop the next fetch'
        assert thread.is_alive(), 'executor must wait for remaining copy ownership'
        assert refs[1]() is not None
    else:
        assert created[2].wait(5)
        assert not created[3].wait(0.1), 'refill must still respect the bound'
        assert refs[1]() is not None and refs[2]() is not None
    for gate in gates:
        gate.set()
    thread.join(5)
    assert not thread.is_alive()
    if fail:
        assert produced == [0,1] and copied == [1]
        assert errors == [('RuntimeError', 'independent-copy-failure')], errors
    else:
        assert produced == list(range(7)) and sorted(copied) == list(range(7)) and not errors
    gc.collect()
    assert all(ref() is None for ref in refs), 'source owners leaked after shutdown'


owned_case()
owned_case(fail=True)
cases += 2

# A borrowed source buffer is rewritten before every yield. Synchronous config
# and the existing streamed-buffer marker must each preserve every original value.
for asynchronous, borrowed in [(False, False), (True, True)]:
    shared = Buffer(0, borrowed=borrowed)
    copied, thread_ids = [], []

    def reused_source():
        for i in range(11):
            shared.value = i
            shared.data[:] = bytes([i]) * len(shared.data)
            yield str(i), shared

    def copy_shared(item):
        copied.append(bytes(item.data))
        thread_ids.append(threading.get_ident())

    consume(reused_source(), copy_shared, asynchronous=asynchronous)
    assert copied == [bytes([i])*4096 for i in range(11)]
    assert thread_ids == [threading.get_ident()] * 11
    cases += 1

# The final drain still surfaces a failure when the source is shorter than the cap.
errors = []
def fail_final(item):
    raise ValueError('final-drain-failure')
consume([('only', Buffer(3))], fail_final, limit=8, errors=errors)
assert errors == [('ValueError', 'final-drain-failure')]
cases += 1

seen = []
consume([], lambda item: seen.append(item))
assert seen == []
try:
    list(bounded(iter([1]), [], max_pending=0))
except ValueError:
    pass
else:
    raise AssertionError('zero bound accepted')
cases += 2
reset_context()
print(json.dumps(dict(event='complete', cases=cases, gpu_used=False,
                     default_async=True, serial_config_honored=True,
                     bounded_source_ownership=True, exceptions_propagated=True)))
