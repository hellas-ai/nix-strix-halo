#!/usr/bin/env python3
"""Exercise the actual serving block without importing GPU/model code."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace


def check(source):
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_process_messages')
    block = next(n for n in method.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'glm_constraint')
    executable = compile(ast.Module(body=block.body, type_ignores=[]), str(source), 'exec')
    cases = [
        ('always', False, True),
        ('always', None, True),
        ('always', True, True),
        (None, False, False),
        (None, None, True),
        (None, True, True),
    ]
    for special_case, toggle, expected in cases:
        calls = []
        class Parser:
            def __init__(self, *args): pass
            def get_structure_constraint(self, *args, **kwargs):
                calls.append(kwargs['thinking_mode'])
        config = SimpleNamespace(special_case=special_case) if special_case else None
        namespace = dict(
            self=SimpleNamespace(template_manager=SimpleNamespace(reasoning_config=config),tool_call_parser='glm47'),
            request=SimpleNamespace(chat_template_kwargs={'enable_thinking':toggle},tools=[],tool_choice='auto',parallel_tool_calls=True),
            FunctionCallParser=Parser,
        )
        exec(executable, namespace)
        assert calls == [expected], (special_case, toggle, calls, expected)
    return len(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('serving_chat', type=Path)
    args = parser.parse_args()
    print(f'PASS: {check(args.serving_chat)} actual serving-block grammar cases')
