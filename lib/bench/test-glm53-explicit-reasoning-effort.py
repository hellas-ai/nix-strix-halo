#!/usr/bin/env python3
"""Check request/default precedence using the installed rendering blocks."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace


def check(path):
    tree = ast.parse(path.read_text())
    defaults = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Attribute)
                    and n.test.attr == 'default_chat_template_kwargs')
    merge = None
    for parent in ast.walk(tree):
        for _, body in ast.iter_fields(parent):
            if not isinstance(body, list):
                continue
            for i, node in enumerate(body):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'extra_template_kwargs'
                    for target in node.targets
                ) and isinstance(node.value, ast.Dict) and not node.value.keys:
                    merge = body[i:i+3]
                    break
            if merge is not None:
                break
        if merge is not None:
            break
    assert merge is not None
    code = compile(ast.Module(body=[defaults, *merge], type_ignores=[]), str(path), 'exec')
    cases = [(None, None, 'low'), ('low', None, 'low'), ('high', None, 'high'),
             ('max', None, 'max'), ('none', None, 'none'),
             ('high', {'reasoning_effort': 'low'}, 'low'),
             (None, {'reasoning_effort': 'max'}, 'max'),
             ('max', {'clear_thinking': False}, 'max')]
    for requested, kwargs, expected in cases:
        request = SimpleNamespace(reasoning_effort=requested, chat_template_kwargs=kwargs)
        scope = dict(request=request, self=SimpleNamespace(
            default_chat_template_kwargs={'clear_thinking': True, 'reasoning_effort': 'low'}))
        exec(code, scope)
        rendered = scope['extra_template_kwargs']
        assert rendered.get('reasoning_effort') == expected, (requested, kwargs, expected)
        assert rendered['clear_thinking'] is (False if kwargs and kwargs.get('clear_thinking') is False else True)
    return len(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('serving_chat', type=Path)
    args = parser.parse_args()
    print(f'PASS: {check(args.serving_chat)} actual rendering precedence cases')
