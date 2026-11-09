#!/usr/bin/env python3
"""CPU-only actual-method qualification of the default-off0061 proposal.

If --source does not contain0061, copy its three exact files into a temporary
tree and apply --patch there. Never mutate the installed source or use a GPU.
"""
import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as N
from unittest.mock import patch

import torch

sys.dont_write_bytecode = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fails(function):
    try:
        function()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Invalid reference was accepted')


def method(path, cls, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    function = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def req(ids=None, prompt=10, rid='fixture', greedy=False):
    params = N(temperature=1., top_p=.95, top_k=1 if greedy else 1 << 30, min_p=0.,
        frequency_penalty=0., presence_penalty=0., repetition_penalty=1., min_new_tokens=0,
        max_new_tokens=4 if ids is None else len(ids), ignore_eos=True, n=1, logit_bias=None,
        custom_params=None if ids is None else {'glm53_forced_decode_ids': list(ids)})
    return N(sampling_params=params, origin_input_ids=[1] * prompt, rid=rid,
        output_ids=[], return_logprob=True, return_sampling_mask=False,
        custom_logit_processor=None, grammar=None)


class Penalizer:
    is_required = False
    def __init__(self, **kwargs): pass
    def filter(self, indices): pass
    def merge(self, other): pass


def actual_batch_class(source, reference):
    module = ModuleType('actual_reference_sampling_batch')
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(torch=torch, dataclasses=dataclasses, ForcedDecodeReference=reference.ForcedDecodeReference,
        refs_from_reqs=reference.refs_from_reqs, TOP_K_ALL=1 << 30,
        get_exec=lambda: N(deterministic=N(enable_deterministic_inference=False),
                           features=N(enable_custom_logit_processor=False), dllm=N(dllm_algorithm=None)),
        is_pin_memory_available=lambda device: False,
        penaltylib=N(BatchedPenalizerOrchestrator=Penalizer, BatchedFrequencyPenalizer=int,
            BatchedMinNewTokensPenalizer=str, BatchedPresencePenalizer=float, BatchedRepetitionPenalizer=bool))
    path = source / 'srt/sampling/sampling_batch_info.py'
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes.extend(node for node in tree.body if getattr(node, 'name', '') in ('SamplingBatchInfo', 'merge_bias_tensor'))
    ast.fix_missing_locations(module_ast := ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module_ast, str(path), 'exec'), namespace)
    return namespace['SamplingBatchInfo']


def batch(cls, requests):
    return cls.from_schedule_batch(N(reqs=requests, device='cpu', spec_algorithm=N(is_none=lambda: True)), 3)


def runner():
    return N(is_draft_worker=False, spec_algorithm=N(is_none=lambda: True),
        model_config=N(hf_text_config=N(model_type='glm5_next_text')),
        supports_sampling_observer=lambda: True, sampling_observer=None, ps=N(tp_rank=0))


def forward(info, mode, lengths):
    return N(sampling_info=info, forward_mode=N(name=mode, is_decode=lambda: mode == 'DECODE'),
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64), seq_lens=torch.tensor(lengths),
        batch_size=len(lengths), return_logprob=True, top_logprobs_nums=[3] * len(lengths),
        token_ids_logprobs=[list(range(3))] * len(lengths), positions=torch.tensor(lengths) - 1)


def test_metadata(source, reference):
    cls = actual_batch_class(source, reference)
    fails(lambda: cls.from_schedule_batch(N(reqs=[req([0, 1])], device='cpu',
        spec_algorithm=N(is_none=lambda: False)), 3))
    original = batch(cls, [req([2, 1, 0, 2], rid='a'), req(None, rid='ordinary')])
    frozen = original.copy_for_forward()
    original_refs = frozen.glm53_forced_decode_references
    original.filter_batch([0], torch.tensor([0]))
    assert len(frozen.glm53_forced_decode_references) == 2
    original.merge_batch(batch(cls, [req([1, 2, 0, 1], rid='b')]))
    assert original.glm53_forced_decode_references[1].ids == (1, 2, 0, 1)
    assert frozen.glm53_forced_decode_references is original_refs
    assert frozen.glm53_forced_decode_references[1] is None
    info = batch(cls, [req([2, 1, 0, 2], rid='a'), req([1, 2, 0, 1], rid='b')])
    a = reference.prepare_reference(runner(), forward(info, 'EXTEND', [10, 9]))
    assert [(row['row'], row['position'], row['token_id']) for row in a['rows']] == [(0, 0, 2)]
    # Req.output_ids remains empty: overlap's CPU sequence mirror selects index2.
    b = reference.prepare_reference(runner(), forward(info, 'DECODE', [12, 11]))
    assert [(row['row'], row['position'], row['token_id']) for row in b['rows']] == [(0, 2, 0), (1, 1, 2)]
    assert reference.prepare_reference(runner(), forward(info, 'DECODE', [14, 14])) is None
    invalid = forward(info, 'DECODE', [10, 11])
    fails(lambda: reference.prepare_reference(runner(), invalid))
    invalid = forward(info, 'EXTEND', [11, 10])
    fails(lambda: reference.prepare_reference(runner(), invalid))
    invalid = forward(info, 'DECODE', [11, 11])
    invalid.seq_lens_cpu = torch.empty(2, device='meta')
    fails(lambda: reference.prepare_reference(runner(), invalid))
    invalid = runner(); invalid.spec_algorithm = N(is_none=lambda: False)
    fails(lambda: reference.prepare_reference(invalid, forward(info, 'DECODE', [11, 11])))
    invalid = runner(); invalid.is_draft_worker = True
    fails(lambda: reference.prepare_reference(invalid, forward(info, 'DECODE', [11, 11])))
    info.has_custom_logit_processor = True
    fails(lambda: reference.prepare_reference(runner(), forward(info, 'DECODE', [11, 11])))
    info.has_custom_logit_processor = False
    for ids in ([], [True], [3], [-1], [0] * 1025):
        fails(lambda ids=ids: reference.refs_from_reqs([req(ids)], 3))
    assert reference.refs_from_reqs([object()], 3, enabled=False) is None
    assert reference.prepare_reference(object(), object(), enabled=False) is None
    return cls


class Gather:
    def __init__(self, events): self.events = events
    def compute_logprobs(self, logprobs, top, token_ids, chosen):
        self.events.append('actual-gather')
        self.logprobs, self.chosen = logprobs.clone(), chosen.clone()
        self.selected = logprobs.gather(1, chosen.long().unsqueeze(1)).flatten()
        return N(write_output_to=lambda output: setattr(output, 'raw_selected', self.selected.clone()))


def test_actual_sampler(source, reference, cls):
    events = []
    def capture(logits, data):
        events.append('actual-capture')
        return reference.capture_raw_logprobs(logits, data)
    def commit(ids, logprobs, data):
        events.append('actual-commit')
        return reference.commit_reference(ids, logprobs, data)
    namespace = {'torch': torch, 'LogitsProcessorOutput': N, 'SamplingBatchInfo': cls,
        'List': list, '_trace_e2e_sampler': lambda stage, **kwargs: events.append(stage),
        '_is_hip': False, '_use_aiter': False, '_disable_aiter_greedy_sample': False,
        'SGLANG_RETURN_ORIGINAL_LOGPROB': False,
        'capture_raw_logprobs': capture,
        'commit_reference': commit}
    sampler_forward = method(source / 'srt/layers/sampler.py', 'Sampler', 'forward', namespace)
    for greedy in (False, True):
        for original_logprobs in (False, True):
            namespace['SGLANG_RETURN_ORIGINAL_LOGPROB'] = original_logprobs
            info = batch(cls, [req([2, 0, 1, 2], greedy=greedy), req([0, 2, 1, 0], greedy=greedy)])
            fb = forward(info, 'EXTEND', [10, 10])
            fixed = reference.prepare_reference(runner(), fb)
            logits = torch.tensor([[.3, .1, -.7], [1.2, -.4, .8]])
            raw = torch.log_softmax(logits, dim=-1)
            gather = Gather(events)
            sampled_probs = []
            def ordinary(probs, *args):
                events.append('actual-ordinary-sample')
                sampled_probs.append(probs.clone())
                return torch.ones(2, dtype=torch.int64), None
            sampler = N(_preprocess_logits=lambda values, info: values,
                use_ascend_backend=False, rl_on_policy_target=None, use_log_softmax_logprob=False,
                enable_deterministic=False, _sample_from_probs=ordinary, output_logprob_processor=gather,
                _sync_token_ids_across_tp=lambda ids, info: events.append('actual-sync'))
            output = N(next_token_logits=logits.clone())
            records = []
            with patch.object(reference.logger, 'info', side_effect=records.append):
                result = sampler_forward(sampler, output, info, True, [3, 3], [[0, 1, 2]] * 2,
                    fb.positions, glm53_reference=fixed)
            assert result.tolist() == [2, 0] and gather.chosen.tolist() == [2, 0]
            assert torch.equal(gather.logprobs, raw)
            assert torch.equal(output.raw_selected, raw[torch.arange(2), result])
            assert output.raw_selected[0] < -.5  # The forced token was not masked to probability1.
            if not greedy:
                assert torch.equal(sampled_probs[0], torch.softmax(logits, dim=-1))
                assert torch.equal(output.next_token_logits, sampled_probs[0])
            assert events.index('preprocess_returned') < events.index('token_sync_enter')
            ordinary_event = 'greedy_returned' if greedy else 'actual-ordinary-sample'
            assert events.index('actual-capture') < events.index(ordinary_event) < events.index('actual-commit')
            assert events.index('actual-commit') < events.index('actual-gather') < events.index('actual-sync')
            for text in records:
                row = json.loads(text.split('GLM53_FORCED_DECODE ', 1)[1])
                assert row['forward_mode'] == 'EXTEND' and row['position'] == 0
                assert len(row['request_sha256']) == 64 and row['raw_logprobs_before_forcing']
            # Inactive invocation has identical old signature and ordinary IDs.
            output = N(next_token_logits=logits.clone())
            result = sampler_forward(sampler, output, info, True, [3, 3], [[0, 1, 2]] * 2, fb.positions)
            expected = torch.argmax(logits, -1) if greedy else torch.ones(2, dtype=torch.int64)
            assert torch.equal(result, expected)
            events.clear()


def test_actual_runner(source, reference, cls):
    path = source / 'srt/model_executor/model_runner.py'
    namespace = {'torch': torch, 'LogitsProcessorOutput': N, 'ForwardBatch': N}
    actual_sample = method(path, 'ModelRunner', 'sample', namespace)
    info = batch(cls, [req([2, 0, 1, 2])])
    fb = forward(info, 'DECODE', [12])
    calls = []
    r = runner()
    r._preprocess_logits = lambda output, info: None
    r.sampler = lambda *args, **kwargs: calls.append(kwargs) or torch.tensor([kwargs['glm53_reference']['rows'][0]['token_id']])
    r.ngram_embedding_manager = N(update_after_decode=lambda **kwargs: calls.append(kwargs))
    with patch.dict(sys.modules, {'sglang.srt.sampling.glm53_forced_decode_reference': reference}):
        result = actual_sample(r, N(next_token_logits=torch.zeros(1, 3)), fb)
        assert result.tolist() == [1] and calls[0]['glm53_reference']['rows'][0]['position'] == 2
        assert calls[1]['next_token_ids'].tolist() == [1]
        # A normal request keeps legacy sampler positional arguments and no kwargs.
        fb.sampling_info.glm53_forced_decode_references = None
        r.sampler = lambda *args: torch.tensor([2])
        assert actual_sample(r, N(next_token_logits=torch.zeros(1, 3)), fb).tolist() == [2]


def test_frozen_requests(source, reference, fixtures, original_source):
    # Execute installed SamplingParams.__post_init__/verify methods so greedy
    # normalization, TOP_K_ALL and JSON custom-parameter transport use the real
    # source rather than a fixture that guesses the constructor behavior.
    path = original_source / 'srt/sampling/sampling_params.py'
    manifest = json.loads((fixtures / 'manifest.json').read_bytes())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['source_sha256']['srt/sampling/sampling_params.py']
    namespace = {'math': math, 'TOP_K_ALL': 1 << 30, '_SAMPLING_EPS': 1e-6,
        'MAX_STOP_COUNT': 32, 'MAX_STOP_REGEX_COUNT': 32, 'MAX_STOP_REGEX_LEN': 256}
    tree = ast.parse(path.read_text())
    helpers = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    helpers.extend(node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and ('REQUEST_REASONING_END_TOKEN_IDS' in target.id)
        for target in node.targets))
    helpers.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'get_request_reasoning_end_token_ids')
    ast.fix_missing_locations(helper_ast := ast.Module(body=helpers, type_ignores=[]))
    exec(compile(helper_ast, str(path), 'exec'), namespace)
    post = method(path, 'SamplingParams', '__post_init__', namespace)
    verify = method(path, 'SamplingParams', 'verify', namespace)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SamplingParams')
    defaults = {field.target.id: eval(compile(ast.Expression(field.value), str(path), 'eval'), namespace)
        for field in node.body if isinstance(field, ast.AnnAssign) and field.value is not None}
    totals = {}
    for kind in ('scores', 'full'):
        fixture = fixtures / (kind + '-requests.json')
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest['fixtures'][kind]['fixture_sha256']
        original_trace = Path(manifest['fixtures'][kind]['source_trace_path'])
        assert hashlib.sha256(original_trace.read_bytes()).hexdigest() == manifest['fixtures'][kind]['source_trace_sha256']
        data = json.loads(fixture.read_bytes())
        count = 0
        for group in data['groups']:
            requests = []
            for case in group:
                params = N(**dict(defaults, **case['body']['sampling_params']))
                post(params)
                verify(params, 154880)
                params.custom_params['__req__'] = object()  # Req's real internal injection is allowed.
                requests.append(N(sampling_params=params, rid=case['body']['rid'],
                    origin_input_ids=case['body']['input_ids'], output_ids=[], return_logprob=True,
                    return_sampling_mask=False, custom_logit_processor=None, grammar=None))
                count += len(case['expected_output_ids'])
                assert case['body']['sampling_params']['custom_params']['glm53_forced_decode_ids'] == case['expected_output_ids']
                assert 'custom_params' not in case['unscored_ordinary_warm_body']['sampling_params']
                assert case['expected_positions'] == {'EXTEND': [0], 'DECODE': list(range(len(case['expected_output_ids'])))[1:]}
            refs = reference.refs_from_reqs(requests, 154880)
            assert len(refs) == len(group) and all(ref is not None for ref in refs)
            assert all(len(ref.ids) == (32 if kind == 'scores' else 4) for ref in refs)
            assert [ref.request_sha256 for ref in refs] == [case['expected_request_sha256'] for case in group]
        assert count == data['committed_outputs'] == (320 if kind == 'scores' else 24)
        totals[kind] = count
    return totals


def qualify(source, fixtures=None, original_source=None):
    path = source / 'srt/sampling/glm53_forced_decode_reference.py'
    with patch.dict(os.environ, {'SGLANG_GLM53_FORCE_DECODE_REFERENCE': '1'}):
        reference = load(path, 'forced_reference_active')
    with patch.dict(os.environ, {'SGLANG_GLM53_FORCE_DECODE_REFERENCE': '0'}):
        off = load(path, 'forced_reference_default_off')
    assert off.refs_from_reqs([object()], 3) is None and off.prepare_reference(object(), object()) is None
    cls = test_metadata(source, reference)
    test_actual_sampler(source, reference, cls)
    test_actual_runner(source, reference, cls)
    if fixtures:
        test_frozen_requests(source, reference, fixtures, original_source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--patch', type=Path, default=Path(__file__).parents[1] / 'patches/0061-glm53-forced-decode-reference.patch')
    p.add_argument('--fixtures', type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='glm53-0061-cpu-') as scratch:
        source = args.source.resolve()
        if not (source / 'srt/sampling/glm53_forced_decode_reference.py').exists():
            copied = Path(scratch) / 'sglang'
            for name in ('srt/sampling/sampling_batch_info.py', 'srt/layers/sampler.py', 'srt/model_executor/model_runner.py'):
                dest = copied / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, dest)
            subprocess.run(['patch', '--batch', '-p1', '-i', str(args.patch.resolve())], cwd=scratch,
                check=True, stdout=subprocess.DEVNULL)
            source = copied
        qualify(source, args.fixtures, args.source.resolve())
    print('PASS0061: default-off/inactive sampler API; actual sampling metadata construction/filter/merge/forward copy; CPU seq_len positions despite delayed host output; prefill index0 and decode index>0; chunk/lookahead exclusion; draft/NEXTN/grammar restrictions; actual sampler raw logsoftmax before in-place softmax, ordinary sampling before forced IDs, forced-ID output gather before TP sync; actual ModelRunner ngram IDs. CPU only; no sampled-law claim.')


if __name__ == '__main__':
    main()
