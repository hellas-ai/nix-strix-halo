"""Offline HTTP-validator regression: widths/metrics are explicit across drafters."""
import copy
import importlib.util
from pathlib import Path

path = Path(__file__).with_name('spec-graph-http-smoke.py')
spec = importlib.util.spec_from_file_location('smoke', path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def fixture(width, label='single-sampled'):
    ids = [30, 31, 32, 33]
    histogram = [0] * (width + 1)
    histogram[width] = 3
    proposed = 3 * (width - 1)
    return {'label': label, 'requested': 4, 'start': 0, 'end': 1,
        'response': {'output_ids': ids, 'meta_info': {
            'completion_tokens': 4, 'output_token_logprobs': [(-.2, token, None) for token in ids],
            'output_top_logprobs': [[(-.2, token, None), (-1.5, token + 10, None), (-3., token + 20, None)] for token in ids],
            'spec_verify_ct': 3, 'spec_num_correct_drafts': 1, 'spec_num_proposed_drafts': proposed,
            'spec_accept_rate': 1 / proposed, 'spec_cap_lens_histogram': histogram}}}


def must_reject(record, widths):
    try:
        smoke.validate(record, True, widths)
    except (AssertionError, KeyError):
        return
    raise AssertionError('Validator accepted invalid width accounting')


for width, policy in ((4, (4, 4, 2, 2)), (8, (8, 4, 4, 2)), (16, (16, 16, 16, 16))):
    result = smoke.validate(fixture(width), True, policy)
    assert result['spec']['spec_num_proposed_drafts'] == 3 * (width - 1)
must_reject(fixture(4), (8, 4, 4, 2))  # Native-MTP C1 expectation must not leak into DFlash.
must_reject(fixture(8), (4, 4, 2, 2))
must_reject(fixture(8), None)
wrong = fixture(8)
wrong['response']['meta_info']['spec_num_proposed_drafts'] = 3 * (16 - 1)
must_reject(wrong, (8, 4, 4, 2))  # Draft computes full block16 but target verifies width8.
mixed = fixture(4, 'stagger-fourth')
mixed['response']['meta_info'].update(spec_cap_lens_histogram=[0, 0, 1, 0, 2],
    spec_num_proposed_drafts=7, spec_accept_rate=1 / 7)
smoke.validate(mixed, True, (8, 4, 4, 2))
wrong = copy.deepcopy(mixed)
wrong['response']['meta_info']['spec_cap_lens_histogram'][2] = -1
must_reject(wrong, (8, 4, 4, 2))
plain = fixture(4)
for key in list(plain['response']['meta_info']):
    if key.startswith('spec_'):
        plain['response']['meta_info'].pop(key)
smoke.validate(plain, False)
assert smoke.parse_widths('8,4,4,2') == (8, 4, 4, 2)
print('PASS: explicit NEXTN4/DFlash8/full-block16 policies; adaptive width accounting; reject cross-algorithm and full-draft-denominator errors')
