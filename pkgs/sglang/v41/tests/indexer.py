#!/usr/bin/env python3
"""Independent literal CPU oracle for gfx1151 low-ratio FP4 paged indexing."""
import inspect
import json
import math
import os
from pathlib import Path
import sys

os.environ['SGLANG_DSV4_FP4_LOGITS_BUDGET_MB'] = '64'
os.environ['SGLANG_USE_AITER'] = '0'
import torch

LEVELS = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
MAX_ERROR = 0.
CASES = 0


def quantize_reference(x):
    """Per-32 ceil-power-of-two scale, nearest E2M1 with even-code ties."""
    x = x.cpu().float().reshape(*x.shape[:-1], 4, 32)
    sf = (x.abs().amax(-1) / 6).clamp_min(1.e-4)
    exponent = torch.ceil(torch.log2(sf.double())).clamp(-126, 127).to(torch.int64)
    scales = torch.exp2(exponent.double())
    normalized = x.double() / scales[..., None]
    distance = (normalized.abs()[..., None] - LEVELS).abs()
    best = distance.amin(-1, keepdim=True)
    # Even codes get the first chance at exact half-way ties.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    equal = (distance == best)[..., order]
    code = order[equal.to(torch.int32).argmax(-1)]
    code |= ((normalized < 0) & (code != 0)).to(torch.int64) * 8
    code = code.flatten(-2).to(torch.uint8)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, (exponent + 127).to(torch.uint8)


def unpack_reference(packed, scale):
    b = packed.cpu().to(torch.int64)
    c = torch.stack((b & 15, b >> 4), -1).flatten(-2)
    v = LEVELS[c & 7] * torch.where((c & 8) != 0, -1., 1.)
    return v * torch.exp2(scale.cpu().double() - 127).repeat_interleave(32, -1)


def expected_q_scale(scale):
    n, heads, _ = scale.shape
    out = torch.zeros((n, 1, 4, 16, 4), dtype=torch.uint8)
    for head in range(heads):
        out[:, 0, :, head % 16, head // 16] = scale[:, head]
    return out


def expected_cache(k, loc, pages):
    packed, sf = quantize_reference(k)
    payload = torch.zeros(pages, 1, 4, 64, 16, dtype=torch.uint8)
    scales = torch.zeros(pages, 1, 4, 64, dtype=torch.uint8)
    physical_values = torch.zeros(pages * 64, 128, dtype=torch.float64)
    values = unpack_reference(packed, sf)
    loc = loc.cpu().long()
    page, off = loc // 64, loc % 64
    payload[page, 0, :, off, :] = packed.reshape(-1, 4, 16)
    scales[page, 0, :, (off % 16) * 4 + off // 16] = sf
    physical_values[loc] = values
    return payload, scales, physical_values


def logits_reference(q, k_values, weights, pages, lengths, scale, width):
    qp, qs = quantize_reference(q)
    qv = unpack_reference(qp, qs)
    out = torch.full((q.shape[0], width), -torch.inf, dtype=torch.float64)
    for row, length in enumerate(lengths.cpu().tolist()):
        cols = torch.arange(length)
        slots = pages.cpu()[row, cols // 64].long() * 64 + cols % 64
        per_head = qv[row] @ k_values[slots].T
        out[row, :length] = (per_head.relu() * weights.cpu()[row].double()[:, None]).sum(0) * scale
    return out


def check_scores(got, expected):
    global MAX_ERROR
    got = got.cpu().double()
    finite = expected.isfinite()
    assert torch.equal(got.isneginf(), expected.isneginf())
    if finite.any():
        error = (got[finite] - expected[finite]).abs().max().item()
        MAX_ERROR = max(MAX_ERROR, error)
        torch.testing.assert_close(got[finite], expected[finite], rtol=2.e-5, atol=2.e-4)


def block_reference(scores, lengths, top_blocks, block_size):
    result = []
    for row, length in enumerate(lengths.cpu().tolist()):
        maxima = [scores[row, lo:min(lo + block_size, length)].max().item()
                  for lo in range(0, length, block_size)]
        if maxima:
            maxima[-1] = math.inf
        result.append(sorted(range(len(maxima)), key=lambda i: (-maxima[i], i))[:top_blocks])
    return result


def check_selection(scores, reference, lengths, page_table, top_blocks):
    global CASES
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        select_candidate_blocks_hip, topk_within_candidate_blocks_hip,
        topk_transform_paged_sorted, candidate_block_scores,
    )
    candidates = select_candidate_blocks_hip(scores, lengths, topk_blocks=top_blocks, block_size=8)
    ref_blocks = block_reference(reference, lengths, top_blocks, 8)
    block_scores = candidate_block_scores(scores, lengths, block_size=8, fill_tail=True).cpu()
    for row, (blocks, length) in enumerate(zip(ref_blocks, lengths.cpu().tolist())):
        selected = candidates.ids[row].cpu().tolist()
        actual = set(i for i in selected if i >= 0)
        expected = set(blocks)
        assert actual == expected, (row, 'candidate set', sorted(expected-actual)[:16], sorted(actual-expected)[:16])
        assert selected.count(-1) == top_blocks - len(blocks)
        assert candidates.compact_lens[row].item() == len(blocks) * 8
        assert torch.isneginf(block_scores[row, math.ceil(length / 8):]).all()
        if length:
            assert torch.isposinf(block_scores[row, (length - 1) // 8])
    for candidate_mode in (False, True):
        page_out = torch.empty(scores.shape[0], 512, dtype=torch.int32, device='cuda')
        raw_out = torch.empty_like(page_out)
        if candidate_mode:
            topk_within_candidate_blocks_hip(scores, lengths, candidates, page_table=page_table,
                page_size=64, page_indices=page_out, raw_indices=raw_out, sort_output=True)
        else:
            topk_transform_paged_sorted(scores, lengths, page_table, page_out, 64, raw_out)
        expected_raw = torch.full_like(raw_out.cpu(), -1)
        expected_page = torch.full_like(page_out.cpu(), -1)
        for row, length in enumerate(lengths.cpu().tolist()):
            reachable = list(range(length)) if not candidate_mode else [
                p for b in ref_blocks[row] for p in range(b * 8, min((b + 1) * 8, length))]
            chosen = sorted(sorted(reachable, key=lambda p: (-reference[row, p].item(), p))[:512])
            expected_raw[row, :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
            for j, p in enumerate(chosen):
                expected_page[row, j] = page_table[row, p // 64].item() * 64 + p % 64
        assert torch.equal(raw_out.cpu(), expected_raw), ('raw selection', candidate_mode)
        assert torch.equal(page_out.cpu(), expected_page), ('paged selection', candidate_mode)
        CASES += 1
    return candidates


def fixture(lengths_list, heads, seed):
    torch.manual_seed(seed)
    n = len(lengths_list)
    cols = math.ceil(max(1, max(lengths_list)) / 64)
    pages = n * cols
    q = torch.randn(n, heads, 128, dtype=torch.bfloat16)
    k = torch.randn(pages * 64, 128, dtype=torch.bfloat16)
    # Explicit ties at a known .125 scale, an all-zero scale group, and non-power-two maxima.
    boundary = torch.tensor([6., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., -.25, -.75, -1.25,
                             -1.75, -2.5, -3.5, -5., 0.] * 2).bfloat16() * .125
    q[0, 0, :32] = boundary
    q[0, 0, 32:64] = 0
    k[0, :32] = boundary
    k[0, 32:64] = 0
    q[-1, -1, -1] = 7.25
    k[-1, -1] = 13.5
    if seed == 71:
        q.zero_()  # Entire zero query, including every scale group.
    loc = torch.randperm(pages * 64).long()
    table = torch.randperm(pages).reshape(n, cols).int()
    weights = torch.randn(n, heads, dtype=torch.bfloat16)
    lens = torch.tensor(lengths_list, dtype=torch.int32)
    return [t.cuda() for t in (q, k, loc, table, weights, lens)], pages


def run_case(lengths_list, heads, seed, graph=False):
    global CASES
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        pack_fp4_query_flydsl, store_fp4_index_k_cache_split, aiter_fp4_paged_mqa_logits,
        prepare_fp4_decode_workspace, prepare_fp4_prefill_workspace,
    )
    tensors, pages = fixture(lengths_list, heads, seed)
    q, k, loc, table, weights, lens = tensors
    payload = torch.zeros(pages, 1, 4, 64, 16, dtype=torch.uint8, device='cuda')
    scales = torch.zeros(pages, 1, 4, 64, dtype=torch.uint8, device='cuda')
    scale = 0.37
    top_blocks = 2048 if max(lengths_list) > 16384 else 128

    def step(decode):
        qp, qs = pack_fp4_query_flydsl(q)
        store_fp4_index_k_cache_split(k, payload, scales, loc, page_size=64, rne=True)
        ws = prepare_fp4_decode_workspace(table, lens, page_table_bucket=64) if decode else None
        scores = aiter_fp4_paged_mqa_logits(q_fp4=qp, q_scale=qs, k_payload=payload,
            k_scale=scales, weights=weights, page_table=table, c4_seq_lens=lens,
            weight_scale=scale, is_decode=decode, decode_workspace=ws,
            prefill_workspace=None if decode else prefill_ws, page_table_bucket=64)
        return qp, qs, scores

    def verify(result, selection=True):
        global CASES
        qp, qs, scores = result
        ref_qp, ref_qs = quantize_reference(q)
        assert torch.equal(qp.cpu().view(torch.uint8), ref_qp)
        assert torch.equal(qs.cpu(), expected_q_scale(ref_qs))
        ref_payload, ref_scales, physical = expected_cache(k, loc, pages)
        assert torch.equal(payload.cpu(), ref_payload), 'K payload byte mismatch'
        assert torch.equal(scales.cpu(), ref_scales), 'K scale byte mismatch'
        expected = logits_reference(q, physical, weights, table, lens, scale, scores.shape[1])
        check_scores(scores, expected)
        if selection:
            check_selection(scores, expected, lens, table, top_blocks)
        CASES += 1
        return expected

    prefill_ws = prepare_fp4_prefill_workspace(table, lens, page_table_bucket=64)
    for decode in (True, False):
        result = step(decode)
        verify(result)
        print(json.dumps({'event': 'case', 'lengths': lengths_list, 'heads': heads,
                          'decode': decode, 'max_abs_error': MAX_ERROR}), flush=True)
    # Same shape, new pages and lengths refresh the existing prefill workspace.
    table.copy_(table.flip(1))
    lens.copy_(torch.tensor(list(reversed(lengths_list)), device='cuda', dtype=torch.int32))
    prefill_ws = prepare_fp4_prefill_workspace(table, lens, workspace=prefill_ws, page_table_bucket=64)
    verify(step(False))
    if graph:
        from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
            select_candidate_blocks_hip, topk_within_candidate_blocks_hip)
        page_out = torch.empty(q.shape[0], 512, dtype=torch.int32, device='cuda')
        raw_out = torch.empty_like(page_out)
        def captured_step():
            result = step(True)
            candidates = select_candidate_blocks_hip(result[2], lens, topk_blocks=top_blocks, block_size=8)
            topk_within_candidate_blocks_hip(result[2], lens, candidates, page_table=table,
                page_size=64, page_indices=page_out, raw_indices=raw_out, sort_output=True)
            return result, candidates
        captured_step()
        torch.cuda.synchronize()
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj):
            result, candidates = captured_step()
        for replay in range(2):
            q.copy_(torch.randn_like(q) * (replay + .7))
            k.copy_(torch.randn_like(k) * (replay + 1.3))
            weights.copy_(torch.randn_like(weights))
            table.copy_(table.flip(1))
            lens.copy_(torch.tensor(lengths_list if replay else list(reversed(lengths_list)), device='cuda', dtype=torch.int32))
            graph_obj.replay()
            torch.cuda.synchronize()
            expected = verify(result)
            expected_blocks = block_reference(expected, lens, top_blocks, 8)
            for row, length in enumerate(lens.cpu().tolist()):
                assert set(candidates.ids[row].cpu().tolist()) - {-1} == set(expected_blocks[row])
                reach = [p for b in expected_blocks[row] for p in range(b*8,min((b+1)*8,length))]
                chosen = sorted(sorted(reach, key=lambda p: (-expected[row,p].item(),p))[:512])
                assert raw_out[row, :len(chosen)].cpu().tolist() == chosen
                assert raw_out[row, len(chosen):].eq(-1).all()
                got_pages = [table[row,p//64].item()*64+p%64 for p in chosen]
                assert page_out[row,:len(chosen)].cpu().tolist() == got_pages
                assert page_out[row,len(chosen):].eq(-1).all()
            print(json.dumps({'event':'graph_replay','changed': ['q','k','weights','page_table','lengths'], 'replay':replay}),flush=True)


def main():
    import sglang
    import sgl_kernel
    import sglang.srt.mem_cache.rust_tree_core.mem_cache as rust_tree
    import sglang.srt.rust_extensions._multimodal as rust_multimodal
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import aiter_fp4_paged_mqa_logits
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_gfx1151 import paged_fp4_logits
    runtime = Path(sys.argv[1]).resolve()
    for item in (sglang, rust_tree, rust_multimodal, aiter_fp4_paged_mqa_logits, paged_fp4_logits):
        assert Path(inspect.getfile(item)).resolve().is_relative_to(runtime)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(':')[0] == 'gfx1151'
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.03)
    print(json.dumps({'event':'provenance','runtime':str(runtime),'arch':props.gcnArchName,
                      'sgl_kernel':sgl_kernel.__file__,'torch':torch.__version__}),flush=True)
    run_case([65],16,71)
    run_case([0,1,511,513,641],32,73)
    run_case([17,16385,16449],32,79,graph=True)
    run_case([0,513,1537],32,83,graph=True)
    torch.cuda.synchronize()
    print(json.dumps({'event':'complete','cases':CASES,'max_abs_error':MAX_ERROR,
                      'rtol':2.e-5,'atol':2.e-4,'packing':'byte-exact','selection':'exact',
                      'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
                      'gpu_peak_reserved':torch.cuda.max_memory_reserved()}),flush=True)

if __name__ == '__main__':
    main()
