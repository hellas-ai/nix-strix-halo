"""Bounded installed-native logical-cutoff check; no model or network access."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CANDIDATE_IDS = list(range(282))


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def score_keys(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return np.where(bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000))


def reference(scores, length, blocks=None, block_size=8, k=512):
    ids = (
        list(range(length))
        if blocks is None
        else [
            int(b) * block_size + j
            for b in blocks
            if b >= 0
            for j in range(block_size)
            if int(b) * block_size + j < length
        ]
    )
    assert len(ids) == len(set(ids))
    keys = score_keys(scores)
    selected = sorted(ids, key=lambda i: (-int(keys[i]), i))[:k]
    return np.array(sorted(selected) + [-1] * (k - len(selected)), dtype=np.int32)


def fixture_values():
    """511 strict winners and two exact cutoff ties in different candidate blocks."""
    values = np.full(2304, -4.0, np.float32)
    values[:511] = 1 + np.arange(511, dtype=np.float32) / 1024
    values[1923] = values[2080] = np.float32(-2.5)
    return values


def cpu_checks():
    scores = fixture_values()
    expected = reference(scores, 2251)
    assert 1923 in expected and 2080 not in expected
    keys = score_keys(scores[:2251])
    cutoff = int(keys[expected].min())
    strict = int((keys > cutoff).sum())
    ties = np.flatnonzero(keys == cutoff)
    assert strict == 511 and ties.tolist() == [1923, 2080]
    rng = np.random.default_rng(173205)
    checks = 0
    for i in range(12):
        blocks = np.asarray(CANDIDATE_IDS, np.int32).copy()
        rng.shuffle(blocks)
        assert np.array_equal(reference(scores, 2251, blocks), expected)
        checks += 1
    # Exact native bit-key ordering, signed zero and neighboring finite values.
    values = np.array(
        [-np.inf, -3.0, -0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)), np.inf],
        np.float32,
    )
    assert score_keys(values).tolist() == sorted(score_keys(values).tolist())
    checks += 1
    # Original causal length excludes newest-block padding even when valid scores are -inf.
    values = np.full(520, -np.inf, np.float32)
    blocks = np.arange(65, dtype=np.int32)[::-1]
    assert np.array_equal(
        reference(values, 513, blocks), np.arange(512, dtype=np.int32)
    )
    checks += 1
    for n in (5121, 8192, 131072):
        values = (1 + np.arange(n, dtype=np.float64) / 65536).astype(np.float32)
        assert reference(values, n).tolist() == list(range(n - 512, n))
        checks += 1
        half = values.astype(np.float16).view(np.uint16)
        if n <= 8192:
            assert len(set(((half | np.uint16(0x8000)) >> np.uint16(8)).tolist())) == 1
    assert not torch.cuda.is_initialized()
    emit(
        "cpu-reference",
        checks=checks,
        synthetic_ties=ties.tolist(),
        chosen=1923,
        cuda_initialized=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    expected_runtime = args.runtime.resolve(strict=True)
    torch.set_num_threads(1)
    cpu_checks()
    import sgl_kernel

    schema = str(torch.ops.sgl_kernel.deepseek_v4_topk_transform_512.default._schema)
    assert (
        "tie_break_ids" in schema
        and "tie_break_lens" in schema
        and "tie_break_block_size" in schema
    ), schema
    emit(
        "installed-schema",
        schema=schema,
        sgl_kernel=str(Path(sgl_kernel.__file__).resolve()),
    )
    if args.cpu_only:
        assert not torch.cuda.is_initialized()
        emit(
            "complete",
            scope="CPU references and real installed native registration",
            gpu_qualified=False,
        )
        return
    torch.cuda.set_per_process_memory_fraction(0.03)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    from sglang.kernels.ops.attention.dsv4 import candidate_blocks_hip as cb

    installed = Path(cb.__file__).resolve()
    assert installed.is_relative_to(expected_runtime), str(installed)
    emit(
        "runtime",
        candidate_module=str(installed),
        device=torch.cuda.get_device_name(),
        kernel_file=str(Path(sgl_kernel.__file__).resolve()),
    )
    rng = np.random.default_rng(173205)
    cases = []
    device = "cuda"
    page_size = 64

    def check_outputs(label, score_rows, lens, blocks, page_table, raw, paged):
        actual = raw.cpu().numpy()
        actual_paged = paged.cpu().numpy()
        pt = page_table.cpu().numpy()
        expected = np.stack(
            [
                reference(s, int(n), None if blocks is None else blocks[r])
                for r, (s, n) in enumerate(zip(score_rows, lens))
            ]
        )
        expected_paged = np.full_like(expected, -1)
        for r in range(len(lens)):
            ok = expected[r] >= 0
            x = expected[r, ok]
            expected_paged[r, ok] = pt[r, x // page_size] * page_size + x % page_size
        if not np.array_equal(actual, expected) or not np.array_equal(
            actual_paged, expected_paged
        ):
            emit(
                "failure-values",
                label=label,
                raw=actual.tolist(),
                expected=expected.tolist(),
                paged=actual_paged.tolist(),
                expected_paged=expected_paged.tolist(),
            )
        assert np.array_equal(actual, expected), (label, "raw")
        assert np.array_equal(actual_paged, expected_paged), (label, "paged")
        cases.append(label)
        emit(
            "case",
            label=label,
            rows=len(lens),
            width=score_rows.shape[1],
            logical_membership_exact=True,
        )

    def run(label, score_rows, lens, blocks=None):
        score_rows = np.ascontiguousarray(score_rows, np.float32)
        lens = np.asarray(lens, np.int32)
        # Row stride wider than visible scores exercises native stride handling.
        backing = torch.full((len(lens), score_rows.shape[1] + 16), 91.0, device=device)
        scores = backing[:, : score_rows.shape[1]]
        scores.copy_(torch.from_numpy(score_rows))
        lengths = torch.tensor(lens, device=device)
        pages = (score_rows.shape[1] + 63) // 64
        pt = torch.tensor(
            np.stack([rng.permutation(pages).astype(np.int32) + 7 for _ in lens]),
            device=device,
        )
        raw = torch.empty((len(lens), 512), dtype=torch.int32, device=device)
        paged = torch.empty_like(raw)
        if blocks is None:
            torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
                scores, lengths, pt, paged, page_size, raw, True
            )
        else:
            ids = np.full((len(lens), 2048), -1, np.int32)
            for r, bs in enumerate(blocks):
                ids[r, : len(bs)] = bs
            ids_gpu = torch.tensor(ids, device=device)
            compact_lens = torch.tensor(
                [min((int(n) + 7) // 8, 2048) * 8 for n in lens],
                dtype=torch.int32,
                device=device,
            )
            candidates = cb.CandidateBlocks(
                ids_gpu,
                compact_lens,
                torch.zeros((len(lens), 1), dtype=torch.int32, device=device),
                16384,
                8,
            )
            cb.topk_within_candidate_blocks_hip(
                scores,
                lengths,
                candidates,
                page_table=pt,
                page_size=page_size,
                page_indices=paged,
                raw_indices=raw,
                sort_output=True,
            )
            assert np.array_equal(ids_gpu.cpu().numpy(), ids)
        check_outputs(label, score_rows, lens, blocks, pt, raw, paged)
        assert np.array_equal(
            scores.cpu().numpy().view(np.uint32), score_rows.view(np.uint32)
        )
        assert bool((backing[:, -16:] == 91.0).all())

    actual = fixture_values()
    block_count = (2251 + 7) // 8
    for i in range(6):
        perm = (
            np.asarray(CANDIDATE_IDS, np.int32)[:block_count].copy()
            if i == 0
            else rng.permutation(block_count).astype(np.int32)
        )
        run("synthetic-cutoff-permutation-" + str(i), actual[None, :], [2251], [perm])
    run("synthetic-cutoff-source", actual[None, :], [2251])
    mixed = np.stack([actual, actual, actual])
    run(
        "oddbatch-varying-masks",
        mixed,
        [2251, 2189, 513],
        [rng.permutation((n + 7) // 8).astype(np.int32) for n in (2251, 2189, 513)],
    )
    all_equal = np.full((3, 2304), -2.5, np.float32)
    run("equal-source", all_equal, [2304, 2251, 513])
    run(
        "equal-permuted-candidates",
        all_equal,
        [2304, 2251, 513],
        [rng.permutation((n + 7) // 8).astype(np.int32) for n in (2304, 2251, 513)],
    )
    negative_inf = np.full((1, 520), -np.inf, np.float32)
    run(
        "negative-infinity-masked-newest",
        negative_inf,
        [513],
        [np.arange(65, dtype=np.int32)[::-1]],
    )
    signed_zero = np.full((1, 1024), -0.0, np.float32)
    signed_zero[0, :311] = 1.0
    signed_zero[0, 311:611] = 0.0
    run("signed-zero-score-key", signed_zero, [1024])
    unique = rng.standard_normal((3, 4096)).astype(np.float32)
    run("unique-oracle", unique, [4096, 2048, 511])
    for n in (0, 1, 511, 512):
        run("identity-" + str(n), np.zeros((1, 512), np.float32), [n])
    # Unique finite winners expose truncation; all-equal also exercises tie refinement after overflow.
    for n in (5121, 8192, 131072):
        ramp = (1 + np.arange(n, dtype=np.float64) / 65536).astype(np.float32)
        run("overflow-finite-ramp-" + str(n), ramp[None, :], [n])
    run("128k-all-equal-logical-ties", np.zeros((1, 131072), np.float32), [131072])
    # Captured consumer replay changes score bytes, causal lengths, publication order and physical pages.
    lens_np = np.array([2251, 2189, 513], np.int32)
    scores_np = np.full((3, 8192), -np.inf, np.float32)
    scores_np[:, : len(actual)] = actual
    scores = torch.tensor(scores_np, device=device)
    lengths = torch.tensor(lens_np, device=device)
    ids_np = np.full((3, 2048), -1, np.int32)
    for r, n in enumerate(lens_np):
        ids_np[r, : (n + 7) // 8] = rng.permutation((n + 7) // 8)
    ids = torch.tensor(ids_np, device=device)
    compact_lens = torch.tensor(((lens_np + 7) // 8) * 8, device=device)
    pt_np = np.stack([rng.permutation(128).astype(np.int32) + 9 for _ in lens_np])
    pt = torch.tensor(pt_np, device=device)
    raw = torch.empty((3, 512), dtype=torch.int32, device=device)
    paged = torch.empty_like(raw)
    candidate = cb.CandidateBlocks(
        ids,
        compact_lens,
        torch.zeros((3, 1), dtype=torch.int32, device=device),
        16384,
        8,
    )

    def step():
        cb.topk_within_candidate_blocks_hip(
            scores,
            lengths,
            candidate,
            page_table=pt,
            page_size=64,
            page_indices=paged,
            raw_indices=raw,
            sort_output=True,
        )

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    original = (scores_np.copy(), lens_np.copy(), ids_np.copy(), pt_np.copy())
    for phase in ("A", "B", "A"):
        if phase == "B":
            scores_np = np.broadcast_to(
                (1 + np.arange(8192, dtype=np.float64) / 65536).astype(np.float32),
                (3, 8192),
            ).copy()
            lens_np = np.array([8192, 5121, 6000], np.int32)
            ids_np = np.full((3, 2048), -1, np.int32)
            for r, n in enumerate(lens_np):
                ids_np[r, : (n + 7) // 8] = rng.permutation((n + 7) // 8)
            pt_np = np.stack(
                [rng.permutation(128).astype(np.int32) + 19 for _ in lens_np]
            )
        else:
            scores_np, lens_np, ids_np, pt_np = (x.copy() for x in original)
        scores.copy_(torch.tensor(scores_np))
        lengths.copy_(torch.tensor(lens_np))
        ids.copy_(torch.tensor(ids_np))
        compact_lens.copy_(torch.tensor(((lens_np + 7) // 8) * 8))
        pt.copy_(torch.tensor(pt_np))
        graph.replay()
        torch.cuda.synchronize()
        check_outputs("graph-" + phase, scores_np, lens_np, ids_np, pt, raw, paged)
    torch.cuda.synchronize()
    emit(
        "complete",
        scope="native top512 ties and coarse-overflow regression",
        cases=len(cases),
        logical_membership_exact=True,
        graph_changed_data=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


if __name__ == "__main__":
    main()
