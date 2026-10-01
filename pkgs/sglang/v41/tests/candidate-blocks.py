"""Bounded level-one block cutoff test; literal oracle, no model or capture data."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def make_logits(lengths, width, kind="tie"):
    rows = np.zeros((len(lengths), width), dtype=np.float32)
    for r, length in enumerate(lengths):
        for b in range((length + 7) // 8):
            value = 2.0 if b < 2046 else (1.0 if b in (2046, 2047) else 0.0)
            if kind == "all-equal":
                value = 0.0
            if kind == "signed-zero":
                value = -0.0 if b % 2 else 0.0
            if kind == "strict" and b == 2047:
                value = np.nextafter(np.float32(1), np.float32(2))
            rows[r, b * 8 : min(b * 8 + 8, length)] = value
        rows[r, length:] = 12345.0  # Future positions must never enter block maxima.
    return rows


def oracle(row, length):
    maxima = [
        max(map(float, row[lo : min(lo + 8, length)])) for lo in range(0, length, 8)
    ]
    if maxima:
        maxima[-1] = math.inf
    chosen = sorted(range(len(maxima)), key=lambda b: (-maxima[b], b))[:2048]
    chosen = [b for b in chosen if maxima[b] != -math.inf]
    return chosen, maxima


def check(ids, compact_lens, rows, lengths, label):
    ids = np.asarray(ids)
    compact_lens = np.asarray(compact_lens)
    for r, length in enumerate(lengths):
        expected, _maxima = oracle(rows[r], int(length))
        actual = ids[r].tolist()
        assert actual == expected + [-1] * (2048 - len(expected)), (
            label,
            r,
            actual[:8],
            expected[:8],
        )
        valid = [x for x in actual if x >= 0]
        assert len(valid) == len(set(valid))
        assert all(x < (length + 7) // 8 for x in valid)
        assert int(compact_lens[r]) == min((int(length) + 7) // 8, 2048) * 8
        if length:
            assert (length - 1) // 8 in valid, "Newest block missing"
        if label.startswith("tie") and length > 16392:
            assert 2046 in valid and 2047 not in valid
    emit("case", label=label, rows=len(lengths), ordered_ids_exact=True)


def cpu_checks():
    # Analytical expectations validate the independent tuple-sort reference;
    # CPU mode deliberately does not emulate the installed selection algorithm.
    checks = 0
    lengths = [16400, 16393, 513, 0]
    for kind in ("tie", "strict", "all-equal", "signed-zero"):
        rows = make_logits(lengths, 20480, kind)
        for row, length in zip(rows, lengths):
            selected, maxima = oracle(row, length)
            assert len(selected) == min((length + 7) // 8, 2048)
            assert len(selected) == len(set(selected))
            if length:
                newest = (length - 1) // 8
                assert selected[0] == newest and maxima[newest] == math.inf
                if length > 16392:
                    winner = 2047 if kind == "strict" else 2046
                    loser = 2046 if kind == "strict" else 2047
                    assert winner in selected and loser not in selected
                assert all(b * 8 < length for b in selected)
            else:
                assert selected == []
            checks += 1
    # The forced partial newest block survives even when other blocks are masked.
    selected, _ = oracle(np.full(17, -np.inf, np.float32), 17)
    assert selected == [2]
    checks += 1
    assert not torch.cuda.is_initialized()
    emit("cpu-reference", checks=checks, cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    torch.set_num_threads(1)
    cpu_checks()
    if args.cpu_only:
        emit(
            "complete",
            scope="Independent CPU reference checks only",
            gpu_qualified=False,
        )
        return
    torch.cuda.set_per_process_memory_fraction(0.03)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1151"
    from sglang.kernels.ops.attention.dsv4 import candidate_blocks_hip as cb

    source = Path(cb.__file__).resolve()
    assert source.is_relative_to(runtime), str(source)
    emit("installed-source", runtime=str(runtime), path=str(source))
    torch.cuda.reset_peak_memory_stats()
    cases = 0

    def step(scores, lens):
        return cb.select_candidate_blocks_hip(
            scores, lens, topk_blocks=2048, block_size=8
        )

    def run(lengths, width, kind, label):
        nonlocal cases
        rows = make_logits(lengths, width, kind)
        backing = torch.zeros((len(lengths), width + 16), device="cuda")
        scores = backing[:, :width]
        scores.copy_(torch.from_numpy(rows))
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        result = step(scores, lens)
        check(
            result.ids.cpu().numpy(),
            result.compact_lens.cpu().numpy(),
            rows,
            lengths,
            label,
        )
        assert np.array_equal(
            scores.cpu().numpy().view(np.uint32), rows.view(np.uint32)
        )
        assert (
            result.block_size == 8
            and result.compact_page_size == 16384
            and bool((result.compact_page_table == 0).all())
        )
        cases += 1

    for repeat in range(4):
        run([16400, 16393, 513], 20480, "tie", "tie-repeat-" + str(repeat))
    for batch in (1, 2):
        run([16400, 16393][:batch], 20480, "tie", "tie-batch-" + str(batch))
    for kind in ("strict", "all-equal", "signed-zero"):
        run([16400, 16393, 513], 20480, kind, kind)
    run([0, 1, 511], 4096, "tie", "small-masks")
    run([131072], 131072, "tie", "tie-128k")
    run([131072], 131072, "all-equal", "all-equal-128k")
    lengths = [16400, 16393, 513]
    rows = make_logits(lengths, 20480, "tie")
    scores = torch.tensor(rows, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    for _ in range(3):
        step(scores, lens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step(scores, lens)
    for phase in ("A", "B", "A"):
        lengths = [16400, 16393, 513] if phase == "A" else [16393, 16400, 521]
        rows = make_logits(lengths, 20480, "tie" if phase == "A" else "strict")
        scores.copy_(torch.from_numpy(rows))
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        check(
            captured.ids.cpu().numpy(),
            captured.compact_lens.cpu().numpy(),
            rows,
            lengths,
            "graph-" + phase,
        )
        cases += 1
    assert cases == 15
    emit(
        "complete",
        scope="installed level-one selector; no full-model quality claim",
        cases=cases,
        exact_logical_block_policy=True,
        changed_graph=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


if __name__ == "__main__":
    main()
