#!/usr/bin/env python3
"""Parse ds41-node journal text for decode graph evidence.  CPU only, read-only, stdlib only.

  graph_logs.py capture JOURNAL...   per rank and graph kind: bucket list, rows per request, elapsed, graph memory, avail mem
  graph_logs.py replay  JOURNAL...   (needs SGLANG_LOG_DECODE_GRAPH_KEY=1) count replays per (worker, mode, raw_bs, key_size);
                                     padded = key_size != raw_bs.  Exit 1 with --require-exact if any replay is padded.

The input is the journal text of ds4-serve (journalctl / the per-run slices under the qualification artifacts), one line per
record, e.g.
  2026-10-02T21:06:38+02:00 strix-3 ds41-node[576244]: [2026-10-02 21:06:38 TP0] Capture target verify CUDA graph end. elapsed=156.56 s, mem usage=1.88 GB, avail mem=24.80 GB.
  ... [... TP0] Decode graph replay: worker=target key_size=4 (bs) mode=TARGET_VERIFY raw_bs=3
"""
import argparse
import collections
import re
import sys

BEGIN = re.compile(
    r"^(?P<ts>\S+) (?P<host>\S+) .*?\[(?P<clock>[^\]]*?) (?P<tp>TP\d+)\] Capture (?P<role>target|draft) (?P<kind>decode|verify) "
    r"(?:CUDA|HIP) graph begin\. backend=(?P<backend>\S+), num_tokens_per_req=(?P<width>\d+), bs=\[(?P<bs>[\d, ]*)\], "
    r"avail mem=(?P<avail>[\d.]+) GB"
)
END = re.compile(
    r"^(?P<ts>\S+) (?P<host>\S+) .*?\[(?P<clock>[^\]]*?) (?P<tp>TP\d+)\] Capture (?P<role>target|draft) (?P<kind>decode|verify) "
    r"(?:CUDA|HIP) graph end\. elapsed=(?P<elapsed>[\d.]+) s, mem usage=(?P<mem>-?[\d.]+) GB, avail mem=(?P<avail>[\d.]+) GB"
)
REPLAY = re.compile(
    r"\[(?P<clock>[^\]]*?) (?P<tp>TP\d+)\] Decode graph replay: worker=(?P<worker>\w+) key_size=(?P<key>\d+) \((?P<axis>\w+)\) "
    r"mode=(?P<mode>\w+) raw_bs=(?P<raw>\d+)(?: slots=(?P<slots>\d+))?"
)


def lines(paths):
    for path in paths:
        with open(path, errors="replace") as handle:
            for line in handle:
                yield path, line.rstrip("\n")


def capture(paths):
    open_ = {}
    rows = []
    for path, line in lines(paths):
        m = BEGIN.search(line)
        if m:
            open_[(path, m["host"], m["tp"], m["role"], m["kind"])] = m
            continue
        m = END.search(line)
        if m:
            key = (path, m["host"], m["tp"], m["role"], m["kind"])
            b = open_.pop(key, None)
            rows.append(
                dict(
                    host=m["host"], tp=m["tp"], role=m["role"], kind=m["kind"],
                    bs=b["bs"].replace(" ", "") if b else "?", width=int(b["width"]) if b else None,
                    elapsed_s=float(m["elapsed"]), mem_gb=float(m["mem"]), avail_gb=float(m["avail"]),
                )
            )
    print(f"{'host':8} {'tp':4} {'graph':14} {'bs':14} {'rows/req':8} {'elapsed s':>9} {'mem GB':>7} {'avail GB':>8}")
    for r in rows:
        print(f"{r['host']:8} {r['tp']:4} {r['role'] + ' ' + r['kind']:14} [{r['bs']}]".ljust(32)
              + f" {r['width'] if r['width'] is not None else '?':<8} {r['elapsed_s']:>9.2f} {r['mem_gb']:>7.2f} {r['avail_gb']:>8.2f}")
    if not rows:
        print("no capture end lines found", file=sys.stderr)
    return rows


def replay(paths, require_exact):
    counts = collections.Counter()
    for _path, line in lines(paths):
        m = REPLAY.search(line)
        if m and m["tp"] == "TP0":                      # one record per replay: rank 0 is enough
            counts[(m["worker"], m["mode"], int(m["raw"]), int(m["key"]))] += 1
    padded = 0
    print(f"{'worker':7} {'mode':14} {'raw_bs':>6} {'key_size':>8} {'replays':>8}  verdict")
    for (worker, mode, raw, key), n in sorted(counts.items()):
        verdict = "exact" if raw == key else f"PADDED (+{key - raw})"
        padded += n if raw != key else 0
        print(f"{worker:7} {mode:14} {raw:>6} {key:>8} {n:>8}  {verdict}")
    print(f"replays={sum(counts.values())} padded={padded}")
    if not counts:
        print("no 'Decode graph replay' lines: was SGLANG_LOG_DECODE_GRAPH_KEY=1 set?", file=sys.stderr)
        return 2
    return 1 if (require_exact and padded) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("journal", nargs="+")
    r = sub.add_parser("replay")
    r.add_argument("journal", nargs="+")
    r.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    if args.cmd == "capture":
        return 0 if capture(args.journal) else 1
    return replay(args.journal, args.require_exact)


if __name__ == "__main__":
    sys.exit(main())
