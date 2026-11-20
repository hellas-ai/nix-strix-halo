#!/usr/bin/env bash
# Build 0049-exact-decode-graph-buckets.patch from the scratch tree (git repo whose HEAD is upstream + patches 0001..0045).
# Plain unified diffs (a/ b/ labels, no git headers), the format of the other patches in pkgs/sglang/v41/patches.
set -euo pipefail
TREE=${TREE:-/mnt/Home/src/nix-strix-halo-exactrows-scratch/tree-t4}
OUT=${1:-/mnt/Home/src/nix-strix-halo-exactrows-t4/pkgs/sglang/v41/patches/0049-exact-decode-graph-buckets.patch}
cd "$TREE"
: > "$OUT"
git -c core.quotepath=off diff --name-only HEAD | sort | while read -r f; do
  git show "HEAD:$f" | diff -u --label "a/$f" --label "b/$f" - "$f" >> "$OUT" || true
done
git ls-files --others --exclude-standard | grep -v '\.orig$\|\.rej$\|__pycache__' | sort | while read -r f; do
  diff -u --label /dev/null --label "b/$f" /dev/null "$f" >> "$OUT" || true
done
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
