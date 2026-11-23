#!/usr/bin/env bash
# usage: mk_overlay.sh <patched-tree> <overlay-dir>
# Builds <overlay>/lib/python3.13/site-packages/sglang = installed runtime B's sglang package with every file that
# differs between the tree's baseline commit and its working tree replaced by the tree's version.
set -euo pipefail
TREE=$1; OV=$2; BASE=${3:-727eb3b}   # BASE: the series-0045 commit of the scratch tree
RT=/nix/store/5qsmp69vmra8f6qw1nbp76qfhck93zdy-sglang-v41-rocm-gfx1151-0.5.20.post1.dev41308
SP=$OV/lib/python3.13/site-packages
if [ ! -d "$SP/sglang" ]; then
  mkdir -p "$SP"
  cp -r "$RT/lib/python3.13/site-packages/sglang" "$SP/sglang"
  chmod -R u+w "$SP/sglang"
fi
cd "$TREE"
{ git diff --name-only "$BASE"; git ls-files --others --exclude-standard; } | sort -u | while read -r f; do
  case "$f" in sglang/*) mkdir -p "$SP/$(dirname "$f")"; cp -f "$TREE/$f" "$SP/$f";; esac
done
echo "overlay ready: $SP"
