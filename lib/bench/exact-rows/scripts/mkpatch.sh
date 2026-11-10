#!/usr/bin/env bash
# usage: mkpatch.sh <tree> <out.patch> [pathspec...]   -- diff of the tree against its baseline commit in this repo's patch style
set -euo pipefail
TREE=$1; OUT=$2; shift 2
cd "$TREE"
git ls-files --others --exclude-standard "$@" | while read -r f; do git add -N "$f"; done
git -c core.quotepath=off diff --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/ HEAD -- "${@:-.}" \
  | grep -v '^diff --git \|^index \|^new file mode\|^deleted file mode\|^similarity index\|^rename from\|^rename to\|^old mode\|^new mode' > "$OUT"
echo "$OUT: $(grep -c '^+' "$OUT") added lines, $(grep -c '^--- ' "$OUT") files"
