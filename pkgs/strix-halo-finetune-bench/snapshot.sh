#!/usr/bin/env bash
# Scrape every numeric counter we can reach without enumerating ahead of
# time. Emits flat `KEY=VALUE` lines on stdout, one per scalar. Lines
# that fail to parse as int are skipped so the consumer can run delta
# arithmetic over the whole file blindly.
#
# Sources walked (each gated on existence):
#   /sys/class/net/<iface>/statistics/<stat>
#   /sys/class/infiniband/<dev>/ports/<port>/counters/<stat>
#   /sys/class/infiniband/<dev>/ports/<port>/hw_counters/<stat>
#   /sys/kernel/debug/thunderbolt_ibverbs/peers
#   /sys/kernel/debug/thunderbolt_ibverbs/summary
#   /sys/kernel/debug/usb4_rdma/peers          (old module path)
#
# The thunderbolt_ibverbs/peers output is structured `key=value` text,
# nested under `peer <id>` / `rail=<idx>` headings. We re-emit those as
# `tb.peer<id>.rail<idx>.<key>=<value>` lines.
set -uo pipefail

emit() {
  # $1 = key, $2 = value. Numbers only.
  case "$2" in
    ''|*[!0-9-]*) return ;;
  esac
  printf '%s=%s\n' "$1" "$2"
}

# --- netdev stats ---
for s in /sys/class/net/*/statistics/*; do
  [ -e "$s" ] || continue
  iface=$(echo "$s" | awk -F/ '{print $5}')
  stat=$(basename "$s")
  v=$(cat "$s" 2>/dev/null) || continue
  emit "eth.$iface.$stat" "$v"
done

# --- IB port counters + hw_counters ---
for d in /sys/class/infiniband/*/ports/*/{counters,hw_counters}/*; do
  [ -e "$d" ] || continue
  # /sys/class/infiniband/<dev>/ports/<n>/{counters|hw_counters}/<stat>
  dev=$(echo "$d" | awk -F/ '{print $5}')
  port=$(echo "$d" | awk -F/ '{print $7}')
  bucket=$(echo "$d" | awk -F/ '{print $8}')
  stat=$(basename "$d")
  v=$(cat "$d" 2>/dev/null) || continue
  emit "ib.$dev.p$port.$bucket.$stat" "$v"
done

# --- thunderbolt_ibverbs debugfs (new module) + usb4_rdma (old module) ---
parse_peers() {
  # Re-flatten `peer <N> backend=... rails=...\n  rail=<R> ... key=value ...`
  # into one flat line per key.
  awk -v ROOT="$2" '
    /^peer / { peer = $2; rail = ""; next }
    /^[[:space:]]+rail=/ {
      n = split($0, toks, /[[:space:]]+/);
      for (i = 1; i <= n; i++) {
        if (toks[i] ~ /^rail=/) {
          rail = toks[i]; sub(/^rail=/, "", rail);
        }
      }
      next
    }
    /^[[:space:]]+[a-zA-Z]/ {
      # Generic "  key=value key=value ..." line.
      n = split($0, toks, /[[:space:]]+/);
      for (i = 1; i <= n; i++) {
        t = toks[i];
        if (index(t, "=") == 0) continue;
        eq = index(t, "=");
        k = substr(t, 1, eq - 1);
        v = substr(t, eq + 1);
        # Only emit pure-integer values (skip "10Gb/s", "0x2", "766/768" etc.)
        if (v !~ /^-?[0-9]+$/) continue;
        if (rail == "")
          print ROOT ".peer" peer "." k "=" v;
        else
          print ROOT ".peer" peer ".rail" rail "." k "=" v;
      }
    }
  ' "$1"
}

for src in /sys/kernel/debug/thunderbolt_ibverbs/peers /sys/kernel/debug/usb4_rdma/peers; do
  [ -r "$src" ] || continue
  root=$(echo "$src" | awk -F/ '{print "dfs." $5}')
  parse_peers "$src" "$root"
done

# --- /proc/net/sockstat (TCP fallback transport bytes) ---
if [ -r /proc/net/sockstat ]; then
  awk '/^TCP:/ {
    for (i = 2; i <= NF; i += 2) {
      v = $(i+1);
      if (v ~ /^[0-9]+$/) print "sock.tcp." $i "=" v;
    }
  }' /proc/net/sockstat
fi
