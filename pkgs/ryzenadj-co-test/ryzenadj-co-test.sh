#!/usr/bin/env bash
set -euo pipefail

MAX_OFFSET=50
STEP=5
DURATION=10

usage() {
  echo "Usage: $0 [-m MAX] [-s STEP] [-d DURATION]"
  echo ""
  echo "Test Curve Optimizer values to find optimal undervolt."
  echo "Requires root privileges and CPU load (e.g., xmrig) running."
  echo ""
  echo "Options:"
  echo "  -m MAX       Maximum negative offset to test (default: 50)"
  echo "  -s STEP      Step size between tests (default: 5)"
  echo "  -d DURATION  Seconds to run at each value (default: 10)"
  echo ""
  echo "Example:"
  echo "  sudo $0 -m 60 -s 10 -d 15"
  exit 1
}

while getopts "m:s:d:h" opt; do
  case $opt in
    m) MAX_OFFSET="$OPTARG" ;;
    s) STEP="$OPTARG" ;;
    d) DURATION="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Error: Must run as root" >&2
  exit 1
fi

echo "=== Curve Optimizer Test ==="
echo "Testing CO values from 0 to -$MAX_OFFSET in steps of $STEP"
echo "Duration per step: ${DURATION}s"
echo ""
echo "| CO Value | Encoded    | Avg Freq (MHz) | Temp (C) |"
echo "|----------|------------|----------------|----------|"

# Baseline
ryzenadj --set-coall=0 >/dev/null 2>&1
sleep "$DURATION"
baseline_freq=$(awk '/cpu MHz/ {sum+=$4; count++} END {printf "%.0f", sum/count}' /proc/cpuinfo)
baseline_temp=$(ryzenadj -i 2>/dev/null | grep -F "THM VALUE CORE" | awk -F'|' '{print $3}' | tr -d ' ' | cut -d. -f1)
printf "| %8d | %10d | %14s | %8s |\n" 0 0 "$baseline_freq" "$baseline_temp"

best_offset=0
best_freq=$baseline_freq

for offset in $(seq "$STEP" "$STEP" "$MAX_OFFSET"); do
  encoded=$((1048576 - offset))

  if ! ryzenadj --set-coall="$encoded" >/dev/null 2>&1; then
    echo "| -$offset | $encoded | FAILED | - |"
    continue
  fi

  sleep "$DURATION"

  freq=$(awk '/cpu MHz/ {sum+=$4; count++} END {printf "%.0f", sum/count}' /proc/cpuinfo)
  temp=$(ryzenadj -i 2>/dev/null | grep -F "THM VALUE CORE" | awk -F'|' '{print $3}' | tr -d ' ' | cut -d. -f1)

  printf "| %8d | %10d | %14s | %8s |\n" "-$offset" "$encoded" "$freq" "$temp"

  if [ "$freq" -gt "$best_freq" ]; then
    best_freq=$freq
    best_offset=$offset
  fi
done

# Reset
ryzenadj --set-coall=0 >/dev/null 2>&1

echo ""
echo "=== Results ==="
echo "Baseline frequency: $baseline_freq MHz"
echo "Best frequency: $best_freq MHz (CO=-$best_offset)"
improvement=$((best_freq - baseline_freq))
pct=$((100 * improvement / baseline_freq))
echo "Improvement: $improvement MHz ($pct%)"
echo ""
echo "Recommended config:"
echo "  services.ryzenadj.curveOptimizer.offset = -$best_offset;"
