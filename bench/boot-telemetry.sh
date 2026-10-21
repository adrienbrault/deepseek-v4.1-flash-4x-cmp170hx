#!/usr/bin/env bash
# One-second host counters during loading only; parent stops us before scoring.
set -euo pipefail
out=${1:?results directory required}
mkdir -p "$out"
children=()
finish() {
    trap - EXIT TERM INT
    for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${children[@]}"; do wait "$pid" || true; done
    date -u +%FT%TZ >> "$out/boot-telemetry-times.txt"
    exit 0
}
trap finish EXIT TERM INT
date -u +%FT%TZ > "$out/boot-telemetry-times.txt"
iostat -dx -y nvme0n1 1 > "$out/boot-iostat.txt" 2>&1 & children+=("$!")
pidstat -dru -h -p ALL 1 > "$out/boot-pidstat.txt" 2>&1 & children+=("$!")
nvidia-smi dmon -s t -d 1 -o DT > "$out/boot-pcie.txt" 2>&1 & children+=("$!")
wait "${children[0]}"
