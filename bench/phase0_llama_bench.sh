#!/usr/bin/env bash
# Phase 0.6 smoke test (plan.md 0.6, autoplan.md Phase 0): three llama-bench tg128 runs,
# each -r 5, with GPU readings (temperature, clocks, power, driver) before and after each run.
#   autopilot/gpu-run.sh bench/phase0_llama_bench.sh
# Writes results/0/llama-bench-{1,2,3}.json and appends to results/0/llama-bench-env.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/autopilot/env.sh"
out="$ROOT/results/0"
bin="$ROOT/llama.cpp/build/bin/llama-bench"
mkdir -p "$out"
for i in 1 2 3; do
  [[ ! -e $out/llama-bench-$i.json ]] || { echo "refusing to overwrite llama-bench-$i.json" >&2; exit 1; }
done
"$ROOT/autopilot/preflight.sh" --gpu >> "$out/llama-bench-env.jsonl" || { echo "preflight not clear" >&2; exit 1; }
for i in 1 2 3; do
  "$bin" -m "$FF_MODEL" -ngl 99 -fa on -p 0 -n 128 -r 5 -o json > "$out/llama-bench-$i.json"
  "$ROOT/autopilot/preflight.sh" --gpu >> "$out/llama-bench-env.jsonl" || true
done
