#!/usr/bin/env bash
# Phase 1's official runs (autoplan.md Phase 1): each script once, raw output into results/1/.
# Preflight (AC power, GPU idle, lock held by us) is checked before the CPU timings and again
# before the GPU timings; its readings go to results/1/preflight.jsonl.
#   autopilot/job.sh phase1-runs -- toy/phase1_run_all.sh      (job.sh holds the GPU lock)
set -euo pipefail
cd "$(dirname "$0")/.."
source autopilot/env.sh
out=results/1
mkdir -p "$out"
pre() { autopilot/preflight.sh --gpu | tee -a "$out/preflight.jsonl"; }
pre
python toy/phase1_tokens.py --out "$out"
python toy/phase1_params.py --out "$out"
python toy/phase1_gguf.py --out "$out"
python toy/phase1_greedy.py --device cpu --out "$out"
pre
python toy/phase1_greedy.py --device cuda --out "$out"
python toy/phase1_sample.py --out "$out"
pre || true
