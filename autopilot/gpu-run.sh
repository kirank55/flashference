#!/usr/bin/env bash
# gpu-run.sh: run a short GPU command in the foreground, holding the GPU lock (autoplan.md rule 8).
#   autopilot/gpu-run.sh <command...>
# Waits for the lock if another GPU job holds it. Long runs (> ~10 min) go through job.sh instead.
FF_LOCK_HELD=1 exec flock /tmp/flashference-gpu.lock "$@"
