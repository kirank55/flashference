# STATE
updated: 2026-10-31T09:24Z   session: 1
status: running              # running | gate-review | blocked | done-all
phase: 0   step: 0.5 build
phase_start_commit: 03efb2f
next: finish the Release build (job build-stock), then 0.6 llama-bench x3 (P-001..P-003 committed in 2e08d41)
timebox: phase 0 used 1/3 units
units: 2 of the 200-unit budget
running_jobs: build-stock (no GPU lock; cmake --build llama.cpp/build -j 4)
toolkit: native              # ~/opt/cuda-12.8, nvcc V12.8.93
noise_floor: (set by Gate 4, with its definition)
tau_numerics: (set in Phase 6, with its evidence file)
second_reviewer: off
## Done
- Bootstrap B1–B6, commit 03efb2f (no review: autoplan.md gates start at Phase 0)
- Phase 0.1–0.4: toolkit native, venv packages, model sha256 matches plan.md
## Decisions
## Deviations
- 2026-10-31 `job.sh --no-lock`: §4.3 says job.sh takes the GPU lock. Downloads and CPU-only builds don't touch the GPU, so they run detached without it and GPU units aren't held up. GPU jobs still always take the lock.
- 2026-10-31 `preflight.sh --gpu`: the dirty-tree check (code 30) is for session start. Just before a GPU run, mid-unit, the tree is legitimately dirty, so `--gpu` skips that one check. The GPU-busy test uses the median of the 5 samples, so one blip from the Windows desktop doesn't count as busy.
- 2026-10-31 build flag `-DLLAMA_OPENSSL=OFF`: b11884 replaced `LLAMA_CURL` with OpenSSL-based HTTPS, and there are no OpenSSL headers here (apt needs sudo). It only drops HTTPS model downloading; compute code is unchanged. `-DLLAMA_CURL=OFF` is still passed, as plan.md 0.5 has it.
- 2026-10-31 `autopilot/gpu-run.sh` (new helper): runs a short foreground GPU command under the lock with `FF_LOCK_HELD=1`, so preflight inside the run doesn't mistake our own lock for another job. job.sh sets the same variable for locked jobs.
## Open questions carried forward
- (resolved 2026-10-31) The guard hook loaded in the bootstrap session itself: `sudo -n true` was blocked with "guard.sh blocked this command (sudo)".
