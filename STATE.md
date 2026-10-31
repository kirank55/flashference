# STATE
updated: 2026-10-31T12:57Z   session: 1
status: gate-review          # running | gate-review | blocked | done-all
phase: 0   step: Gate 0 review
phase_start_commit: 03efb2f
next: Gate 0 review round 1 (reviews/phase-0-r1.md); on PASS, Phase 1 starts with its predictions
timebox: phase 0 used 3/3 units (0.2–0.4; 0.5–0.6 run; 0.6 report + clock investigation)
units: 4 of the 200-unit budget
running_jobs: none
toolkit: native              # ~/opt/cuda-12.8, nvcc V12.8.93
noise_floor: (set by Gate 4, with its definition)
tau_numerics: (set in Phase 6, with its evidence file)
second_reviewer: off
## Done
- Bootstrap B1–B6, commit 03efb2f (no review: autoplan.md gates start at Phase 0)
- Phase 0.1–0.4: toolkit native, venv packages, model sha256 matches plan.md (1ec302a)
- Phase 0.5–0.6: Release build 63412d873; llama-bench tg128 68.32 tok/s screened, 70.60 raw (b4e140f raw, report in this commit); WSL2 clock problem measured (d8bb89a)
## Decisions
- 2026-10-31 Timing on this WSL2 machine (NOTEBOOK Phase 0.5–0.6): the kernel tick is 10188–10192 µs instead of 10000, so CLOCK_MONOTONIC and the wall clock run ~1.9 % fast, and timesyncd steps the wall clock back ~0.6 s every 32 s. (1) Same-session ratios are fine as measured. (2) Absolute GPU timings use CUDA events (GPU clock). (3) Absolute rates from CLOCK_MONOTONIC (perf_counter, llama-server timings) are reported as measured, plus a corrected value using the rate measured in that session. (4) Durations never come from the wall clock (no time.time()). (5) llama-bench numbers always get the step-screening rule of bench/phase0_report.py, with raw numbers shown next to them. preflight.sh now records kernel_tick_us.
## Deviations
- 2026-10-31 `job.sh --no-lock`: §4.3 says job.sh takes the GPU lock. Downloads and CPU-only builds don't touch the GPU, so they run detached without it and GPU units aren't held up. GPU jobs still always take the lock.
- 2026-10-31 `preflight.sh --gpu`: the dirty-tree check (code 30) is for session start. Just before a GPU run, mid-unit, the tree is legitimately dirty, so `--gpu` skips that one check. The GPU-busy test uses the median of the 5 samples, so one blip from the Windows desktop doesn't count as busy.
- 2026-10-31 build flag `-DLLAMA_OPENSSL=OFF`: b11884 replaced `LLAMA_CURL` with OpenSSL-based HTTPS, and there are no OpenSSL headers here (apt needs sudo). It only drops HTTPS model downloading; compute code is unchanged. `-DLLAMA_CURL=OFF` is still passed, as plan.md 0.5 has it.
- 2026-10-31 `autopilot/gpu-run.sh` (new helper): runs a short foreground GPU command under the lock with `FF_LOCK_HELD=1`, so preflight inside the run doesn't mistake our own lock for another job. job.sh sets the same variable for locked jobs.
- 2026-10-31 llama-bench screening rule (an addition to autoplan.md §7.2's validity rules): a repetition is excluded only if it is > 0.3 s shorter than its run's median AND the journal shows a clock change inside the run's window. Applied once (run 2, repetition 4); raw numbers stay in every table.
## Open questions carried forward
- What sets the kernel tick to ~10190 µs, and does it survive a WSL restart? (BLOCKED.md soft entry; owner action optional.)
- (resolved 2026-10-31) The guard hook loaded in the bootstrap session itself: `sudo -n true` was blocked with "guard.sh blocked this command (sudo)".
