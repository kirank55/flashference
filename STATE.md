# STATE
updated: 2026-10-31T13:39Z   session: 1
status: gate-review          # running | gate-review | blocked | done-all
phase: 0   step: Gate 0 review
phase_start_commit: 03efb2f
next: Gate 0 review round 2 (reviews/phase-0-r2.md; r1 FAIL, fixes in this commit); on PASS, Phase 1 starts with its predictions
timebox: phase 0 used 3/3 units (0.2–0.4; 0.5–0.6 run; 0.6 report + clock investigation)
units: 5 of the 200-unit budget (incl. the review-1 fix unit)
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
- 2026-10-31 Timing on this WSL2 machine (NOTEBOOK Phase 0.5–0.6): the kernel tick (nominal 10000 µs) is re-set about every 8 s to 10160–10238 µs (results/0/tick-watch*.json), so CLOCK_MONOTONIC and the wall clock run +1.6 to +2.4 % fast, +1.94 % on average. timesyncd steps the wall clock back ~0.6 s every 32 s. (1) Same-session ratios are fine when each number averages over many 8 s stretches. A single few-second timing carries up to ±0.4 % clock error, and our own timers use CLOCK_MONOTONIC_RAW or CUDA events. (2) Absolute GPU timings use CUDA events (GPU clock). (3) Absolute rates from CLOCK_MONOTONIC (perf_counter, llama-server timings) are reported as measured, plus a corrected value using the rate measured in that session. (4) Durations never come from the wall clock (no time.time()). (5) llama-bench numbers always get the step-screening rule of bench/phase0_report.py, with raw numbers shown next to them. preflight.sh now records kernel_tick_us.
## Deviations
- 2026-10-31 `job.sh --no-lock`: §4.3 says job.sh takes the GPU lock. Downloads and CPU-only builds don't touch the GPU, so they run detached without it and GPU units aren't held up. GPU jobs still always take the lock.
- 2026-10-31 `preflight.sh --gpu`: the dirty-tree check (code 30) is for session start. Just before a GPU run, mid-unit, the tree is legitimately dirty, so `--gpu` skips that one check. The GPU-busy test uses the median of the 5 samples, so one blip from the Windows desktop doesn't count as busy.
- 2026-10-31 build flag `-DLLAMA_OPENSSL=OFF`: b11884 replaced `LLAMA_CURL` with OpenSSL-based HTTPS, and there are no OpenSSL headers here (apt needs sudo). It only drops HTTPS model downloading; compute code is unchanged. `-DLLAMA_CURL=OFF` is still passed, as plan.md 0.5 has it.
- 2026-10-31 `autopilot/gpu-run.sh` (new helper): runs a short foreground GPU command under the lock with `FF_LOCK_HELD=1`, so preflight inside the run doesn't mistake our own lock for another job. job.sh sets the same variable for locked jobs.
- 2026-10-31 llama-bench screening rule (an addition to autoplan.md §7.2's validity rules): a repetition is excluded only if it is > 0.3 s shorter than its run's median AND the journal shows a clock change inside the run's window. Applied once (run 2, repetition 4); raw numbers stay in every table.
- 2026-10-31 Phase 1 drafts written before Gate 0 (reviews/phase-0-r1.md B1): at 09:23–09:25Z, during the Phase 0 build, the executor drafted toy/phase1_*.py. None was run (one was syntax-checked with ast.parse), and there is no output. They are recorded unchanged in 05f4335. Phase 1 numbers already seen before its predictions (configs, weight file sizes, llama-bench's model_size and n_params) are listed in NOTEBOOK.md "Phase 0 review round 1: fixes", and Phase 1's predictions must say so. Rule from now on: nothing for phase N+1 is written before gate N passes.
- 2026-10-31 P-006 void: its run started in the same second as its prediction commit, so the order can't be proved. It is shown but not credited (check.py lists void predictions); P-007 retested it.
## Open questions carried forward
- What re-sets the kernel tick every ~8 s, and does it survive a WSL restart? Reviewer's hypothesis: a Hyper-V time-sync loop (kernel cmdline has hv_utils.timesync_implicit=1) fighting timesyncd's 32 s steps (reviews/phase-0-r1.md I2). BLOCKED.md soft entry; owner action optional.
- Phase 5.2: `~/opt/cuda-12.8/bin/cuda-gdb` exists (reviews/phase-0-r1.md I1), which may stand in for gdb.
- (resolved 2026-10-31) The guard hook loaded in the bootstrap session itself: `sudo -n true` was blocked with "guard.sh blocked this command (sudo)".
