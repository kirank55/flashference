# Phase 0 review, round 1

- **Reviewer:** a fresh-context subagent (autoplan.md §2, §5.3), 2026-10-31. Brief: `autopilot/review-brief.md` with phase = Phase 0, phase_start_commit = `03efb2f`, N = 0.
- **Scope:** `git diff 03efb2f..382c369` (7 commits); everything under `results/0/` and `results/env/`; NOTEBOOK.md (prediction log, the Phase 0 entries, `## Gate 0 answers`); STATE.md and BLOCKED.md; `check.py --phase 0` and `--phase B`.
- **What I stayed away from:** I read nothing from `v2` beyond check.py's byte comparison of plan.md, and I never opened `prompts/test/` (check.py reports it empty).
- **Where my work is:** my scripts are in `/tmp/review-0/` (`recompute.py`, `pred_tests.sh`), and my throwaway clone is `/tmp/review-0/clone1`.

## Summary

The Phase 0 results are sound:
- All three Gate 0 items are CORRECT.
- Every headline number comes out the same when I recompute it from the raw files with my own script.
- All five predictions were committed before their evidence.
- The executor's explanation of the one wild llama-bench repetition holds up. I checked it against the llama.cpp source, libstdc++, the live system journal and the raw timings, and it is right.
- No test prompts were used, and Phase 0 makes no losslessness claims.

The phase fails on one BLOCKING issue: Phase 1 work started during Phase 0 and was recorded nowhere. Five Phase 1 scripts were written at 09:23–09:25Z while the Phase 0 build ran. Fixing this takes a written account, not a re-run. Phase 0 has used 3/3 units, and a fix of this size fits within the 1.5× allowance in autoplan §6. The NON-BLOCKING issues are about record-keeping. The main one is the kernel-tick figures: only tick 10189 is in a raw file, and at 13:20Z I read 10199–10201.

## 1. Machine checks

Both runs below were made after `source autopilot/env.sh`.

**`flock /tmp/flashference-gpu.lock python autopilot/check.py --phase 0`** exited 0 with `PASS: 0 failing check(s)`:
- generated tables: 3/3 match
- predictions: "5 predictions, 11 evidence files in order, 0 pending"
- test ledger and test prompts: SKIP (no test runs; test set not written yet)
- rule-5 files: unchanged vs 03efb2f
- `nvcc`: `cuda_12.8.r12.8/compiler.35583870_0`
- `nvidia-smi`: `NVIDIA GeForce RTX 3050 A Laptop GPU, 592.82`
- torch: `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU`
- llama-bench JSON: 3 valid of 3
- model sha256: matches the file, and the file matches plan.md
- `llama-server --version`: `version: 0.7.0-dev (build 11884, commit 63412d873)`

**`python autopilot/check.py --phase B`** exited 0, with all 10 bootstrap checks passing:
- `.gitignore` is exactly B2's list, and the index has no gitlinks.
- plan.md is byte-identical to `v2:plan.md`.
- `ref/flashference` is at v1.0 (a9bcb57d7), and `prompts/v1` equals v1's prompt files.
- guard.sh handles 36/36 test cases as expected, and `.claude/settings.json` registers it.
- STATE.md, NOTEBOOK.md and review-brief.md exist.

The guard is also live in this reviewer session. It blocked my own `rm -rf /tmp/review-0/clone` with "rm -rf outside the repo and ~/opt".

**Deliberately broken inputs.** I restored each one with `git restore`. Afterwards `git status --porcelain` showed only `?? toy/`, which was there before I started.

| # | broken input | check | result |
|---|---|---|---|
| a | NOTEBOOK.md summary line: `68.32` changed to `68.33` | `check.py --tables` | FAIL "committed table differs from command output", exit 1; PASS again after restore |
| b | `results/0/llama-bench-2.json`: `"n_gen": 128` changed to `64` | `check.py --phase 0` | FAIL tables (1/3 match), FAIL "3 llama-bench … 2 valid of 3", exit 1 |
| c | scratch clone, T1: widen P-001's range in a later commit | `--predictions` | FAIL "prediction text or evidence path changed after it was committed" |
| d | T2: new P-006 whose evidence (`clock-steps.json`) was committed earlier | `--predictions` | FAIL "evidence … is not strictly after the prediction" |
| e | T3: P-007, then evidence committed later but with `started_utc` before the prediction | `--predictions` | FAIL "records 1 timestamp(s) before the prediction commit" |
| f | T4: P-008 committed with `got` already filled in | `--predictions` | FAIL "first committed with a result already filled in" |
| g | T3b probe: as T3, but the early time sits under the key `run_start_utc` | `--predictions` | **PASS** (a gap; see N5) |

Test b shows a second gap. The prediction-order check still passed with an edited evidence file in the working tree (N5).

## 2. Headline numbers, recomputed

`python3 -I /tmp/review-0/recompute.py` reads only the raw files. It doesn't import `bench/phase0_report.py`. It also checks llama-bench's own fields: `samples_ts` = 128/`samples_ns`, and `avg_ts` is the mean of `samples_ts`.

| quantity | NOTEBOOK | recomputed | raw source |
|---|---|---|---|
| run means, all 5 reps (tok/s) | 68.40 / 75.21 / 68.19 | 68.4047 / 75.2149 / 68.1867 | `samples_ns`, llama-bench-{1,2,3}.json:45 |
| within-run SD % (all 5) | 0.32 / 20.32 / 0.18 | 0.3206 / 20.3231 / 0.1821 | same |
| raw mean of run means | 70.60 | 70.6021 | same |
| raw between-run spread (max/min − 1) | 10.31 % | 10.3073 % | same |
| outlier: run 2, rep 4 | 1.248 s (102.56 tok/s) | 1.2481 s (102.558) | llama-bench-2.json:45 |
| the other 14 reps | 1.865–1.881 s | 1.8646–1.8815 s | same |
| shortfall | 0.622 s | 0.6218 s (vs run-2 median 1.8698 s) | same |
| run 2 with rep 4 removed: mean / SD % | 68.38 / 0.20 | 68.3790 / 0.1995 | same |
| screened mean of run means | 68.32 | 68.3235 | same |
| screened spread / largest within-run SD | 0.32 % / 0.32 % | 0.3197 % / 0.3206 % | same |
| journal clock changes inside run windows | run 1 09:35:43.111, run 2 09:36:14.740, run 3 none | same (the live journal matches the snapshot line for line) | timesync-snapshot.txt:4–8 |
| wall-clock steps | 4: −0.591, −0.598, −0.631, −0.617 s | same | clock-steps.json `steps[]` |
| time between steps (guest monotonic) | 32.27, 32.30, 32.19 s | 32.2646, 32.3000, 32.1855 s (see N8) | same |
| host / guest elapsed time | 120.209 / 122.544 s | 120.2092 / 122.5439 s | `host[]` first and last entries |
| guest CLOCK_MONOTONIC rate vs Windows | +1.94 % | +1.9422 % from the endpoints; least-squares fit over all 13 readings +1.9379 % (max residual 26.7 ms) | `host[]` |
| CLOCK_MONOTONIC gain over CLOCK_MONOTONIC_RAW | 2.362 s (+1.95 %) | 2.3616 s (+1.9476 %) | `summary` |
| RAW vs host | −0.005 % | −0.0052 % (same constant-ratio assumption) | derived |
| step spacing by the host clock | ~31.6 s | 31.636 s | derived |
| tick-only rate | 1.88–1.91 % | 10189 gives +1.890 %; 10189 is the only tick value in a file (N1) | timesync-snapshot.txt:40 |
| host-time tok/s [est] | ~69.6 | 69.65 | derived |
| P-001 error vs 68 | +3.8 % raw, +0.5 % screened | +3.83 %, +0.48 % | derived |
| build | 387 steps, 13 min 12 s, exit 0, no warnings | `[387/387]`; 09:19:20 to 09:32:32Z; exit 0; no line matches "warning"; build-stock.log is byte-identical to the job log | results/0/build-stock*, results/_jobs/build-stock/ |

**Other independent checks:**
- **Model hash:** `sha256sum` of the GGUF gives `ad1549ee…07bbc2`, which equals plan.md:140 (plan.md writes it in upper case). The file's mtime is 2026-10-30 05:10 (rule 3).
- **CUDA runfile:** `md5sum ~/opt/dl/cuda_12.8.1_570.124.06_linux.run` gives `e281945bf4ea42788db29457ebc1e6ae`, the value in autoplan.
- **Binaries:** the sha256 of all 14 files in `llama.cpp/build/bin` equals toolchain.txt:48–61, so nothing was rebuilt after the snapshot.
- **llama.cpp checkout:** at 63412d873 (tag b11884), with 0 lines from `git status --porcelain`.
- **venv:** `uv pip freeze` today is identical to `results/env/pip-freeze.txt`.
- **SmolLM2-135M-Instruct:** `model.safetensors` is 269,060,552 bytes (`ls -L`), and `du` without `-L` reports 3.5M.
- **nsys:** reports version 2024.6.2.225.
- **CMakeCache.txt:**
  - `CMAKE_BUILD_TYPE=Release`
  - CUDA architecture 89
  - nvcc from `~/opt/cuda-12.8`
  - `LLAMA_CURL=OFF` and `LLAMA_OPENSSL=OFF`
  - `GGML_CUDA_GRAPHS=ON`
  - generator Ninja, run from `.venv/bin/ninja`

## 3. Gate 0 answers (NOTEBOOK.md:102–108)

1. **`nvidia-smi`, `nvcc --version` and `torch.cuda.is_available()` all work: CORRECT.** Each claim is in results/0/toolchain.txt:
   - nvcc reports release 12.8, V12.8.93 (line 8, exit 0).
   - nvidia-smi lists the RTX 3050 A Laptop GPU with compute capability 8.9 and driver 592.82 (line 15).
   - torch prints `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU` (line 20).

   I re-ran all three live through `check.py --phase 0`, and all passed.
2. **llama-bench runs the bf16 model on the GPU, and the number and the sha256 are recorded: CORRECT.**
   - The run settings are in the JSON. llama-bench-1.json shows `backends: CUDA` at line 7, `n_gpu_layers: 99` at line 20, the RTX 3050 A as `gpu_info` at line 6, the bf16 model at line 9, and `build_commit 63412d873` at line 3.
   - The GPU readings back this up: power went from 7.52 W to 41.37 W and the SM clock from 1935 to 2505 MHz between the readings before and after run 1 (llama-bench-env.jsonl:1–2).
   - 68.32 tok/s screened and 70.60 tok/s raw are both what the raw data give (§2).
   - The sha256 in results/0/model-sha256.txt equals both the file and plan.md:140.
   - One caveat, not a fault: 68.32 is a guest-clock figure, about 1.9 % low. The entry above says so (NOTEBOOK.md:88), but the answer itself doesn't (N2).
3. **The repo has its first notebook entry committed: CORRECT.**
   - `git show 03efb2f:NOTEBOOK.md` contains `## 2026-10-31 — Bootstrap` at line 14.
   - `git ls-remote origin` shows `refs/heads/op` = 382c369, and 03efb2f is an ancestor of it.
   - GitHub's activity log records the push of 03efb2f at 09:10:58Z.

**The extra autoplan Phase 0 checks (NOTEBOOK.md:108): CORRECT.** toolchain.txt:25 shows the version line, STATE.md:10 says `toolkit: native`, and all four autoplan Phase 0 machine checks pass.

## 4. Predictions

The prediction text and evidence path are unchanged since each first commit. check.py verifies that, and test T1 shows it would catch an edit.

| id | committed | evidence first committed (times inside the file) | range | result | executor's call | my call |
|---|---|---|---|---|---|---|
| P-001 | 2e08d41, 09:17:07Z | b4e140f, 09:37:20Z (test_time 09:35:43, 09:36:08, 09:36:33Z) | 62–74 tok/s | 70.60 raw, 68.32 screened | inside | inside |
| P-002 | same | same | 0–2 % | 10.31 % raw, 0.32 % screened | miss on raw, explained | miss, explained. The script committed before the run (`1ec302a:bench/phase0_report.py`) computes the raw value. |
| P-003 | same | same | 0–1 % per run | 0.32 / 20.32 / 0.18 % raw | run 2 misses, explained | same |
| P-004 | 4f42290, 09:39:04Z | d8bb89a, 12:50:13Z (started_utc 12:47:37Z) | 3–4 steps, 31–32 s apart, each 0.4–0.8 s back | 4 steps; 32.19–32.30 s apart (guest monotonic); 0.591–0.631 s | count and size inside; spacing missed | Same. But 4f42290's commit message says the ~31.5 s spacing came from reading journalctl before the prediction, so the spacing part predicted almost nothing. |
| P-005 | same | same | +1.3–2.5 % | +1.942 % | inside | inside |

Every miss has a written explanation (autoplan §5.2), and the explanations hold (§6).

## 5. Losslessness

Phase 0 makes no output-equality claims: there are no speculative configs and no output comparisons. The only "can't change the target's output" claims are about the build, and both hold:
- **Stock tree:** the llama.cpp checkout is clean at 63412d873.
- **`-DLLAMA_OPENSSL=OFF`:** this flag only switches cpp-httplib's SSL support (`vendor/cpp-httplib/CMakeLists.txt:128`).
  - That support is used only in `common/http.h:221`, `tools/server/server-http.cpp:162` and `tools/server/server-models.cpp:2773`. None of these is in `src/` or `ggml/`.
  - The machine has no `/usr/include/openssl`, so the deviation was needed.

## 6. The clock explanation

**Why llama-bench sees the wall clock and llama-server doesn't:**
- `tools/llama-bench/llama-bench.cpp:40–43` times with `high_resolution_clock`.
- In libstdc++ that is the wall clock: `/usr/include/c++/13/bits/chrono.h:1289` reads `using high_resolution_clock = system_clock;`.
- `test_time` is set with `time(NULL)` (llama-bench.cpp:1553), after the model loads and the context is created. A 1-token warm-up follows (2415–2436), then each repetition is timed between lines 2484 and 2513.
- `ggml/src/ggml.c:576` (`ggml_time_us`) reads `CLOCK_MONOTONIC`. llama-server's timings use that function (`tools/server/server-common.h:426, 437`).

**The live journal agrees with the snapshot:**
- The five snapshot lines are identical to `journalctl -o short-precise --since 09:35:00 --until 09:37:30` today.
- The system timezone is Etc/UTC, so the journal and `test_time` share one time base.

**The step falls inside repetition 4 whatever the timing details.** On the pre-step clock, the run-2 step happened at about 09:36:15.36. My script tries every stamp time from 08.000 to 08.999 s past 09:36 and warm-up lengths of 0 to 0.5 s (9 combinations), and repetition 4 contains the step every time. So the reasoning at NOTEBOOK.md:69 doesn't depend on the warm-up length it leaves unstated.

**The detected steps are the journal's clock changes.** The 4 steps in clock-steps.json land on the journal's "Clock change detected" lines at 12:47:37.572, 12:48:09.239, 12:48:40.908 and 12:49:12.477 to the millisecond.

**timesyncd as the cause.** The notebook says timesyncd is "very likely" what steps the clock. The snapshot fits that:
- It shows a 32 s poll, an offset of about −0.6 s, and "Packet count: 710" (timesync-snapshot.txt:26, 33, 36).
- From boot at 06:20:59 to 12:53, polls every 32.25 s come to about 729. Subtracting about 17 for the 11:27–11:36 outage (lines 18–22) gives about 712 [derived, mine]. So timesyncd has been finding the clock out by about 0.6 s and stepping it on essentially every poll since boot.

**Hard rules:**
- **Rule 1 (git):**
  - GitHub's repo activity API (`gh api repos/kirank55/flashference/activity`) lists three pushes during the run, all of type `push` (none `force_push`) and all to `refs/heads/op`: 09:10:58, 09:20:24 and 12:58:05Z.
  - `v2` is unchanged at 1969856.
  - The tags `v1.0` and `archive/*` are the same objects locally and on the remote.
  - The remote has no `main` branch, but the owner's account deleted it at 2026-10-30T05:04:13Z, before `op` existed.
- **Rules 2, 4 and 5:**
  - `ref/flashference` is clean.
  - The largest committed file is plan.md at 54 KB.
  - Each `results/` file was committed exactly once and never changed afterwards.
  - The rule-5 files are unchanged.
- **Rules 6, 7 and 9:**
  - The runfile came from developer.download.nvidia.com (`results/_jobs/dl-cuda/cmd`).
  - It was installed with `--toolkit --toolkitpath` into `~/opt`.
  - `/tmp/cuda-installer.log` says "Driver not installed" and "Disabling driver option for WSL install", with 0 errors.
- **Rule 12:** nothing touches `prompts/test` in the diff, and there is no test ledger.
- **Rule 8:** I can't check this one from the records (N6).

## 7. BLOCKING issues

**B1. Phase 1 work was done before Gate 0, and it isn't recorded anywhere.**

What's on disk:
- The working tree has five untracked files: `toy/phase1_tokens.py`, `phase1_greedy.py`, `phase1_params.py`, `phase1_gguf.py` and `phase1_sample.py`, 25,769 bytes in all.
- Their mtimes run from 2026-10-31 09:23:24 to 09:25:16Z. That is inside the build-stock job (09:19:20–09:32:32Z), when STATE.md said `next: finish the Release build (job build-stock), then 0.6 llama-bench x3` (`1ec302a:STATE.md`).
- Their docstrings name plan.md 1.1–1.5. They write to `results/1`, which is op's layout; v2 used `results/phase1/` (autoplan.md:358). So they were almost certainly written by the executor, or by an agent it started.

What the record says:
- The files are in no commit, no NOTEBOOK entry and no STATE.md section. STATE.md counts "phase 0 used 3/3 units" and "units: 4", and Deviations doesn't mention them.
- STATE.md:6 says "on PASS, Phase 1 starts with its predictions", yet Phase 1 code already exists.

They were opened after they were written:
- `phase1_tokens`, `phase1_greedy` and `phase1_params` were all read at 09:24:55.46.
- `phase1_gguf` was read 33 ms after it was written, and `phase1_sample` at 09:26:52.
- The filesystem is mounted `relatime`, so these access times were set before my review.
- Whether any script was actually run can't be told from the disk: env.sh sets `PYTHONDONTWRITEBYTECODE=1`, and `results/1/` doesn't exist.

Why this blocks:
- plan.md:13 says: "You don't start the next phase until you pass the gate."
- autoplan §4.0 and §4.3 call for one unit at a time, where a unit ends in a commit, and §3 makes STATE.md the only place that says what's next.
- Rule 5's route for deviations is a written entry, with its reason, that the next reviewer accepts or rejects. A deviation nobody wrote down can't be reviewed. The review brief, item 5, lists "deviations without a reason".
- It also puts Phase 1's predictions at risk (rule 11). autoplan's Phase 1 needs predictions committed before 1.2's timings, 1.3's embedding share and 1.4's byte totals, which are exactly what `phase1_greedy`, `phase1_params` and `phase1_gguf` compute. If any of them ran before those predictions, `check.py --predictions` can't detect it, because git won't keep the 09:23 dates.
- The untracked `toy/` also makes the next session-start preflight exit 30 (preflight.sh:33).

**What fixing it needs.** Only a written account; nothing has to be re-run:
1. A Deviations entry in STATE.md saying what these files are, who wrote them, when, and why outside the unit sequence.
2. A statement of whether any of them was run, and if so what output was seen, so the Phase 1 reviewer can judge whether the Phase 1 predictions are independent of it.
3. A tree that preflight accepts. For example, commit the drafts with that note, so their content and date are on record. The choice is the executor's.

No Phase 0 number or grade depends on this issue.

## 8. NON-BLOCKING issues

**N1. The kernel-tick numbers aren't sourced, and the tick has moved since.** Rule 10, and "[unv] used as fact".
- Only `tick=10189` (freq +91.485 ppm) is in a raw file (timesync-snapshot.txt:40). The 10188 and 10191 ends of the range come from the unsaved 70 s watch.
- Even so, NOTEBOOK.md:82 puts the range under a [meas] tag, and P-005's "learned" field (NOTEBOOK.md:16) cites the range to timesync-snapshot.txt.
- STATE.md:19 and BLOCKED.md:5 say "10188–10192"; 10192 appears in no file.
- "It changed by only ~0.03 %" (NOTEBOOK.md:85) is an untagged [unv] figure, used to support decision (1).

My own read-only `adjtimex` readings, taken 4 s apart at 13:20:13–13:20:33Z:

| reading | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| tick (µs) | 10201 | 10199 | 10199 | 10201 | 10201 | 10201 |
| freq (ppm) | −100.3 | −41.1 | −41.1 | −34.1 | −34.1 | −16.8 |

- That is outside the stated range. It means a rate change of about 0.11 % since 12:53, and the tick changes within 4 s.
- Decisions (1)–(5) still hold, because they rely on ratios against bracketing controls from the same session and on a correction measured in each session. But:
  - The tick numbers should be corrected: tag the range [unv], drop 10192, and record a tick time series. preflight now logs `kernel_tick_us`, which helps.
  - The [est] host-time figure of about 69.6 tok/s is uncertain by at least ±0.1 %.

**N2. The screening rule was made after the data were seen, but the docstring says otherwise.**
- `bench/phase0_report.py:7` says the rule was "fixed before it was applied". In fact it was written after the outlier was seen. The version committed before the run (`1ec302a`) had no screening, and the 0.3 s threshold is "half the smallest step measured" on the 12:47 measurement, three hours later.
- The notebook itself is honest about this: raw numbers appear everywhere, P-002 and P-003 are called misses, and the rule is listed under Deviations.
- The docstring should say "post hoc for the Phase 0 data, applied prospectively from now on".
- The Gate 0 answer leads with the screened 68.32 without saying it is a guest-clock figure.

**N3. Rule 15: a missing "What I did not test" section.**
- The Phase 0.1–0.4 entry (NOTEBOOK.md:32–46) has none. Candidates:
  - SmolLM2-135M was neither loaded nor hash-recorded.
  - The md5 check and the CUDA test program's output weren't saved.
  - nsys wasn't run.
- The 0.5–0.6 list (NOTEBOOK.md:96–100) could add:
  - The clock rate wasn't measured during the runs, only three hours later.
  - Tick stability within a session wasn't recorded.
  - llama-bench wasn't re-run with a monotonic timer to confirm the screened value directly.

**N4. Numbers with no source, or the wrong one** (rule 10):
- **NOTEBOOK.md:51,** "the executables are 17.9 KB launchers": true of llama-bench and llama-server (17,888 B), but not of llama-cli (1,297,152 B) or llama-tokenize (40,808 B). The figure is also untagged.
- **NOTEBOOK.md:40,** "the log reports … CUDA graphs on": `cmake-configure.log` has no CUDA-graphs line. The evidence is toolchain.txt:41 (`GGML_CUDA_GRAPHS:BOOL=ON`).
- **NOTEBOOK.md:12 and :92:** P-001's 63.7–64.9, 65–68 and "about 71", and the "1.4–6.6 % slower", cite no file. The sources are `archive/wsl:results/2026-10-27-wsl-smoke.md` (WSL plain 64.85 and 63.73; "1.4% to 6.6% lower"; Windows 65–68) and plan.md:167 ("about 71").
- **NOTEBOOK.md:35,** "8 GB of RAM and 8 GB of swap": untagged, with no file.
- **NOTEBOOK.md:36:** the md5 match and the installer warnings aren't saved under `results/`.
  - The installer log exists only at `/tmp/cuda-installer.log`, which won't survive a WSL restart, and it also warns about `/usr/share/applications`.
  - I re-checked the md5 myself, and it matches.
- **NOTEBOOK.md:69,** "5.7–6.9 s later": depends on an unstated warm-up length. The conclusion holds anyway (§6).

**N5. Gaps in check.py** (no effect on Phase 0, whose evidence uses `test_time` and `started_utc`):
1. The evidence-time test only scans the five `TIME_KEYS` (check.py:126). An evidence file whose times sit under another key passes without being tested (probe T3b). Require at least one recognised timestamp in every evidence file.
2. A raw evidence file edited after its commit isn't caught (test b). Compare the file with its first committed version. I checked this by hand for Phase 0.
3. `check_phase0` starts a CUDA context through torch without taking the GPU lock itself (check.py:347).

**N6. Rule 8 can't be checked from the records.** When `FF_LOCK_HELD=1` is set, preflight skips the lock test and prints `"lock_held":false` (preflight.sh:37). So `llama-bench-env.jsonl` would read the same whether or not `phase0_llama_bench.sh` ran under `gpu-run.sh`. Recording `FF_LOCK_HELD` or the lock holder in the JSON would fix that.

**N7. STATE.md bookkeeping.**
- The `updated:` stamps are typed by hand and run ahead of their commits: 09:24Z in 1ec302a, which was committed at 09:20:23Z, and 09:25Z in the bootstrap commit at 09:10:57Z.
- After b4e140f, STATE.md kept `running_jobs: build-stock` and an old `next` for about 3 h, until 382c369.

**N8. A rounding nit.** The clock table shows 32.27 s for the first gap between steps, but the raw guest-monotonic difference is 32.2646 s. The value is rounded twice: to 3 decimals in `clock_steps.py`, then to 2 in `phase0_report.py`.

## 9. Informational

- **I1.** `~/opt/cuda-12.8/bin/cuda-gdb` exists. That may settle Phase 5.2's gdb question (autoplan.md:416).
- **I2. A lead for the open question "what sets the tick?"** This is my own hypothesis and is unverified:
  - **What I saw:** something changes the tick every few seconds (N1). The kernel command line includes `hv_utils.timesync_implicit=1`, `/dev/ptp_hyperv` exists, and timesyncd steps the clock every 32 s.
  - **What the data show:** in `clock-steps.json` `host[]`, the guest's wall clock minus the Windows clock swings between −257 and +274 ms around a mean of +25 ms.
  - **The hypothesis:** two time-sync loops are fighting.
    - One keeps the guest on the Windows clock on average, using the tick.
    - The other, timesyncd, steps the guest to the NTP server, which would be about 0.3 s behind Windows.
    - That arrangement gives a rate of about 0.3/16 ≈ 1.9 % and steps of about 32 × 1.9 % ≈ 0.6 s.
  - **What follows if it's right:** BLOCKED.md:7's "doesn't fix the tick" may be wrong. It's worth checking before Phase 2 or 4 relies on the correction.
- **I3.** After this review, `git status` shows `?? reviews/` (this file) and `?? toy/`, which was there before I started (B1).

## Limits of this review

- I did not re-run llama-bench or the clock measurement; I only checked the recorded files.
- I did not run `sudo -n true` to test the guard live, because a reviewer doesn't run sudo. The guard blocking my `rm -rf` is the live evidence.
- I couldn't prove who wrote or ran `toy/`.

verdict: FAIL
