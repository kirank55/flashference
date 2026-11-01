# Phase 0 review, round 3

- **Reviewer:** a fresh-context subagent (autoplan.md §2, §5.3), 2026-11-01, about 05:40–06:15Z. Brief: `autopilot/review-brief.md` with phase = Phase 0, phase_start_commit = `03efb2f`, N = 0. Rounds 1 and 2 (`reviews/phase-0-r1.md`, `reviews/phase-0-r2.md`) returned FAIL. An earlier round-3 reviewer was cut off before it wrote anything. Its log shows only reads and scratch scripts in `/tmp`, so this is still round 3.
- **Scope:**
  - `git diff 03efb2f..0b9590d` (19 commits);
  - everything under `results/0/` and `results/env/`;
  - NOTEBOOK.md (the prediction log, every Phase 0 entry, `## Gate 0 answers`, both review-fix entries);
  - STATE.md, BLOCKED.md, `autopilot/`, `bench/`, `toy/` and `docs/drafts/` (read, not run);
  - `check.py --phase 0` and `--phase B`.
- **What I stayed away from:**
  - `v2`: nothing beyond check.py's byte comparison of plan.md. I didn't read the output of the executor's 09:00:00Z `git show v2 --stat` call.
  - `prompts/test/`: one `ls -la prompts/*` listed the directory, which is empty. I opened no file there.
  - The Phase 1 drafts: I ran none of them.
  - Timings: preflight returns 20 (on battery), so I ran no llama-bench and no clock-rate measurement (§4.2).
- **Sources outside the repo:**
  - The executor's session log (`~/.claude/projects/-home-kiran-code-wsl-flashference/61205b4c-….jsonl`). From it I used:
    - tool-call inputs (commands and file paths) for the whole session;
    - the results of four calls: the two llama-bench launches (09:33:29Z, 09:33:41Z) and their wait loop (09:33:59Z), the `sudo -n true` test (09:11:04Z) and the CUDA test program (09:18:33Z);
    - a search of all tool results for eight specific numbers, which printed only the time each first appeared;
    - the first user prompt.

    I didn't read the executor's reasoning.
  - The round-1, round-2 and cut-off round-3 reviewers' logs: only their tool-call inputs.
- **Where my work is:** `/tmp/review-0-r3/`:
  - scripts: `recompute.py`, `pred_history.py`, `gotfields.py`, `pred_probes.py`, `toolcalls.py`, `toolresult.py`, `fullcmd.py`, `grepcmds.py`, `firstseen.py`, `r2_integrity.py`, `untagged.py`;
  - outputs: `*.out`, `toolcalls-all.txt`, `session-files-rerun.txt`;
  - the throwaway clone: `/tmp/review-0-r3/clone`.
- **Environment since the work:** WSL restarted at about 05:34Z and cleared `/tmp`. The lock file and the earlier reviewers' scratch directories are gone, so I couldn't repeat the `/tmp` part of the file census or round 2's access-time evidence. I used the executor's command log for `/tmp` instead (§6).

## Summary

**Round 2's blocking issue is resolved, and so is round 1's.**
- The sixth draft is recorded, unchanged: `docs/drafts/ff-phase1-checks.py`, with the same sha256, size and mtime round 2 found in `/tmp`.
- The 1eef389 edit is reverted: `toy/` has the same blob ids as 05f4335.
- NOTEBOOK.md and STATE.md now give a complete, time-ordered account.
- I checked the account against every file write and every command in the executor's session log, not only against its own census:
  - The only Phase 1 files ever written are the five `toy/` drafts and `/tmp/ff-phase1-checks.py`.
  - None was ever run. The only execution-like step was `ast.parse` at 09:24:55Z.
  - Every Phase 1 number the executor had seen is listed. Two times are slightly off (N4).

**The rest of the phase holds up:**
- All four Gate 0 items are CORRECT.
- Every headline number comes out the same when I recompute it from the raw files with my own script.
- All seven predictions were first committed with `got __`, before their evidence, and their `got` fields were filled in once and never changed. P-006's VOID is justified.
- No test prompts were used, and Phase 0 makes no losslessness claims.
- Both committed review reports are exactly what their reviewers wrote.

I found **no BLOCKING issue**. Phase 0 meets each standard in the brief:
- the machine checks pass and fail on broken inputs;
- the numbers reproduce;
- the gate answers are right;
- nothing is used as fact that isn't;
- no prediction came after its result;
- there is no test-set use;
- every report has its "What I did not test".

The NON-BLOCKING issues are below. The most useful ones:
- one more GPU-lock lapse that the list leaves out (N1);
- gaps in the prediction-order check, including a one-second tie window that is narrower than this machine's clock steps (N2);
- the clock is behaving differently since the restart, which Phase 1 must re-measure (I3).

## 1. Machine checks

Both runs were made after `source autopilot/env.sh`. check.py takes the GPU lock itself for the torch probe; the lock was free.

**`python autopilot/check.py --phase 0`** exited 0 at 05:42:43Z with `PASS: 0 failing check(s)`:
- generated tables: 3/3 match
- prediction ids consecutive, none deleted
- predictions: 7, with 6 checked in order, 0 pending, and P-006 void (same-second tie verified)
- test ledger and test prompts: SKIP (no test runs; test set not written)
- rule-5 files: unchanged vs 03efb2f
- `nvcc`: `Build cuda_12.8.r12.8/compiler.35583870_0`
- `nvidia-smi`: `NVIDIA GeForce RTX 3050 A Laptop GPU, 592.82`
- torch: `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU`
- llama-bench JSON: 3 valid of 3
- model sha256: matches the file and plan.md
- `llama-server --version`: `version: 0.7.0-dev (build 11884, commit 63412d873)`

**`python autopilot/check.py --phase B`** exited 0 at 05:42:58Z:
- All bootstrap checks pass:
  - `.gitignore` is exactly B2's list, and the index has no gitlinks;
  - plan.md is byte-identical to `v2:plan.md`;
  - `ref/flashference` is at v1.0 (a9bcb57d7), and `prompts/v1` equals v1's prompt files;
  - guard.sh handles all 36 cases as expected, and settings.json registers it;
  - the three state files exist.
- The new self-test passed: "9 cases ok".
- The self-test checks only exit codes, so I rebuilt its cases in my own clone with the failure lines printed (cases 1–4 in the probe table below). Each fails for the intended reason.

**Deliberately broken inputs.** I made two breaks in the real working tree and restored each with `git restore`; afterwards `git status --porcelain` was empty. The other probes ran in `/tmp/review-0-r3/clone`, with commit dates set by hand.

| # | broken input | check | result |
|---|---|---|---|
| A | NOTEBOOK.md: `screened: mean of run means 68.32` changed to `68.31` | `--tables` | **FAIL** "committed table differs from command output", 2/3 match, exit 1. PASS after restore. |
| B | `results/0/llama-bench-3.json`: `"n_gen": 128` changed to `64` | `--phase 0` | **FAIL**, 8 failing checks: two tables differ; "3 llama-bench … 2 valid of 3"; P-001, P-002 and P-003 each "evidence results/0/llama-bench-3.json changed after its first commit". Exit 1. PASS after restore. |

| # | probe in the clone | expected | result |
|---|---|---|---|
| 1 | P-007's line deleted | fail | FAIL "deleted: ['P-007']" |
| 2 | P-002, a raw miss, relabelled `got VOID` later | fail | FAIL "marked VOID but its order is provable; a miss can't be voided" |
| 3 | P-008 first committed already VOID | fail | FAIL "first committed with a result already filled in" |
| 4 | text evidence with a time before the prediction | fail | FAIL "records 1 timestamp(s) before the prediction commit" |
| 5 | run in the prediction commit's own second, not marked VOID | fail | FAIL "order unprovable at 1 s resolution (mark VOID and retest)" |
| 6 | same, marked VOID after the evidence commit | pass, as void | PASS, void |
| 7 | VOID mark committed before the evidence | fail | FAIL "VOID mark first appears before its evidence was committed" |
| 8 | evidence deleted after the result was filled in | fail | FAIL "changed after its first commit" |
| 9 | JSON evidence: late `started_utc`, plus an early epoch number (round 2's T9) | should fail | **PASS**, a gap (N2a) |
| 10 | run stamped 06:00:00.100, prediction commit stamped 05:59:59. A 0.6 s backward step between a run started at 06:00:00.1 and a commit 0.3 s later records exactly this. | should fail | **PASS**, a gap (N2b) |
| 11 | P-002's result rewritten later, from the 10.31 % miss to an in-range 1.50 % | should fail | **PASS**, a gap (N2c) |

## 2. Headline numbers, recomputed

`python3 -I /tmp/review-0-r3/recompute.py` reads only `results/0/*` and imports nothing from the repo. It also asserts llama-bench's own fields: `samples_ts` = `n_gen`/`samples_ns`, and `avg_ts`, `avg_ns` and `stddev_ts` match the samples.

| quantity | NOTEBOOK / STATE | recomputed | raw source |
|---|---|---|---|
| run means, all 5 reps (tok/s) | 68.40 / 75.21 / 68.19 | 68.4047 / 75.2149 / 68.1867 | `samples_ns`, llama-bench-{1,2,3}.json:45 |
| within-run SD %, all 5 | 0.32 / 20.32 / 0.18 | 0.3206 / 20.3231 / 0.1821 | same |
| raw mean of run means; spread (max/min − 1) | 70.60; 10.31 % | 70.6021; 10.3073 % | same |
| outlier, run 2 rep 4 | 1.248 s (102.56 tok/s) | 1.2481 s (102.558) | llama-bench-2.json:45 |
| the other 14 reps | 1.865–1.881 s | 1.8646–1.8815 s | same |
| shortfall | 0.622 s | 0.6218 s vs the run-2 median of 1.8698 s | same |
| run 2, reps 1–3 | 5.620 s | 5.6200 s | same |
| run 2 without rep 4: mean / SD % | 68.38 / 0.20 | 68.3790 / 0.1995 | same |
| screened: mean / spread / largest SD | 68.32 / 0.32 % / 0.32 % | 68.3235 / 0.3197 % / 0.3206 % | same |
| P-001 error vs 68 | +3.8 % raw, +0.5 % screened | +3.827 %, +0.476 % | derived |
| journal clock changes in run windows | run 1 09:35:43.111, run 2 09:36:14.740, run 3 none | same | timesync-snapshot.txt:4–8 |
| repetition holding the 09:36:14.740 step | rep 4 | rep 4 in all 3600 cases I tried: stamp offsets of 0–0.99 s, warm-ups of 0–0.5 s, step sizes of 0.59–0.64 s | derived |
| clock window; steps | 123.6 s; −0.591, −0.598, −0.631, −0.617 s | 123.6236 s; same | clock-steps.json `summary`, `steps[]` |
| gaps between steps (guest monotonic) | 32.26, 32.30, 32.19 s | 32.2646, 32.3000, 32.1855 s | `steps[].guest_monotonic_s` |
| host / guest-monotonic elapsed; rate | 120.209 / 122.544 s; +1.94 % | 120.2092 / 122.5439 s; +1.9422 %. Least squares over all 13 readings gives +1.9379 % (max residual 26.7 ms). | `host[]` |
| MONOTONIC gain over RAW; RAW vs host | 2.362 s (+1.95 %); −0.005 % | 2.3616 s (+1.9476 %); −0.0052 % | `summary`; derived |
| step spacing by the host clock | ~31.6 s [derived: 32.25 / 1.0194] | 31.636 s | derived |
| tick in the snapshot | 10189, freq +91.5 ppm | 10189, +91.485 | timesync-snapshot.txt:40 |
| tick-watch.json (P-006, void) | 10160–10238 µs, 13 changes, mean 10195 | 10160–10238; 13; mean 10194.95 by sample, 10194.77 weighted by time | `rows[]` |
| tick-watch-2.json (P-007) | 10170–10217 µs, 12 changes, median 8.07 s, mean 10194 | 10170–10217; 12; median 8.069 s over 11 gaps (8.05–16.57); 10193.56 / 10193.55 | `rows[]` |
| rate from the tick | +1.6 to +2.4 % per stretch; +1.94–1.95 % mean | +1.60 to +2.38 %; +1.948 % and +1.935 % | derived |
| one stretch vs the mean rate | up to ±0.4 % | −0.34 / +0.42 % (tick-watch.json); ±0.23 % (tick-watch-2.json) | derived |
| host-time tok/s [est] | ~69.6 | 69.649 (68.3235 × 1.0194) | derived |
| journal step intervals at bench time; 3 h later | 31.604–31.640 s; 31.569–31.669 s | 31.604–31.640; 31.569–31.669 | timesync-snapshot.txt; clock-steps.json `realtime_after_step` |
| one 32.25 s poll at +1.94 % | 0.614 s | 0.6137 s | derived |
| build | 387 steps, 13 min 12 s, exit 0, no warnings | `[387/387]`; 09:19:20–09:32:32Z; exit 0; 0 lines with "warning". The job log is byte-identical to build-stock.log. | build-stock-job.txt, build-stock.log, results/_jobs/build-stock/ |
| binary sizes | 17,888-byte launchers; llama-cli 1.3 MB; llama-tokenize 41 KB | 17,888 / 1,297,152 / 40,808 B | build-bin-ls.txt:27–30 |
| GPU state at each reading | idle, on AC, memory clock 8000 MHz | util 0, memory 0 MiB, battery 2, memory clock 8000 MHz at all 4 readings | llama-bench-env.jsonl:1–4 |

**Other independent checks:**
- **Model:** the GGUF's sha256 equals results/0/model-sha256.txt and plan.md:140, live in check.py. Its mtime is still 2026-10-30 05:10:52 (rule 3).
- **Binaries:** the sha256 of all 14 binaries and libraries in `llama.cpp/build/bin` equals toolchain.txt:48–61, so nothing was rebuilt after the measurement.
- **llama.cpp:** the checkout is at 63412d873 with no changes, and no file outside `build/` was modified after 08:55Z.
- **The clock explanation holds against the source:**
  - `llama-bench.cpp:40–43` times with `high_resolution_clock`, and `/usr/include/c++/13/bits/chrono.h:1289` makes that `system_clock`.
  - `test_time` comes from `time(NULL)` (llama-bench.cpp:1553).
  - The repetitions are timed between lines 2484 and 2513, after the warm-up at 2415–2441.
  - `ggml_time_us` reads `CLOCK_MONOTONIC` (ggml.c:576–580).
- **HF weights:** the first-level blob names in the HF cache equal hf-models.txt's sha256 values, and I hashed the 135M file myself (`5af571cb…`).
- **CUDA install:** md5 OK (cuda-install.txt:4); "Driver not installed" (line 12); the six "Skipping copy" notices are lines 84, 140, 150, 229, 230 and 242. The md5sum.txt fetch at 09:03:20Z is in the executor's log.
- **Remote:** `git ls-remote origin` shows `refs/heads/op` = 0b9590d = HEAD, and `v2` = 1969856. The three tags are the same objects locally and on the remote. Every push in the log is `git push origin op`.
- **The review record:** the committed `reviews/phase-0-r1.md` and `reviews/phase-0-r2.md` are exactly what the two reviewers wrote. I rebuilt each from its reviewer's Write and Edit calls, and the text is identical.

## 3. Gate 0 answers (NOTEBOOK.md:112–118)

1. **`nvidia-smi`, `nvcc --version` and `torch.cuda.is_available()` all work in WSL** (NOTEBOOK.md:114): **CORRECT.**
   - results/0/toolchain.txt:8 has `release 12.8, V12.8.93` (exit 0 at line 10).
   - Line 15 lists the RTX 3050 A Laptop GPU, compute capability 8.9, driver 592.82.
   - Line 20 has `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU`.
   - All three passed live in my `check.py --phase 0` run at 05:42:43Z.
2. **llama-bench runs the bf16 model on the GPU; the number and the sha256 are written down** (NOTEBOOK.md:115): **CORRECT.**
   - llama-bench-1.json shows `build_commit 63412d873` (line 3), the RTX 3050 A as `gpu_info` (6), `backends: CUDA` (7), `qwen35 0.8B BF16` (9), `n_gpu_layers: 99` (20) and `flash_attn: 1` (25). Runs 2 and 3 show the same.
   - The GPU readings agree: power went from 7.52 W to 41.37 W and the SM clock from 1935 to 2505 MHz around run 1 (llama-bench-env.jsonl:1–2).
   - 68.32 and 70.60 equal my 68.3235 and 70.6021.
   - The [est] 69.6 equals 68.3235 × 1.0194 = 69.65. The answer rightly declines to correct the raw 70.60 that way.
   - "About 1.9 % fast" cites clock-steps.json, measured three hours after the runs. The 0.5–0.6 entry says so (NOTEBOOK.md:96, 109), and the step intervals at bench time match those three hours later (§2).
   - The sha256 in model-sha256.txt equals the file and plan.md:140.
3. **The repo has its first notebook entry committed** (NOTEBOOK.md:116): **CORRECT.**
   - `git show 03efb2f:NOTEBOOK.md` line 14 is `## 2026-10-31 — Bootstrap`.
   - 03efb2f is an ancestor of `origin/op`, and `origin/op` = 0b9590d = HEAD.

**autoplan.md's extra Phase 0 checks** (NOTEBOOK.md:118): **CORRECT.**
- `llama-server --version` reports 63412d873 (toolchain.txt:25, and live).
- STATE.md:10 says `toolkit: native`.
- All four autoplan Phase 0 machine checks pass.

## 4. Predictions

`/tmp/review-0-r3/pred_history.py` and `gotfields.py` walk every committed version of NOTEBOOK.md:
- Each of P-001..P-007 was first committed with `got __ | error __ | learned __`.
- Fields 0–4 (id, date, step, prediction, evidence) never changed afterwards, and no line was ever removed.
- Each `got` and `error` was set once, from `__` to the result, and never edited.
- Only `learned` was revised, and only where the reviews asked: P-005 in 45b79d9, and P-001 and P-006 in a5764ff.
- Every evidence file was added after its prediction, in a descendant commit, and was touched by that one commit only.

| id | prediction commit | evidence first committed (times in the file) | range | result (mine) | my call |
|---|---|---|---|---|---|
| P-001 | 2e08d41, 09:17:07Z, before the build or toolkit existed | b4e140f, 09:37:20Z (test_time 09:35:43, 09:36:08, 09:36:33Z) | 62–74 tok/s | 70.6021 raw, 68.3235 screened | inside |
| P-002 | same | same | 0–2 % | 10.3073 % raw, 0.3197 % screened | raw miss, explained |
| P-003 | same | same | 0–1 % per run | 0.3206 / 20.3231 / 0.1821 % | run 2 miss, explained |
| P-004 | 4f42290, 09:39:04Z | d8bb89a, 12:50:13Z (started 12:47:37Z) | 3–4 steps, 31–32 s apart, 0.4–0.8 s back | 4 steps; 32.19–32.30 s apart (guest monotonic), 31.64 s (host); 0.591–0.631 s | count and size inside; spacing missed, explained |
| P-005 | same | same | +1.3–2.5 % | +1.9422 % | inside |
| P-006 | 088c7dd, 13:30:53Z | cfe6232, 13:33:58Z (started 13:30:53Z) | 10185–10210 µs, ≥ 3 changes | VOID; data 10160–10238, 13 changes | The void is justified. The executor's log shows `git commit … && … python bench/tick_watch.py` in one command, so the run did follow the commit, but the records can't prove it. |
| P-007 | 1eef389, 13:36:19Z | 187374c, 13:38:26Z (started 13:36:27.794Z) | 10140–10260 µs, 12–16 changes, median gap 7.5–8.5 s | 10170–10217; 12; 8.069 s | inside. Its range came from P-006's data, as the line says, so it tests reproducibility. |

P-001's prior figures match their source, `git show archive/wsl:results/2026-10-27-wsl-smoke.md` (a v1 tag, not v2): WSL plain 64.85 and 63.73 tok/s, Windows 65–68, "1.4% to 6.6% lower".

## 5. Losslessness

Phase 0 makes no output-equality claims: there are no speculative configs and no output comparisons. Every commit's "why it can't change the target's output" concerns documentation, a measurement, unrun drafts or a stock build, and each holds:
- The llama.cpp checkout is clean at 63412d873, and the binaries are unchanged.
- `-DLLAMA_OPENSSL=OFF` reaches only cpp-httplib:
  - `vendor/cpp-httplib/CMakeLists.txt:128`;
  - `CPPHTTPLIB_OPENSSL_SUPPORT` in `common/http.h:221`, `tools/server/server-http.cpp:162` and `tools/server/server-models.cpp:2773`.

  None of these is in `src/` or `ggml/`.

## 6. Earlier rounds' blocking issues, checked

**Round 1, B1: Phase 1 drafts written before Gate 0 and not recorded. Resolved.**
- STATE.md:26 has the deviation, and NOTEBOOK.md:122–132 and :156–170 give the account.
- 05f4335 records the five drafts. Their sizes and sha256 prefixes match its message.
- The tree is clean. Today preflight returns 20 because the laptop is on battery, not because of the tree.

**Round 2, B1(a): the sixth artifact, `/tmp/ff-phase1-checks.py`. Resolved.**
- `docs/drafts/ff-phase1-checks.py` matches what round 2 found in `/tmp`:
  - sha256 `c3bac4c57d43a03b54de631a28c504eec28dc7afd554f7dfadf5b95885b4fa22`;
  - 3446 bytes, 73 lines;
  - mtime 2026-10-31T09:27:33.378169348Z.
- The move is in the log at 18:12:34Z (`mv /tmp/ff-phase1-checks.py docs/drafts/…`).
- NOTEBOOK.md:162 and STATE.md:26 record it.
- check.py doesn't import it.

**Round 2, B1(b): the 1eef389 edit to the drafts. Resolved.**
- 44a21cf ran `git restore --source 05f4335 -- toy/`.
- `git diff 05f4335 HEAD -- toy/` is empty, and `git ls-tree` gives the same five blob ids at both commits.
- NOTEBOOK.md:124, :131, :165 and STATE.md:26 now describe the edit and its revert.

**Round 2, B1 item 3: how to check that nothing else exists. Resolved.**
- `bench/session_files.sh` and `results/0/session-files.txt` give a census anyone can repeat.
- I re-ran it today. For `$HOME`, every file with an mtime before 18:20Z on 10-31 is the same as in the committed census. The differences are:
  - files restored or edited later;
  - the `/tmp` entries, cleared by the restart.
- The census only counts some trees (N6), so I also checked the executor's session log directly:
  - **Writes:** every Write and Edit path, and every shell redirect, `cp`, `mv` and `tee` target. The only Phase 1 files are the five `toy/` drafts (09:23:24–09:25:16Z) and `/tmp/ff-phase1-checks.py` (09:27:33Z).
  - **Runs:** every command that names `phase1_`, `toy/` or `ff-phase1`. Besides writing, the commands that touched the drafts were:
    - the in-place edit plus `ast.parse` at 09:24:55Z;
    - `git add` (13:33:58Z, 13:36:19Z);
    - the timespec edit (13:36:06Z);
    - the restore and move (18:12:34Z).

    None executes a draft.
  - **What else was seen:** every Phase-1-relevant read appears in the account at NOTEBOOK.md:157–165:
    - the v2 file listing, the configs and generation configs, the tokenizer `cmp`, the `inspect` calls;
    - the v1 README and B4 greps (I re-ran the README grep on `ref/flashference`; its output matches the account);
    - the weight sizes and llama-bench's `model_size`.

    The only gaps are small and come out in its favour (N4).
  - **Other agents:** all five Agent calls in the session were reviewers.

## 7. BLOCKING issues

None.

I held each brief item to its standard (§1–§5 above):
- **Machine checks** pass, and they fail on broken inputs.
- **Numbers:** every headline number reproduces from raw data. A few secondary numbers lack a tag, but each is correct and sourced nearby (N3).
- **Gate answers:** all four are CORRECT.
- **Estimates:** no [est] or [unv] number is used as a fact. The 69.6 tok/s is labelled [est] wherever it appears.
- **Predictions:** none was written after its result.
- **Test set:** not used, and no comparison mixes two changes.
- **Deviations:** every one in STATE.md has a reason. N1 is one unlisted instance of an already-recorded lapse that affected no data.
- **"What I did not test":** every Phase 0 report has one (NOTEBOOK.md:50, :104, :152, :192).
- **Earlier blocking issues:** both are fixed (§6).

## 8. NON-BLOCKING issues

**N1. One more GPU-lock lapse is missing from the list** (rule 8; STATE.md:27, NOTEBOOK.md:179).
- **The lapse:** at 09:33:18Z the executor ran `llama.cpp/build/bin/llama-bench --help` without the lock. That binary runs `ggml_cuda_init` even for `--help`. Under the lock today, the same unchanged binary (its sha256 equals toolchain.txt:58) prints `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 4093 MiB)`.
- **Part of the same call:** the 09:18:33Z call that ran the CUDA test program also ran a torch CUDA probe (its output is the "torch still reports …" at NOTEBOOK.md:40). That is covered by "~09:18:34" in spirit.
- **No data were affected.** The only measurement began at 09:35:24Z under `gpu-run.sh`, and no other GPU job existed.
- **For the record:** `llama-server --version` does not initialize CUDA, which I verified, so check.py's unlocked call to it is fine.
- **For the owner:** the 0.2 test program launched a kernel without the lock. autoplan §1 says a broken hard rule calls for a `severity: hard` entry. The executor, following round 2's non-blocking grade, recorded it under Deviations instead. The owner may want to rule on whether brief device queries and test kernels count as "GPU runs".

**N2. Gaps in check.py's prediction-order check** (autoplan §5.2). None affects Phase 0: I audited the whole history by hand (§4).
- **(a) Epoch numbers aren't checked** (probe 9; round 2's T9 still passes). Evidence such as clock-steps.json carries epoch nanoseconds (`guest_realtime_ns`) that check.py ignores.
- **(b) The one-second tie window is narrower than this machine's clock steps** (probe 10).
  - Git commit times and evidence stamps both come from `CLOCK_REALTIME`. This machine steps that clock back by 0.591–0.631 s (clock-steps.json), and since the restart timesyncd reports an offset of −1.503 s (I3).
  - So a run that started before its prediction commit can be recorded as starting at least 1 s after the commit's second, if a step lands between the two.
  - Fixes: a margin of 1 s plus the largest step, or evidence stamped with `CLOCK_MONOTONIC_RAW` and the boot id.
  - In Phase 0, every evidence file starts at least 8 s after its prediction, except P-006, which is VOID.
- **(c) A filled-in result can be rewritten later** without any check failing (probe 11). §5.2 doesn't require this check, but `got` is where misses are recorded. Locking `got` and `error` once they are filled would close it. Phase 0's are unchanged (§4).

**N3. Some prose numbers still carry no tag** (rule 10), despite 0b9590d's "tag remaining prose numbers". All are correct by my recompute, and most have a source nearby.
- NOTEBOOK.md:37: 8 GB RAM and swap. The source cited is autoplan.md:18, not a raw file.
- :58: v1's 64.85, 63.73, 65–68 and 1.4–6.6 %. The file is cited, but there is no tag.
- :74: 8000 MHz, and the "50 %" speed-up.
- :77: 5.620 s, 09:36:13.62–15.12, 15.35 and ~1.87 s. These are derived, with no formula tag.
- :93: ±0.4 %, derived from tick-watch.
- :127, :158, :164: weight sizes. They are in hf-models.txt and toolchain-extra.txt, but untagged here.
- :186–187: the I1 and I2 figures.

**N4. Small inaccuracies in the time-ordered account** (NOTEBOOK.md:156–165, 192):
- **:163, "09:37 … llama-bench's JSON showed model_size … model_n_params …":** in the executor's tool results these values first appear at 12:54:45Z. The 09:37:34Z command printed only the JSON's keys.
- **The GGUF's 1,557,662,528 bytes:** first shown by the session-start `ls -la … models/*` at 08:59:51Z. The time-ordered list leaves it out, though the round-1 list includes it (:127).
- **:192** implies `/tmp/a` and `/tmp/b` were deleted, but session-files.txt:93–94 lists both at 18:19Z.

None of these changes anything: all came before any Phase 1 prediction.

**N5. The timebox reading isn't recorded as a deviation.**
- STATE.md:7 counts review rounds 1–2 and their fixes outside Phase 0's 3-unit timebox, citing autoplan §9 ("timeboxes … plus review rounds").
- That reading is defensible, and it decides whether Phase 0 is past the 1.5× line, which is a hard block under §6 and §8. Counting the two fix rounds as units would give 5 units, over the 4.5 limit.
- Round 1 wrote "a fix of this size fits within the 1.5× allowance", which implies the other reading. The owner may want to confirm the executor's.

**N6. The census can't rule out work in the trees it only counts.**
- `bench/session_files.sh` only counts some trees the executor can write to. These include `~/.local`, which rule 7 allows for installs, `~/.claude`, `llama.cpp/` and `.venv/`.
- I checked them for the session's window:
  - `~/.local` has only the CUDA installer's four `.desktop` files (09:17:55–09:18:08Z) and a claude CLI update;
  - `llama.cpp/` outside `build/` has nothing modified since 08:55Z;
  - `ref/` has nothing modified after its creation at 09:03–09:04Z;
  - the session log shows no other writes.
- Nothing is missing. But a census meant to be repeatable should justify each "counted only" group, or list the executor-writable ones.

## 9. Informational

- **I1. The session's own instruction.** The executor's first prompt (2026-10-31T08:59:35Z) was "You are the executor. implement phase 0 and 1 in autoplan.md." That is not §9's start sentence. It is the likely context for drafting Phase 1 during the Phase 0 build. The B1 account (NOTEBOOK.md:123) gives "to use the time the build took" and doesn't mention the prompt. The owner should know that this session was scoped to Phases 0 and 1.
- **I2. The CUDA download began during bootstrap.** Job `dl-cuda` started at 09:03:20Z, before the bootstrap commit at 09:10:57Z. Bootstrap has no gate, and it was a background job, so I don't count it as an issue; the notebook doesn't mention it.
- **I3. The clock has changed since the restart.** Phase 1 should re-measure before relying on STATE.md Decisions. My read-only readings at 05:58Z were all single reads, no timed run [unv: not saved to a raw file]:
  - `adjtimex`: tick = 10000 µs, freq −0.5 ppm;
  - `timedatectl timesync-status`: Offset −1.503 s, poll 32 s;
  - the journal: 49 "Clock change detected" lines since 05:34Z, about 31 s apart;
  - `CLOCK_MONOTONIC` 1520.612 s against `CLOCK_MONOTONIC_RAW` 1445.790 s, so MONOTONIC has averaged +5.2 % since boot.

  STATE.md:19 and BLOCKED.md:5–8 describe the earlier boot: a tick of 10160–10238, steps of about 0.6 s, ±0.4 % per stretch. Decision (3) already says to correct by the rate measured in each session. But llama-bench screening's 0.3 s threshold ("half the smallest step measured") and the ±0.4 % figure should be re-derived once a clock measurement is allowed again (on AC). It also answers part of the open question at NOTEBOOK.md:100: the stepping survived the restart; the tick value didn't.
- **I4. For the Phase 1 reviewer:**
  - The drafts are unchanged since 05f4335.
  - The executor saw the configs, weight sizes, `model_size` and `model_n_params`, and v1's "~0.5 GB (bf16) matrix" line before any Phase 1 prediction (NOTEBOOK.md:125–130, 157–165).
  - Round 2's I4 notes still apply: `phase1_greedy.py` uses `perf_counter`, and scipy comes in through scikit-learn.

## Limits of this review

- **Timings:** I ran no llama-bench, `clock_steps.py` or `tick_watch.py`, because preflight returns 20 (on battery). I checked only the recorded files. A fresh clock-rate measurement is needed before Phase 1 relies on the clock decisions (I3), and it must wait for AC power.
- **`/tmp`:** the restart cleared it, so I couldn't repeat the `/tmp` part of the census or round 2's access-time evidence. For `/tmp` I relied on the executor's session log.
- **The session log:** I used only tool-call inputs, the five results named at the top, and a search for eight numbers that printed only the time each first appeared. I didn't read the executor's reasoning.
- **What the log can't show:** work held only in the agent's context or memory. The executor says so too (NOTEBOOK.md:192).
- **`v2`:** I didn't read it beyond check.py's plan.md comparison, so I can't confirm the claim that the 09:00:00Z `git show v2 --stat` printed "no measured numbers" (NOTEBOOK.md:157). The Gate 1 reviewer may look at v2.
- **My changes:** my two deliberate breaks were restored with `git restore`, which changed the mtimes of NOTEBOOK.md and `results/0/llama-bench-3.json`, not their content. The only new file in the repo is this report.

verdict: PASS
