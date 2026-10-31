# Phase 0 review, round 2

- **Reviewer:** a fresh-context subagent (autoplan.md §2, §5.3), 2026-10-31. Brief: `autopilot/review-brief.md` with phase = Phase 0, phase_start_commit = `03efb2f`, N = 0. Round 1 (`reviews/phase-0-r1.md`) returned FAIL.
- **Scope:** `git diff 03efb2f..583f751` (14 commits); everything under `results/0/` and `results/env/`; NOTEBOOK.md (prediction log, the Phase 0 entries, `## Gate 0 answers`, the review-fix entry); STATE.md, BLOCKED.md, `autopilot/`, `bench/`; `check.py --phase 0` and `--phase B`.
- **What I stayed away from:**
  - `v2`: I read nothing from it beyond check.py's byte comparison of plan.md.
  - `prompts/test/`: never opened. check.py reports it empty.
  - The Phase 1 drafts: I read them and ran none of them.
  - `/tmp/review-0-r2`, the scratch directory of the earlier, cut-off attempt.
- **Sources outside the repo:**
  - File timestamps in the Hugging Face cache and in `/tmp`.
  - The executor's session log (`~/.claude/projects/-home-kiran-code-wsl-flashference/61205b4c-….jsonl`). From it I took only tool-call inputs (commands and file paths) and the results of five calls: the two llama-bench launches, one `date`/`stat` call, and the two v1 greps. I didn't read the executor's reasoning.
  - The round-1 reviewer's log, used only to rebuild the report it wrote.
- **Where my work is:** `/tmp/review-0-r2b/`:
  - scripts: `recompute.py`, `pred_tests.sh`, `pred_history.py`, `transcript_grep.py`, `transcript_full.py`, `transcript_result.py`, `r1_tamper.py`;
  - the throwaway clone: `/tmp/review-0-r2b/clone`.

## Summary

**The Phase 0 work holds up:**
- All four Gate 0 items are CORRECT.
- Every headline number comes out the same when I recompute it from the raw files with my own script.
- All seven predictions were committed with `got __` before their evidence. None was edited or deleted afterwards. P-006's void is justified.
- No test prompts were used, and Phase 0 makes no losslessness claims.
- Most of round 1's fixes work. I broke the new evidence checks on purpose, and each one fails as it should.

**Round 1's blocking issue (B1) is only partly fixed.**
- The core of the executor's account is true: none of the five drafts was ever run, and the Phase 1 numbers it had already seen are listed. I confirmed this two ways, from Hugging Face cache access times and from the executor's own command log.
- But the account is incomplete in two ways:
  - **(a)** A sixth piece of Phase 1 work is recorded nowhere: `/tmp/ff-phase1-checks.py`, Phase 1's machine checks for check.py, written at 09:27:33Z during the Phase 0 build.
  - **(b)** The drafts were edited again before Gate 0, in Phase 0 commit `1eef389`. Four minutes later, the account says that later Phase 1 commits will change them and that nothing for phase N+1 is written before gate N passes.
- This is the same kind of issue round 1 blocked on, so it blocks again (B1 below).
- Fixing it takes a written account only. Nothing has to be re-run, and no Phase 0 number or grade depends on it.

**Main non-blocking issues:**
- check.py's new `VOID` path exempts a prediction from every order check, and a deleted prediction goes unnoticed.
- Three brief CUDA uses ran outside the GPU lock.
- A few numbers are stale or have no source.

## 1. Machine checks

Both runs were made after `source autopilot/env.sh`.

**`python autopilot/check.py --phase 0`** exited 0 with `PASS: 0 failing check(s)`. I didn't wrap it in `flock`: check.py now takes the lock itself for the torch check, so an outer lock would deadlock. The GPU lock was free beforehand, and no job was running.
- generated tables regenerate: 3/3 match
- predictions committed before their evidence: 7 predictions, 12 evidence files in order, 0 pending, void (not credited): P-006
- test ledger and test prompts: SKIP (no test runs; test set not written yet)
- rule-5 files: unchanged vs 03efb2f
- `nvcc`: `Build cuda_12.8.r12.8/compiler.35583870_0`
- `nvidia-smi`: `NVIDIA GeForce RTX 3050 A Laptop GPU, 592.82`
- torch: `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU`
- llama-bench JSON: 3 valid of 3
- model sha256: matches the file, and plan.md's value matches
- `llama-server --version`: `version: 0.7.0-dev (build 11884, commit 63412d873)`

**`python autopilot/check.py --phase B`** exited 0, and all 10 bootstrap checks pass:
- `.gitignore` is exactly B2's list, and the index has no gitlinks.
- plan.md is byte-identical to `v2:plan.md`.
- `ref/flashference` is at v1.0 (a9bcb57d7), and `prompts/v1` equals v1's prompt files.
- guard.sh handles all 36 cases as expected, and `.claude/settings.json` registers it.
- STATE.md, NOTEBOOK.md and review-brief.md exist.

The guard is also live in this session: it blocked a harmless `rm -rf` of a nonexistent `/tmp/review-0-r2b/…` path.

**preflight** (`autopilot/preflight.sh`, start mode) exits 0 on the committed tree (17:59:59Z). It records `kernel_tick_us` (10194) and `gpu_lock` (`free`). `FF_LOCK_HELD=1 … --gpu` without the lock exits 12 (`claimed-but-free`), as round 1's N6 fix intends.

**Deliberately broken inputs.** I broke two inputs in the real working tree and restored each with `git restore`; afterwards `git status --porcelain` was empty. The other probes ran in `/tmp/review-0-r2b/clone`, with commit dates set by hand.

| # | broken input | check | result |
|---|---|---|---|
| T4 | `results/0/clock-steps.json`: `"n_steps": 4` changed to `5` (working tree) | `--predictions` | **FAIL**: "P-004/P-005: evidence … changed after its first commit", exit 1. PASS after restore. |
| T8 | NOTEBOOK.md: `mean of run means 68.32` changed to `68.33` | `--tables` | **FAIL**: "committed table differs from command output", 2/3 match, exit 1. PASS after restore. |
| T5 | clone: P-009 evidence stamped `started_utc` 0.5 s into the prediction commit's second | `--predictions` | **FAIL**: "records 1 timestamp(s) before the prediction commit". The new rule works. |
| T6 | clone: P-010 JSON evidence with no recognised time key | `--predictions` | **FAIL**: "is JSON but records no recognised timestamp". The new rule works. |
| T1 | clone: P-008 first committed with `got VOID: …` already filled in | `--predictions` | **PASS**. This is a gap (N1). |
| T2 | clone: P-002's raw miss relabelled `got VOID: …` in a later commit | `--predictions` | **PASS**, with P-002 listed as void. A gap (N1). |
| T3 | clone: P-002's line deleted | `--predictions` | **PASS**, "6 predictions". A gap (N1). |
| T7 | clone: `.txt` evidence that records a time before the prediction | `--predictions` | **PASS**. A gap (N1). |
| T9 | clone: JSON evidence with a late `started_utc` plus an earlier epoch number under another key | `--predictions` | **PASS**. A gap (N1). |

## 2. Headline numbers, recomputed

`python3 -I /tmp/review-0-r2b/recompute.py` reads only `results/0/*` and imports nothing from the repo. It also asserts llama-bench's own fields: `samples_ts` = `n_gen`/`samples_ns`, `avg_ts` = mean(`samples_ts`), `avg_ns` = mean(`samples_ns`).

| quantity | NOTEBOOK | recomputed | raw source |
|---|---|---|---|
| run means, all 5 reps (tok/s) | 68.40 / 75.21 / 68.19 | 68.4047 / 75.2149 / 68.1867 | `samples_ns`, llama-bench-{1,2,3}.json:45 |
| within-run SD %, all 5 | 0.32 / 20.32 / 0.18 | 0.3206 / 20.3231 / 0.1821 | same |
| raw mean of run means; raw spread (max/min − 1) | 70.60; 10.31 % | 70.6021; 10.3073 % | same |
| outlier, run 2 rep 4 | 1.248 s (102.56 tok/s) | 1.2481 s (102.558) | llama-bench-2.json:45 |
| the other 14 reps | 1.865–1.881 s | 1.8646–1.8815 s | same |
| shortfall | 0.622 s | 0.6218 s (vs the run-2 median of 1.8698 s) | same |
| run 2, reps 1–3 | 5.620 s | 5.6200 s | same |
| run 2 without rep 4: mean / SD % | 68.38 / 0.20 | 68.3790 / 0.1995 | same |
| screened: mean / spread / largest SD | 68.32 / 0.32 % / 0.32 % | 68.3235 / 0.3197 % / 0.3206 % | same |
| P-001 error vs 68 | +3.8 % raw, +0.5 % screened | +3.827 %, +0.476 % | derived |
| journal clock changes inside run windows | run 1 09:35:43.111, run 2 09:36:14.740, run 3 none | same | timesync-snapshot.txt:4–8 |
| repetition holding the 09:36:14.740 step | rep 4 | rep 4 in all 660 cases I tried: stamp offsets of 0–0.95 s, warm-ups of 0–0.5 s, step sizes of 0.59–0.64 s | derived |
| clock window; steps | 123.6 s; 4: −0.591, −0.598, −0.631, −0.617 s | 123.6236 s; same | clock-steps.json `summary`, `steps[]` |
| time between steps (guest monotonic) | 32.26, 32.30, 32.19 s | 32.2646, 32.3000, 32.1855 s | `steps[].guest_monotonic_s` |
| host / guest-monotonic elapsed | 120.209 / 122.544 s | 120.2092 / 122.5439 s | `host[]` first and last |
| guest CLOCK_MONOTONIC rate vs Windows | +1.94 % | +1.9422 % from the endpoints; least squares over 13 readings +1.9379 % (max residual 26.7 ms) | `host[]` |
| CLOCK_MONOTONIC gain over RAW; RAW vs host | 2.362 s (+1.95 %); −0.005 % | 2.3616 s (+1.9476 %); −0.0052 % | `summary`; derived |
| step spacing by the host clock | ~31.6 s | 31.636 s | derived |
| tick in the snapshot | 10189, freq +91.5 ppm | 10189, +91.485 | timesync-snapshot.txt:40 |
| tick-watch.json (P-006, void) | 10160–10238, 13 changes, mean 10195 | 10160–10238; 13; 10194.95 sample mean, 10194.77 time-weighted | `rows[]` |
| tick-watch-2.json (P-007) | 10170–10217, 12 changes, median 8.07 s, mean 10194 | 10170–10217; 12; median 8.069 s over 11 gaps (8.05–16.57 s); 10193.56 / 10193.55 | `rows[]` |
| rate from tick | +1.6 to +2.4 % per stretch; +1.94–1.95 % mean | +1.60 to +2.38 %; +1.935 / +1.948 % | derived |
| one-stretch error vs the mean rate | up to ±0.4 % | −0.34 / +0.42 % (tick-watch.json); ±0.23 % (tick-watch-2.json) | derived |
| host-time tok/s [est] | ~69.6 | 69.65 (68.3235 × 1.0194) | derived |
| build | 387 steps, 13 min 12 s, exit 0, no warnings | `[387/387]`; 09:19:20–09:32:32Z; exit 0; 0 lines with "warning"; build-stock.log equals the job log | build-stock-job.txt, build-stock.log |
| binary sizes | 17,888-byte launchers; llama-cli 1.3 MB; llama-tokenize 41 KB | 17,888 / 1,297,152 / 40,808 B | build-bin-ls.txt:27–30 |

**Other independent checks:**
- **Model hash:** `sha256sum` of the GGUF equals results/0/model-sha256.txt and plan.md:140. The file's mtime is still 2026-10-30 05:10:52 (rule 3).
- **CUDA runfile:** md5 `e281945b…`, as in autoplan.md.
- **Binaries:** the sha256 of all 14 binaries and libraries equals toolchain.txt:48–61, so nothing was rebuilt.
- **CMakeCache.txt:**
  - `CMAKE_BUILD_TYPE=Release`
  - CUDA architecture 89
  - nvcc from `~/opt/cuda-12.8`, ninja from `.venv`
  - `LLAMA_CURL=OFF` and `LLAMA_OPENSSL=OFF`
- **llama.cpp checkout:** clean at 63412d873.
- **venv:** `uv pip freeze` is identical to results/env/pip-freeze.txt. It has pytest 9.2.0, matplotlib 3.11.2 and scikit-learn 1.9.1.
- **HF weights:** the HF blob names equal the sha256 values in results/env/hf-models.txt, as hf-models.txt says.
- **Remote and git history:**
  - `git ls-remote origin` shows `refs/heads/op` = 583f751 = HEAD.
  - `v2` is unchanged at 1969856, and the three tags are the same objects locally and on the remote.
  - The reflog shows only linear commits on `op` since bootstrap.
- **Round 1's report:** the committed `reviews/phase-0-r1.md` is exactly what the round-1 reviewer's Write call produced, rebuilt from its log, so nothing in it was changed.

## 3. Gate 0 answers (NOTEBOOK.md:112–118)

1. **`nvidia-smi`, `nvcc --version` and `torch.cuda.is_available()` all work: CORRECT.**
   - toolchain.txt:8 has `release 12.8, V12.8.93`.
   - toolchain.txt:15 lists the RTX 3050 A Laptop GPU, compute capability 8.9, driver 592.82.
   - toolchain.txt:20 has `2.15.0+cu132 True …`.
   - All three passed live in my `check.py --phase 0` run today.
2. **llama-bench runs the bf16 model on the GPU; the number and the sha256 are written down: CORRECT.**
   - llama-bench-1.json shows `build_commit 63412d873` (line 3), the RTX 3050 A as `gpu_info` (line 6), `backends: CUDA` (line 7), `qwen35 0.8B BF16` (line 9) and `n_gpu_layers: 99` (line 20). Runs 2 and 3 show the same.
   - The GPU readings back this up: power went from 7.52 W to 41.37 W around run 1 (llama-bench-env.jsonl:1–2).
   - 68.32 screened and 70.60 raw equal my 68.3235 and 70.6021.
   - The sha256 in model-sha256.txt equals the file and plan.md:140.
   - "About 1.9 % higher" has the right direction and size: the clock runs fast, so durations come out long and tok/s low. See N6 for a wording nit.
3. **The repo has its first notebook entry committed: CORRECT.**
   - `git show 03efb2f:NOTEBOOK.md` line 14 is `## 2026-10-31 — Bootstrap`.
   - 03efb2f is an ancestor of HEAD, and `origin/op` = 583f751 = HEAD.

**autoplan.md's extra Phase 0 checks (NOTEBOOK.md:118): CORRECT.**
- `llama-server --version` reports 63412d873 (toolchain.txt:25, and live today).
- STATE.md:10 says `toolkit: native`.
- All four autoplan Phase 0 machine checks pass.

## 4. Predictions

`/tmp/review-0-r2b/pred_history.py` walks every committed version of NOTEBOOK.md:
- Each of P-001..P-007 was first committed with `got __`.
- No prediction text, evidence path, date or step field changed later.
- No line was ever removed.

| id | committed | evidence first committed (times in the file) | range | result | my call |
|---|---|---|---|---|---|
| P-001 | 2e08d41, 09:17:07Z | b4e140f, 09:37:20Z (test_time 09:35:43, 09:36:08, 09:36:33Z) | 62–74 tok/s | 70.60 raw, 68.32 screened | inside |
| P-002 | same | same | 0–2 % | 10.31 % raw, 0.32 % screened | raw miss, explained |
| P-003 | same | same | 0–1 % per run | 0.32 / 20.32 / 0.18 % raw | run 2 misses, explained |
| P-004 | 4f42290, 09:39:04Z | d8bb89a, 12:50:13Z (started_utc 12:47:37Z) | 3–4 steps, 31–32 s apart, 0.4–0.8 s back | 4 steps; 32.19–32.30 s apart (guest monotonic); 0.591–0.631 s | count and size inside; spacing missed, explained |
| P-005 | same | same | +1.3–2.5 % | +1.942 % | inside |
| P-006 | 088c7dd, 13:30:53Z | cfe6232, 13:33:58Z (started_utc 13:30:53Z, the same second) | 10185–10210 µs, ≥3 changes | VOID; data 10160–10238, 13 changes | the void is justified from the records; the miss isn't explained (N2) |
| P-007 | 1eef389, 13:36:19Z | 187374c, 13:38:26Z (started_utc 13:36:27.794Z) | 10140–10260 µs, 12–16 changes, median gap 7.5–8.5 s | 10170–10217, 12, 8.07 s | inside, all three. The line says its range was set from P-006's data, so it tests reproducibility, not understanding. |

## 5. Losslessness

Phase 0 makes no output-equality claims: there are no speculative configs and no output comparisons. Every commit's "why it can't change the target's output" is about documentation, measurement or a stock build. Each holds:
- The llama.cpp checkout is clean at 63412d873, and the binaries are unchanged.
- Round 1 checked that `-DLLAMA_OPENSSL=OFF` touches only cpp-httplib.

## 6. Round 1's blocking issue (B1), checked

**What the fix gets right:**
- **The drafts were never run.** No command in the executor's log runs any `toy/phase1_*.py`. The only commands that touch them are:
  - 09:24:55.329Z: an in-place edit of `phase1_gguf.py`, then `ast.parse` on it, as NOTEBOOK.md:124 says;
  - 13:33:58Z: `git add`;
  - 13:36:06Z: the timespec edit (B1(b));
  - 13:36:19Z: `git add`.
- **The access times agree on their own.** The filesystem is mounted `relatime`, so a read updates atime when atime ≤ mtime or atime is over 24 h old:
  - SmolLM2-360M `tokenizer_config.json`: atime 2026-10-30T09:10:57Z, unchanged. So no 360M tokenizer was loaded on 10-31 after 09:10:57Z.
  - SmolLM2-135M `tokenizer_config.json`: atime 09:14:11.849Z, earlier than its mtime of 09:14:11.926Z. So it has never been read since the download.
  - Qwen3.5-0.8B `tokenizer_config.json`: atime 2026-10-25, so not read on 10-31.
  - SmolLM2-135M `model.safetensors`: first read after its download at 13:33:42Z, which is the hashing for hf-models.txt.
  - What this rules out:
    - `phase1_tokens.py` loads all three tokenizers, and `phase1_greedy.py` loads the 360M tokenizer first, so neither ever ran.
    - `phase1_sample.py` loads the 360M tokenizer right after a pure-torch self-test, so it never got past that self-test.
    - `phase1_params.py` loads the 135M weights second, so it never got that far before 13:33:42Z.
    - `phase1_gguf.py` reads only the GGUF, which llama-bench and check.py also read, so its access time says nothing. The command log is the evidence for it.
- **The committed drafts are the originals.** 05f4335's drafts match the byte sizes and sha256 prefixes in its message, and they total the 25,769 bytes round 1 counted.
- **The "already seen" list is accurate.** It matches the executor's commands:
  - 09:15:48Z: `ls -laL` of the weight files;
  - 09:20:47Z: `cat` of the three `config.json` files;
  - 09:21:45Z: `cat` of two `generation_config.json` files and `cmp` of the `tokenizer.json` files;
  - 09:21:54Z: API inspection of `DynamicCache` and `GGUFReader`; no model file was opened;
  - 09:26:29Z and 09:26:38Z: greps of v1's README and B4 report.
  - The only gaps are trivial (N5).
- **The tree is clean**, and preflight exits 0.

**What it misses:** B1 below.

## 7. BLOCKING issues

**B1. The account of Phase 1 work done before Gate 0 is still incomplete, so round 1's B1 isn't fully fixed.**

**(a) A sixth Phase 1 artifact, recorded nowhere.**
- **The file:** `/tmp/ff-phase1-checks.py`: 3,446 bytes, 73 lines, sha256 `c3bac4c57d43a03b…`.
- **When:** mtime 2026-10-31T09:27:33.378Z. That is inside the build-stock job (09:19:20–09:32:32Z), two minutes after the last draft.
- **What it is:** a `@phase("1")` check function for check.py. It:
  - lists the five drafts in `PHASE1_RUNS`;
  - re-runs them from a fresh `git worktree` with `HF_HUB_OFFLINE=1`;
  - compares their `deterministic` output with the committed output;
  - checks that 1.2's three methods give identical ids.

  These are exactly autoplan's Phase 1 machine checks (autoplan.md:354–357).
- **Who wrote it:** the executor. Its own log records `Write: /tmp/ff-phase1-checks.py` at 09:27:33.360Z.
- **Never read again:** its atime equalled its mtime, 09:27:33.378169348, until I read it at 17:49:49Z for this review.
- **The record:** the file appears in no commit, no NOTEBOOK entry and no STATE.md section. A grep of every commit message and every tracked file finds no mention. The account at NOTEBOOK.md:123 and STATE.md:26 names only the five `toy/` files.

**(b) The drafts were changed again before Gate 0, and the account says otherwise.**
- **The edit:** commit `1eef389` (13:36:19Z, a Phase 0 commit) changes `isoformat(timespec="seconds")` to `"milliseconds"` in all five drafts. The edit itself ran at 13:36:06Z. It is the only difference between `05f4335:toy/` and `HEAD:toy/`, and only that commit's body mentions it.
- **The account:** commit `45b79d9` at 13:40:09Z, four minutes after the edit, added these lines, and none of them mentions the edit:
  - NOTEBOOK.md:124: "Commit 05f4335 records the drafts exactly as they were … Later Phase 1 commits will change them."
  - NOTEBOOK.md:131: "From now on, nothing for phase N+1 is written before gate N passes."
  - STATE.md:26: "They are recorded unchanged in 05f4335."
- **The effect:** anyone who reads NOTEBOOK.md and STATE.md will take `toy/` at HEAD to be 05f4335's version, and will think the new rule had been kept.

**Why this blocks.** The reasons are round 1's:
- plan.md:13 says you don't start the next phase until you pass the gate.
- autoplan §4.0 and §4.3 call for one unit at a time, and §1 rule 5 puts deviations, with their reasons, under Deviations in STATE.md.
- Brief item 5 lists "deviations without a reason".
- Round 1 failed the phase for undisclosed Phase 1 work and asked for an account of it. An account that leaves part of that work out doesn't fix the issue, and its new rule was broken in the same fix round.

What doesn't depend on it:
- No Phase 0 number or grade.
- Phase 1's predictions aren't at risk: nothing was run, and the `/tmp` file contains no numbers.

**What fixing it needs.** Writing only; nothing has to be re-run:
1. Add (a) and (b) to the STATE.md deviation and to the NOTEBOOK account. Say what each is, when it happened, and that the `/tmp` file was never run or read. Say what happens to the file: commit it with a note, or delete it, and say which.
2. For the 1eef389 edit, either revert it until Phase 1 or record it as a deviation with its reason. Then correct NOTEBOOK.md:124 and STATE.md:26 to match.
3. Say how you made sure nothing else exists, so the next reviewer can repeat it. For example, give the `find` over the repo, `/tmp` and `~` for files written 09:10–13:45Z.

## 8. NON-BLOCKING issues

**N1. Gaps in check.py's prediction-order check** (autoplan §5.1, §5.2). Phase 0 itself is clean: I audited the whole history by hand (§4).
- **The VOID exemption skips every check.** `if got.startswith("VOID"): … continue` (check.py:186–187) runs before the first-commit check, the unchanged-text check and the evidence checks.
  - A prediction first committed already marked VOID passes (T1).
  - So does a miss relabelled VOID after the fact (T2). It shows up only as "void (not credited)", with no sign that it had been a miss.
  - Fix: run the same first-commit, text and evidence-order checks on VOID lines, and require the VOID reason to be added in or after the evidence commit.
- **A deleted prediction line isn't detected (T3).** Fix: fail if an id that was ever committed (`git log -p NOTEBOOK.md`) is missing now.
- **Times are checked only in JSON evidence, and only under recognised keys** (T7, T9). Fix: require JSON evidence, or scan text evidence for ISO times too.

**N2. P-006's miss isn't explained** (§5.2).
- The data, 10160–10238 µs, fall outside the predicted 10185–10210, and voiding takes P-006 out of the miss count.
- The void itself is conservative, which is fine. The executor's log shows the commit and the run were sequential in one command (13:30:53.655Z: `git commit … && … python bench/tick_watch.py`); the records just can't prove it.
- But §5.2 asks why a range missed. Here, a handful of spot readings undersampled a tick that is re-set every ~8 s over about ±40 µs. Put that in P-006's `learned` field.

**N3. Three brief CUDA uses ran outside the GPU lock** (rule 8).
- **Which:**
  - the 10-line CUDA test program, which launched a kernel at about 09:18:34Z (NOTEBOOK.md:39);
  - `bench/phase0_toolchain.sh`'s torch probe, twice: 09:32:45Z and 09:33:07Z (toolchain.txt:18–21).
- **The script still takes no lock:** `bench/phase0_toolchain.sh:11` calls `python -c 'import torch …'` directly.
- **No data were affected.** Nothing else was using the GPU then: the build is CPU-only, and the only measurement started at 09:35:41Z under `gpu-run.sh`.
- This is the same class as round 1's N5.3, which was fixed only in check.py. Take the lock in the script, and note the lapse in STATE.md.

**N4. Numbers that are stale or have no source** (rule 10):
- NOTEBOOK.md:100 asks "What set `tick` to 10188?". 10188 is in no raw file, even though NOTEBOOK.md:135 says the ranges were corrected throughout.
- NOTEBOOK.md:41 still tags 269,060,552 bytes "[unv: `ls -L`, not saved]", but results/env/hf-models.txt:3 now records it. The "3.5 MB" from `du` is untagged.
- NOTEBOOK.md:127: the GGUF's 1,557,662,528 bytes is in no results file.
- NOTEBOOK.md:38:
  - "Nsight Systems 2024.6.2" is in no results file. `nsys --version` today prints 2024.6.2.225.
  - The installer summary leaves out the `/etc/ld.so.conf.d` and pkgconfig warnings, and six "Skipping copy" warnings (cuda-install.txt:84, 124, 140, 150, 229, 230, 237, 241, 242).

**N5. Small inaccuracies in the record:**
- **Commit 05f4335's message:**
  - It cites a NOTEBOOK section "Phase 1 drafts written early", which doesn't exist.
  - It says the account "is in NOTEBOOK.md and STATE.md Deviations" six minutes before 45b79d9 wrote it.
  - Its subject doesn't follow `<phase.step>: <what>`.
- STATE.md:17 "report in this commit" is stale.
- **The B1 "already seen" list:**
  - It leaves out the two `generation_config.json` files printed at 09:21:45Z.
  - It gives v1's head as "about one third of the bytes", but the README line the executor printed says "~0.5 GB (bf16)" (ref/flashference/README.md:37).
  - Its "09:23–09:25Z" covers the drafts only. The Phase 1 reads and writes ran from 09:20:47 to 09:27:33Z.

**N6. Gate 0 answer 2 overstates the correction.** "So the true rate is about 1.9 % higher" holds for the screened 68.32. It doesn't hold for the raw 70.60, which includes the step-shortened repetition. It is also an estimate: the entry tags it [est] (NOTEBOOK.md:96), but the answer doesn't.

**N7. P-001's `learned` field is misleading.** "the raw one only because one repetition was wrong" reads as if 70.60 is inside the range only because of the bad repetition. Without it the mean is 68.32, which is also inside.

**N8. Bootstrap read v2's file list (before this phase, for the Gate 1 reviewer).**
- At 09:00:00.845Z the executor ran `git show v2 --stat | head -40 && git ls-tree -r --name-only v2 | head -50`. So it saw v2's tip commit message and its file list: names, not contents.
- autoplan B1 allows only plan.md from v2.
- I didn't look at what was printed, since I may read only plan.md from v2.
- The Gate 1 reviewer compares Phase 1 numbers with v2's, so they should check whether that output showed any Phase 1 numbers.

## 9. Informational

- **I1. Support for the 1.9 % correction at bench time.** The rate during the runs wasn't measured, but two things fit the same rate. They are consistent with it; they don't prove it.
  - The journal's intervals between clock changes were 31.604–31.640 s at 09:35–09:37, against 31.569–31.669 s at 12:47–12:49.
  - Repetition 4 lost 0.622 s. One 32.25 s poll at +1.94 % gains 0.614 s [derived: 32.25 × 0.0194 / 1.0194].
- **I2. The warm-up assumption at NOTEBOOK.md:77 holds.**
  - llama-bench-3.json's mtime (09:36:42.903Z) minus its 9.386 s of repetitions puts test_time + warm-up ≤ 09:36:33.517. So run 3's warm-up was ≤ 0.52 s.
  - Run 2's conclusion only needs ≤ 0.73 s.
  - llama-bench-2.json's mtime reads 13:15:36Z because round 1 restored it after a deliberate break; its content equals the commit.
- **I3. The draft reads round 1 saw were not runs.** Round 1 saw `phase1_tokens`, `phase1_greedy` and `phase1_params` read at 09:24:55.46. No executor command read them then: the only command, at 09:24:55.329Z, edits and parses `phase1_gguf.py`. That fits the agent harness re-reading files it had just written. It wasn't a run, as the tokenizer access times show (§6).
- **I4. Notes for the Phase 1 reviewer.**
  - `phase1_greedy.py` times with `time.perf_counter` (CLOCK_MONOTONIC, about 1.9 % fast here), but STATE.md Decisions (1) and NOTEBOOK.md:93 say Phase 1 timers read CLOCK_MONOTONIC_RAW or CUDA events.
  - `phase1_sample.py` needs scipy. scipy 1.18.1 is installed, through scikit-learn, but autoplan 0.3 doesn't name it.

## Limits of this review

- I did not re-run llama-bench, `clock_steps.py` or `tick_watch.py`; I checked only the recorded files.
- I did not run the Phase 1 drafts or `/tmp/ff-phase1-checks.py`.
- I didn't read v2 beyond check.py's plan.md comparison, and I didn't open `prompts/test/`.
- **The access-time evidence relies on two assumptions:**
  - that the filesystem keeps `relatime` semantics (it is mounted `relatime`);
  - that nothing reset the access times.

  My own `cat` of `/tmp/ff-phase1-checks.py` updated its atime. Its earlier value is quoted above.
- From the executor's log I used only tool-call inputs and the five results named at the top, not its reasoning.

verdict: FAIL
