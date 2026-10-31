# autoplan.md: running plan.md with no human in the loop

Written 2026-10-31 for branch `op` of `~/code-wsl/flashference`.

**What this is.** `plan.md` is the technical spec: what to build, measure and decide. This file is the operating manual for an AI agent that carries out plan.md with nobody watching. It sets the order of work, how each step is checked, what the agent may never do, and when it stops. Where the two files disagree, this one wins. Wherever plan.md says "you", read "the agent".

**Owner decisions (2026-10-31)**

1. **Results plus a study trail.** The agent does every phase, including the learning exercises. It writes the gate answers and plain-language explanations so the owner can study the run afterwards.
2. **Push to `origin/op` only.** Nothing else leaves the machine. `main`, `v2`, the tags and v1's README are never touched. The Phase 9 audit of v1 is written as files for the owner to apply.
3. **Fresh start on `op`.** Only plan.md comes over from `v2`. Phase 1 is redone from scratch.

**Machine snapshot (checked 2026-10-31)**

| Item | State |
|---|---|
| GPU | RTX 3050 A Laptop, 4 GB, compute capability 8.9, Windows driver 592.82, idle |
| WSL2 | Ubuntu 24.04. 7 GiB RAM + 8 GiB swap visible, 12 CPU threads, 832 GB free disk. `.wslconfig` sets `memory=8GB`, `swap=8GB` |
| Missing | `nvcc`, `ninja`, `gdb`, `nsys`, `ccache`. **No passwordless sudo.** |
| Present | cmake 3.28.3, gcc, uv, `claude` 2.1.314, `codex` 0.166.0. Windows interop works: `powershell.exe` reports BatteryStatus 2 (on AC) |
| `.venv` | torch 2.15.0+cu132 (sees the GPU), transformers 5.22.0, gguf 0.19.0, datasets 5.1.0, numpy 2.5.4 |
| Models | `models/Qwen_Qwen3.5-0.8B-bf16.gguf`. In the HF cache: `HuggingFaceTB/SmolLM2-360M-Instruct` and `Qwen/Qwen3.5-0.8B` (safetensors). `SmolLM2-135M-Instruct` isn't downloaded yet |
| llama.cpp | `llama.cpp/` checkout at 63412d873 (b11884), branch `stock`, not built |
| `op` branch | README, a `.gitignore` that ignores `models` and `toy`, and `llama.cpp` committed as a bare gitlink. The `ref/flashference` worktree directory was deleted; git lists it as prunable |
| Downloads | The CUDA 12.8.1 runfile and the b11884 Ubuntu CUDA 12.8 release archive are both reachable |

---

## 0. What changes from plan.md

plan.md was written for a person learning as they go, and many of its checks rely on that person. Each one gets a replacement that a script or a second agent can check:

| plan.md relies on | Without a human |
|---|---|
| You write the Phase 1–4 code; AI only tutors | The agent writes all the code. The study trail explains it, and a reviewer reads it. |
| You answer each gate's questions, then an AI grades you | The executor answers in NOTEBOOK.md with evidence. A reviewer with a fresh context grades the answers against the raw data and the source (§5.3). |
| Rule 1: predict before you measure | The prediction is committed before the run that tests it, and `autopilot/check.py` proves the order from git history (§5.2). |
| Rule 2 and Appendix A: explain the code back before you commit it | Every commit message says what changed and why it can't change the target's output. Every llama.cpp diff also gets a reviewer pass. |
| Appendix E reading and videos | The agent reads only what a step needs and cites it. No videos. |
| 0.1: edit `.wslconfig`, run `wsl --shutdown` | Skipped, because `.wslconfig` already gives 8 GB. Builds use `-j 4`, then `-j 2` if a build is killed. |
| 0.2: `sudo apt` the CUDA toolkit | A user-space CUDA 12.8 toolkit under `~/opt`, with no sudo. Fallbacks are in Phase 0. |
| Gate 6: rewrite patch 1 in an afternoon without AI | A clean-room rebuild: a fresh agent rebuilds patch 1 from the design note alone (Phase 6). |
| 9.4: hand the README to someone else | A fresh agent with only the README reproduces one headline ratio. |
| 9.5: update flashference's README | The audit and a proposed README patch go into `results/`. Nothing is applied to `main`. |
| 8.10: borrow or rent another GPU | Out of scope, because it needs money or a person. The bigger-model part stays. |
| Appendix F: you keep the laptop plugged in and the GPU idle | The harness checks AC power and GPU idle before each session and records temperature and clocks. Sessions whose controls drift are thrown out (§7). |

---

## 1. Hard rules

If any of these is broken: stop, write a `severity: hard` entry in `BLOCKED.md` (§8), and stop executing.

**Git and files**

1. Commit only to branch `op` in this repo, and to local branches inside `llama.cpp/`. Push only with `git push origin op` (fast-forward). Never force-push, never rewrite pushed commits, never push another branch, never create, move or delete a tag (`v1.0`, `archive/*`), never commit to `main` or `v2`.
2. `ref/flashference/` is read-only. Never build or edit in it.
3. Never modify `models/Qwen_Qwen3.5-0.8B-bf16.gguf`. A derived GGUF gets a new file name and a tensor-hash check (8.1, 8.8).
4. Never commit model files, build trees, `.venv/` or nsys reports. Gzip raw result files over 5 MB.
5. After bootstrap, never edit `plan.md`, `autoplan.md`, `.claude/settings.json` or `autopilot/guard.sh`. `check.py` compares their hashes with the bootstrap commit. If a step is impossible as written, write the deviation and the reason under **Deviations** in STATE.md, take the closest workable reading, and let the next reviewer accept or reject it.

**Machine**

6. No `sudo`, no `apt`, no NVIDIA driver inside WSL. Never write under `/mnt/c`, never run `wsl.exe`, never install anything on Windows. Read-only checks through `powershell.exe` and `nvidia-smi` are fine.
7. Install only into `~/opt/`, `~/.local/` and `.venv/`.
8. One GPU job at a time. Every GPU run holds `flock /tmp/flashference-gpu.lock`.
9. Downloads come only from huggingface.co, github.com (llama.cpp source and releases; the micromamba binary as a fallback), developer.download.nvidia.com, PyPI and, as a fallback, conda-forge. The agent may read papers and docs that a step needs. The only upload is `git push origin op`.

**Science** (plan.md's 8 rules, made checkable)

10. No number is typed into a report by hand. Tables come from scripts and sit between generated-table markers (§4.4). Every number in prose cites a file and carries one tag:
    - **[meas]**: read from a raw file.
    - **[derived]**: arithmetic on measured numbers. Show the formula.
    - **[est]**: an estimate.
    - **[unv]**: seen but not recorded.
11. Every prediction states a range and is committed before the run that tests it (§5.2).
12. Tune only on dev prompts. Only `bench.py --final` can run test prompts, and only for a config frozen in an earlier commit. Every test run is appended to `results/test-ledger.jsonl`. The executor never prints or opens files in `prompts/test/`.
13. Losslessness is checked every time (§7.4). An unexplained output mismatch stops that experiment.
14. One change per experiment, against a control from the same session.
15. Every report ends with "What I did not test".
16. Don't spend experiments on what v1 already tried and closed:
    - flag or env toggles beyond those plan.md names
    - `--spec-draft-p-min`
    - history-adaptive draft length (plan.md 6.4's ablation is the exception)
    - a separate Q4 draft model
    - tree drafts in b11884

---

## 2. Roles

- **Executor.** The agent doing the work. It is started by the sentence in §9 and works through units one after another in that session (§4).
- **Reviewer.** A different agent session with a fresh context: a subagent started with the Agent tool, or a separate `claude -p` process. It is never the executor's own conversation. It gets only `autopilot/review-brief.md` and file paths, and it fixes nothing. Each review round uses a new reviewer.
- **Test-set author.** A fresh agent that writes `prompts/test/` once, in Phase 4, so the test prompts never sit in the executor's context.
- **Second reviewer (optional, off by default).** `codex exec`, a check by a different model, at the gates marked ★ (Phase 6, 8.4, 8.7, the final review). Turn it on with `second_reviewer: codex` in STATE.md. It uses the owner's Codex quota.

---

## 3. Layout and state

plan.md's lab layout, plus:

```
autoplan.md            # this file
plan.md                # copied unchanged from v2 (bootstrap B1)
STATE.md               # progress board: the only place that says what's next
NOTEBOOK.md            # journal, prediction log, study trail (plan.md Appendix B)
BLOCKED.md             # only while something needs the owner (§8)
STOP                   # the owner creates it to stop the executor after the current unit
autopilot/
  preflight.sh         # environment and GPU checks (§4.2)
  env.sh               # CUDA_HOME, PATH, LD_LIBRARY_PATH, HF_HUB_OFFLINE=1
  job.sh               # runs long GPU jobs detached (§4.3)
  check.py             # machine checks: --phase N, --all, --predictions, --tables
  review-brief.md      # reviewer prompt (§5.3)
  guard.sh             # PreToolUse hook that blocks rule-breaking commands (B5)
  logs/                # session logs (git-ignored)
.claude/settings.json  # registers guard.sh
reviews/               # phase-<N>-r<K>.md, exp-8.x-r<K>.md
patches/<branch>/      # git format-patch stock..<branch>, for every llama.cpp branch
docs/llama-notes.md    # Phase 5 deliverable
docs/study-guide.md    # Phase 9: the owner's way into the study trail
prompts/dev/  prompts/test/  prompts/test.sha256  prompts/v1/  prompts/stress/
```

`llama.cpp/` stays a plain, git-ignored checkout. Its branches are `stock` (b11884, 63412d873), `log`, `my-trim`, `my-trim-cr`, `ff-both` and `exp-8.x`. Every result records the llama.cpp commit it ran on, and `patches/` makes every branch rebuildable from a clean b11884.

**STATE.md** stays under about 150 lines. Old entries move to `docs/history.md`.

```markdown
# STATE
updated: 2026-11-02T14:05Z   session: 17
status: running              # running | gate-review | blocked | done-all
phase: 4   step: 4.1 A/A test
phase_start_commit: 1a2b3c4
next: A/A session, 3 × plain (plan.md 4.1)
timebox: phase 4 used 3/6 units
running_jobs: none
toolkit: native | hybrid
noise_floor: (set by Gate 4, with its definition)
tau_numerics: (set in Phase 6, with its evidence file)
second_reviewer: off
## Done
- Phase 0 gate PASS (reviews/phase-0-r1.md), commit 9f8e7d6
## Decisions
- (date) Gate 7: GEMVs at __ % of measured bandwidth → 8.3/8.4/8.9 in; gaps __ % → 8.6 out
## Deviations
## Open questions carried forward
```

---

## 4. Executor protocol

### 4.0 The loop

When told to execute this plan, the executor repeats §4.1–4.5, one work unit at a time, without asking anyone anything. It stops only when:
- STATE.md says `status: done-all`;
- `BLOCKED.md` has a `severity: hard` entry;
- a file named `STOP` exists in the repo root (delete it, note the stop in STATE.md, and end);
- the session itself ends (context, crash, rate limit). Nothing is lost: the next "execute" resumes from STATE.md.

Before each unit, re-read STATE.md rather than trusting memory of earlier units: the conversation may have been summarized. Never ask the owner a question. If a decision is needed, take the option this file prefers, or log a soft block and take the safest option. If nothing is unblocked, log a hard block and stop.

Run shell commands yourself. A command that takes more than about 10 minutes goes through `job.sh` (§4.3), and you wait for it by polling, never with one long blocking call.

### 4.1 Start

1. If STATE.md doesn't exist, this unit is **Bootstrap** (§6).
2. `source autopilot/env.sh`. Read STATE.md, BLOCKED.md if it exists, this file's section for the current phase, plan.md's section for the current step, and the last two NOTEBOOK.md entries. That's all a session needs to resume.
3. Run `autopilot/preflight.sh`.

### 4.2 Preflight exit codes

| Code | Meaning | Action |
|---|---|---|
| 0 | OK | Go on. |
| 11 | Our own GPU job is running (the lock is held) | Do only non-GPU units. |
| 10 | GPU busy with something else: utilization above 5 % or memory used above 500 MiB, sampled 5 times over 10 s | A GPU unit re-checks every 10 min, for up to 1 h. Meanwhile do a non-GPU unit if one is queued. If the GPU is still busy after 1 h, log a soft block and keep doing non-GPU units, or wait. |
| 20 | On battery: `powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus"` returns something other than 2 | Same as 10. |
| 30 | A crashed session left uncommitted changes | Inspect them. Commit finished work. Move half-written raw results to `results/_incomplete/` (git-ignored) and re-run them. Never `git reset --hard` or `git clean`. |
| 40 | Under 20 GB disk free, or `nvidia-smi` fails inside WSL | Hard block (§8). |

### 4.3 The unit

A work unit is one plan.md step or sub-step that ends in a commit: at most about 2 hours of work and about 150 new lines of code. A unit that measures something takes three commits:

1. **Predict.** Add the prediction line to NOTEBOOK.md (§5.2). Commit `P-0xx: predict …`.
2. **Run.** Raw output goes to the exact path the prediction names. Commit the raw files.
3. **Report.** Generate the tables. Write the NOTEBOOK.md entry (plan.md Appendix B plus **What this shows**, §5.4). Fill in the prediction's result and error. Commit.

**Long runs.** Anything that takes more than about 10 minutes runs detached: `autopilot/job.sh <name> -- <command>`. It takes the GPU lock, writes `results/_jobs/<name>/` (pid, command, start time, exit code; git-ignored) and keeps running after the session ends. `job.sh --running` exits 0 while any job is running. Add the job to `running_jobs` in STATE.md. While it runs, do non-GPU units. If there are none, check it every few minutes. When it finishes, commit its output. If the session ends first, the next one picks the job up from STATE.md. If a job runs past 3× its expected time, kill it and mark its output invalid.

**Commit messages:** `<phase.step>: <what>`, then a body with:
- **What** changed.
- **Why it can't change the target's output**, or "analysis only".
- **Evidence:** the file paths.

After each commit on a llama.cpp branch, run `git -C llama.cpp format-patch stock..<branch> -o patches/<branch>/` and commit the patches here.

### 4.4 Generated tables

Every table in a report sits between markers that name the command that produced it:

```
<!-- gen: python bench/report.py --session results/4/2026-11-03-aa.jsonl --table ratios -->
| config | tok/s | … |
<!-- /gen -->
```

`python autopilot/check.py --tables` re-runs every marked command and fails if any output differs from the committed table. That check enforces rule 10, so the commands must be deterministic.

### 4.5 End

Update STATE.md: what's done, what's next, timebox counts, running jobs, new soft blocks. Commit, then `git push origin op`. If the push fails, note that in STATE.md and carry on. Then go back to §4.0 and start the next unit.

---

## 5. Gates

Every phase, and every experiment in Phase 8, ends with a gate in three parts. All three must pass.

### 5.1 Machine checks

`python autopilot/check.py --phase N` runs that phase's checks (listed in §6) and these global ones:
- tables regenerate (§4.4)
- predictions were committed in time (§5.2)
- the test ledger lists only frozen configs, with one test run per frozen config
- `prompts/test/` matches `prompts/test.sha256`
- the files in rule 5 are unchanged

Each phase adds its own checks to `check.py` before it asks for review.

### 5.2 Prediction log

One line per prediction, near the top of NOTEBOOK.md (plan.md Appendix B):

```
P-007 | 2026-11-20 | 8.1 | code ratio 1.95× (1.85–2.05) because draft head bytes halve | evidence: results/8.1/2026-11-20-code-sweep.jsonl | got __ | error __ | learned __
```

- When the line is committed, `got`, `error` and `learned` are still `__`. The `evidence:` field names the exact file the run will write.
- `check.py --predictions` finds the commit that added the line (`git log --reverse -S'P-007 |' -- NOTEBOOK.md`) and the commit that added the evidence file. The prediction must come first, both in history and in time.
- `bench.py` refuses to run without `--predict P-007,…` naming lines already committed with `got __`, and it writes those ids into the raw file's header. For pure re-runs, use `--no-prediction "<reason>"`.
- A result outside the predicted range is a **miss**. A miss doesn't block, but an unexplained miss does. Write why it missed. If the reason is that your model of the system was wrong, write the corrected model and use it for the next prediction.

### 5.3 Independent review

When the machine checks pass, the executor starts a reviewer with this brief. Bootstrap saves it as `autopilot/review-brief.md`:

> You're reviewing work another agent did. Don't trust anything in NOTEBOOK.md until you've checked it against raw files or source code.
> Inputs: the plan.md and autoplan.md sections for <phase>; `git diff <phase_start_commit>..HEAD`; the raw files under `results/<phase>/`; the gate answers in NOTEBOOK.md.
> 1. Run `python autopilot/check.py --phase <N>`. Pick one check and confirm it fails on a deliberately broken input.
> 2. Recompute every headline number from the raw files with your own short script, not the executor's.
> 3. Grade each gate answer CORRECT, INCOMPLETE or WRONG. Give the reason and the file:line or data that shows it.
> 4. Check every losslessness claim against the hash and divergence data.
> 5. Look for numbers with no source, [est] or [unv] numbers used as facts, predictions written after the result, test-set use, two changes measured as one, deviations without a reason, and a missing "What I did not test".
> List BLOCKING and NON-BLOCKING issues. Don't fix anything. End with one line: `verdict: PASS` or `verdict: FAIL`.

The report goes to `reviews/phase-<N>-r<K>.md`. On FAIL, the executor fixes every BLOCKING issue and asks a **new** reviewer. After a third FAIL:
- in Phases 0–6, hard block;
- in Phase 7, follow that phase's failure rule;
- in an experiment, mark it `stopped: review failed` and move on.

### 5.4 Study trail (owner decision 1)

- Every NOTEBOOK.md entry has a **What this shows** paragraph: 3–6 sentences for a reader who knows basic Python but no C++, CUDA or ML maths. Use plan.md's terms: L1/L2/L3 and the Appendix D glossary.
- Every plan.md gate question is answered under `## Gate N answers`, with tags and citations. The reviewer's grade goes next to each answer.
- Each key idea gets one plot in `analysis/<phase>/`: bandwidth, the GEMV sweep, the batch sweep, acceptance against K, coverage against acceptance, and the time budget.
- Phase 9 collects all of this in `docs/study-guide.md`.

---

## 6. Phases

Each phase below lists what to do, what changes without a human, its machine checks, and its timebox in work units. When a timebox runs out, write up what exists and record it in STATE.md. If the gate is close, continue up to 1.5× the timebox. Otherwise:
- in Phases 0–6, hard block;
- in Phase 7, follow its failure rule;
- in an experiment, stop it.

### Bootstrap (1 unit)

- **B1.** Copy plan.md from v2: `git show v2:plan.md > plan.md`. Nothing else comes from `v2`; don't read or copy its `toy/`, `results/` or NOTEBOOK.md (decision 3).
- **B2.** Fix `op`. Run `git rm --cached llama.cpp`, which drops the gitlink and keeps the checkout. Delete the stale `toy/__pycache__/`. Replace `.gitignore` with:
  ```
  models/
  llama.cpp/
  ref/
  .venv/
  __pycache__/
  *.pyc
  results/_incomplete/
  results/_jobs/
  autopilot/logs/
  prof/
  *.nsys-rep
  *.sqlite
  ```
- **B3.** Recreate the reference: `git worktree prune && git worktree add --detach ref/flashference v1.0`. Copy `ref/flashference/prompts/*.json` to `prompts/v1/`. These are v1's prompt sets, used only to diagnose Phase 6.3. They aren't test prompts.
- **B4.** Write NOTEBOOK.md (plan.md Appendix B layout, empty prediction log), STATE.md and `autopilot/`: `env.sh`, `preflight.sh`, `job.sh`, `check.py` (global checks only for now) and `review-brief.md`.
- **B5.** Write `autopilot/guard.sh` and register it in `.claude/settings.json` as a `PreToolUse` hook on `Bash`. The hook reads the tool call's JSON on stdin and exits with code 2, which blocks the command, on any of these:
  - `sudo` or `apt`
  - any push other than `git push origin op`, and any push with `--force`, `-f` or `--tags`
  - creating, moving or deleting a tag
  - `git reset --hard`, `git clean`, `git branch -D`
  - checking out `main` or `v2`
  - writes under `/mnt/c`, or `wsl.exe`
  - `rm -rf` outside the repo and `~/opt`

  Match whole command words, not substrings: `LLAMA_MTP_ADAPT=1` and "adapt" must pass. The guard prevents accidents. It won't stop an agent that tries to get round it.
- **B6.** Commit everything, including `autoplan.md`, as `bootstrap`, then push.

**Checks:**
- `.gitignore` is as above, and `git ls-files -s` has no gitlinks.
- `plan.md` is byte-identical to `v2:plan.md`.
- `ref/flashference` is at `v1.0`.
- Piping `{"tool_input":{"command":"sudo true"}}` into `guard.sh` exits 2, and piping a command containing `LLAMA_MTP_ADAPT=1` exits 0.
- In the first session after bootstrap, run `sudo -n true` once: the hook must block it. If it doesn't, log a soft block. The rules still apply; only the guard is off.

### Phase 0: Workbench (3 units)

plan.md 0.1–0.6, with these changes:

- **0.1:** skipped (§0).
- **0.2, without sudo:**
  ```bash
  mkdir -p ~/opt/dl ~/opt/tmp && cd ~/opt/dl
  curl -L -C - -O https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run  # 5.4 GB
  echo "e281945bf4ea42788db29457ebc1e6ae  cuda_12.8.1_570.124.06_linux.run" | md5sum -c   # from NVIDIA's md5sum.txt, 2026-10-31
  sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=$HOME/opt/cuda-12.8 \
     --no-man-page --override --tmpdir=$HOME/opt/tmp      # --toolkit never installs a driver
  ```
  - `env.sh` sets `CUDA_HOME=$HOME/opt/cuda-12.8`, puts `$CUDA_HOME/bin` first on `PATH`, and sets `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CUDA_HOME/lib64`. Putting WSL's directory first keeps its own `libcuda.so` in use. CUDA 12.8 is also the version of the official b11884 Ubuntu build that v1's WSL smoke test used. If the installer fails, read `/tmp/cuda-installer.log`.
  - Get `ninja` with `uv pip install ninja`. cmake is already installed. Skip ccache.
  - **Fallback 1:** conda-forge in user space. Download the `micromamba` binary, then run `micromamba create -p ~/opt/cuda-conda -c conda-forge cuda-nvcc cuda-cudart-dev libcublas-dev cuda-version=12.8`.
  - **Fallback 2:** v1's hybrid build (`git show archive/wsl:scripts/reproduce.sh`). Host code is built with `-DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON` and runs on the official `llama-b11884-bin-ubuntu-cuda-12.8-x64.tar.gz` libraries. Set `toolkit: hybrid` in STATE.md. This allows host-code patches only (`src/`, `common/`, `tools/`). That covers Phases 0–6, 8.1–8.5, 8.8 and 8.10. It blocks 8.6, 8.7, 8.9 and the debug build. Log a soft block asking the owner to run plan.md 0.2's apt commands, and carry on.
- **0.3:** add `pytest matplotlib scikit-learn` to `.venv`, then save `uv pip freeze` to `results/env/pip-freeze.txt`.
- **0.4:** check the GGUF's sha256 against plan.md's value. Download `HuggingFaceTB/SmolLM2-135M-Instruct`. `env.sh` sets `HF_HUB_OFFLINE=1`, because a Hub network check once hung a run on this machine for minutes. Download steps override it with `HF_HUB_OFFLINE=0`.
- **0.5:** `llama.cpp/` already exists, so don't clone it; check that `git -C llama.cpp status` is clean. **Every measured binary is a Release build** (`-DCMAKE_BUILD_TYPE=Release`, in `build/`). This departs from plan.md on purpose: every number must come from binaries built the same way, and v1 measured Release builds. If Phase 5 needs gdb, make a separate RelWithDebInfo build in `build-dbg/` and never measure it. Build `llama-quantize` and `llama-eval-callback` when a step needs them.
- **0.6:** first commit a prediction for llama-bench tg128 (plan.md gives v1's Windows figure, about 71 tok/s). Then run `llama-bench … -r 5 -o json` three times and save the output to `results/0/`.

**Checks:**
- `nvcc --version` works, or STATE.md says `toolkit: hybrid`.
- `nvidia-smi` and `torch.cuda.is_available()` work.
- The llama-bench JSON is committed, and the sha256 is recorded.
- `llama-server --version` reports 63412d873.

### Phase 1: One token (4 units)

plan.md 1.1–1.5, written from scratch in `toy/phase1_*.py`, with raw output in `results/1/`.

- Commit predictions before 1.2's timings, 1.3's embedding share and 1.4's byte totals.
- 1.2: interleave the three methods across repeats (A B C A B C …) so drift affects all three alike. Report medians and ranges.
- **Checks:**
  - Every script runs from a clean checkout with `HF_HUB_OFFLINE=1`.
  - Deterministic outputs (token ids, byte and parameter counts) are identical on a re-run.
  - 1.2's three methods give identical ids.
- **Gate 1:** answer plan.md's 5 questions (§5.4). The reviewer also compares the deterministic numbers (GGUF byte counts, parameter counts, vocabulary size) with `git show v2:results/phase1/gguf.json` and `params.json`. That is an independent replication. Only the reviewer looks at v2, and only after the executor's numbers are committed. A mismatch is BLOCKING until it's explained.

### Phase 2: Speed = bytes (4 units)

plan.md 2.1–2.5, in `toy/phase2_*.py`. Time with CUDA events, warm up first, and repeat many times. Raw output goes to `results/2/` and plots to `analysis/2/`.

- 2.1: find the GPU's memory bus width in a source you cite, and compute the theoretical peak. Commit a prediction of the measured copy bandwidth before you measure it.
- 2.2: before the sweep, commit a prediction of the 485 MiB head GEMV time. Fit `t = a + b·bytes` and keep the residuals.
- 2.3: before the sweep, commit a prediction of where the batch curve bends.
- **Gate 2 prediction:** plain bf16 decode tok/s in llama.cpp, from bytes, bandwidth and overhead only. Phase 4.2 checks it. The Phase 0 llama-bench number is already known, so state how far the first-principles number is from it, and why.
- **Checks:** the plots and fit JSON exist, and the 2.5 argmax-flip example reproduces with a fixed seed.

### Phase 3: Speculation from scratch (6 units)

plan.md 3.1–3.6, including the optional 3.6 (decision 1). The code is in `toy/spec.py` and the tests in `toy/test_spec.py`.

- 3.2: the exactness test is a pytest. On 20 prompts, in fp32 on the CPU, speculative output must equal `greedy(target)` token for token, and must equal `generate(assistant_model=draft)`. Keep `n` at 64 tokens or less so CPU runs stay short. Then, in bf16 on the GPU, count mismatches against plain bf16 greedy. For each one, record the first differing position and the plain top-2 logit gap there.
- 3.3: commit the speedup prediction, from the measured `T_target(1)`, `T_target(γ+1)` and `T_draft`, before measuring the speedup.
- 3.5: the corpora are WikiText-2 (through `datasets`) and the Python standard library in `/usr/lib/python3.12/`. "The output still equals target greedy" is also a pytest.
- **Gate 3:** answer all 6 questions. Question 6, flashference's "How it works" in your own words with the experiment behind every headline number, uses `ref/flashference/README.md` and its `results/`. The reviewer checks each number against its file.
- **Checks:** `pytest toy/` passes, and the cost-model prediction and its error are logged.

### Phase 4: Harness (6 units)

plan.md Phase 4, with these additions.

- **Prompts.**
  - Write `prompts/README.md` first: file format, domains, chat format, and a length target of about 256 output tokens.
  - The executor writes `prompts/dev/`: 10 prompts each for prose, code, copy-heavy and multilingual (es, de, zh, hi). Replace any dev prompt whose output stops before 128 tokens.
  - Then a test-set author (a fresh agent) writes `prompts/test/`. Its brief is `prompts/README.md` plus: "don't read `prompts/dev/`; write 10 prompts per domain; run `sha256sum prompts/test/* > prompts/test.sha256`; commit without describing the prompts in the commit message".
- **bench.py** follows plan.md's spec, and also:
  - runs the AC and idle checks from preflight;
  - records temperature, clocks, power draw and driver version before and after each config;
  - sends `"return_tokens": true`, so the token ids are saved as well as the text hash;
  - records the llama.cpp commit, the binary's path and sha256, and the full command line and env in every line;
  - has the `--predict` gate (§5.2);
  - in `--final` mode, refuses unless the config file was committed before the run and the prompts match `prompts/test.sha256`. Each final run appends to `results/test-ledger.jsonl`.
- **report.py** follows plan.md's spec and supports the table check (§4.4). `bench/test_report.py` tests the ratio and σ formula, acceptance and hash equality on small handmade JSONL files.
- **bench/divergence.py** is the losslessness classifier (§7.4), needed from Phase 6 on.
- 4.1 A/A test. 4.2 checks the Gate 2 prediction and fills in its error. 4.3 `-bs`: predict first.
- **Gate 4** sets `noise_floor` in STATE.md, with its exact definition: which spread, measured over what.
- **Checks:**
  - `pytest bench/` passes.
  - The A/A report regenerates.
  - `--final` refuses a config that isn't frozen (test it with a dummy).

### Phase 5: Reading llama.cpp (5 units)

plan.md Phase 5. The deliverable is `docs/llama-notes.md`, written for the study trail. It covers:
- the call chain from the HTTP request down to `llama_decode`;
- a mermaid diagram of one speculative cycle: the graphs that run on the GPU, where the host waits, and where tokens are accepted and state rolled back;
- the matmul kernel chosen for 1 token and for 3;
- what happens to the DeltaNet state when drafts are rejected;
- an annotation of every hunk of `ref/flashference/patches/b4-trim-draft-head.patch`.

Notes:
- Every code reference is written `path:line` with a short quoted fragment. `check.py --phase 5` checks that each fragment appears within 3 lines of that line at commit 63412d873.
- 5.1: logging behind an env var, on branch `log`. Test: with the variable unset, output hashes equal `stock`'s on dev prompts, and speed is within the noise floor.
- 5.2: gdb isn't installed and there's no sudo. Try conda-forge `gdb` through micromamba into `~/opt/gdb`, with the `build-dbg/` build. If that fails, trace the call chain through the source, confirm it with temporary log lines on a throwaway branch, and say so in the notes.
- 5.4: what the code says about kernel choice stays [unv] until Phase 7's trace shows which kernels actually run.
- **Checks:** every reference resolves, the `log` branch hash test passes, and every patch hunk is annotated.

### Phase 6: Rebuild and reproduce (8 units) ★

plan.md Phase 6, with these additions.

- **6.1:** the design note comes first, in NOTEBOOK.md. Each of the 4 implementation steps is a commit on `my-trim` with its own test. A reviewer reads each step's diff with the brief "find anything that could change the target model's output".
- **Clean-room check** (replaces "rewrite it in an afternoon"). A fresh agent gets only the design note and `docs/llama-notes.md`, not the diff or v1's patch. It implements patch 1 on `my-trim-cr`, starting from `stock`. Its outputs must hash-equal `my-trim`'s on the dev prompts. Every difference between the two diffs is explained in NOTEBOOK.md. If the check fails, improve the design note; the failure doesn't block.
- **6.2:** check the tokenizer against the GGUF's ids on 20 strings, as plan.md says.
- **6.3:** build the reproduction table with **v1's ranking file** (`ref/flashference/results/2026-10-26-bf16-b4-token-rank.txt`), so the patch is the only change. Your own 6.2 ranking gets one extra row. If a ratio is more than about 10 % off v1's, investigate in this order, one change at a time:
  1. Run that config on `prompts/v1/`, to separate the effect of the prompts.
  2. Run `ff-both` against `my-trim` on the same config, to separate the effect of the patch.
  3. Compare acceptance. A different acceptance points at the draft side. The same acceptance at a different speed points at cost.

  If the gap is still unexplained after 2 more units, record it as a finding and continue with your own measured baseline. Later phases compare only against same-session controls, so they don't depend on v1's numbers.
- **Set `tau_numerics`.** Collect at least 20 first divergences between untrimmed MTP and plain. If the dev prompts don't give 20, build `prompts/stress/` now (see 8.4) and use it. Run `divergence.py` on each.
  - Set `tau_numerics` to 2 × the largest plain top-2 logit gap among them. Record it in STATE.md with the evidence file.
  - Plot these gaps next to the top-2 gaps at all positions. Numerics flips should sit at the very low end.
  - If they don't, or if any divergent token is outside the plain top 2, the reference itself is wrong. Hard block.
- **6.4:** `ff-both` is `stock` plus v1's two patches applied unchanged with `git apply`. Commit a prediction before measuring with ADAPT on and off.
- **Checks:** the patch hash tests pass, the reproduction table regenerates, every mismatch is classified, and the clean-room result is recorded.

### Phase 7: Profile (5 units)

plan.md Phase 7.

- nsys should come with the toolkit (`find ~/opt/cuda-12.8 -name nsys`). First check that a CUDA trace under WSL2 isn't empty, i.e. that `nsys stats --report cuda_gpu_kern_sum` has kernel rows.
- **If nsys is missing or the trace is empty:** build the budget inside llama.cpp. Turn CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS=1`) and time each graph node with an eval callback that synchronizes. The synchronizing distorts the total, so use the result only for shares and tag the budget [partial]. Take a Gate 7 decision from it only if the share is at least 10 percentage points clear of the threshold. Log a soft block asking for a trace on the Windows side.
- Commit `nsys stats` CSV output, not `.nsys-rep` files.
- **Gate 7 decisions** come from a script. It applies plan.md's thresholds (80 % of measured bandwidth; idle gaps plus re-captures at 5 % or more of a cycle) to the budget and writes the result under Decisions in STATE.md.
- **Checks:** the budget sums to the measured ms per token within 5 %, and the decision script's output matches STATE.md.
- **Failure rule:** Phase 7 is the one phase whose failure doesn't stop the run, because only 8.6 and 8.9 depend on its decisions. If its gate fails three times or its timebox runs out, log a soft block, record `Gate 7: no decision` under Decisions, skip 8.6 and 8.9, and go on to Phase 8.

### Phase 8: Experiments

Work this queue in order. Cheap, likely wins come first. Then the new certified-head idea, then bit-exact decoding, then work that depends on conditions, then research-grade training. If an item's condition is false, write a one-paragraph report saying why (`skipped: <condition>`).

| # | Experiment | Condition | Timebox (units) |
|---|---|---|---|
| 1 | 8.1 quantized drafter, bigger K | Phase 6 passed | 6 |
| 2 | 8.2 better draft vocabularies | Phase 6 passed | 8 |
| 3 | 8.5 drafter arbitration | Phase 6 passed | 5 |
| 4 | 8.3 certified head, offline | Phases 3 and 5 passed | 6 |
| 5 | 8.4 certified head in llama.cpp ★ | 8.3 met its "Done when" | 10 |
| 6 | 8.7 bit-exact speculation ★ | Phase 7 finished (passed or no decision); `toolkit: native` | 12 |
| 7 | 8.6 fewer host round-trips | Gate 7 says so; `toolkit: native` | 6 |
| 8 | 8.9 lossless compression | Gate 7: GEMVs are bandwidth-bound. Step 2 only if step 1's entropy is 85 % or less, with `toolkit: native` | 2 + 8 |
| 9 | 8.8 self-distilled MTP head | 8.1 and 8.2 finished | 15 |
| 10 | 8.10 bigger model | Phase 6 passed, and a Qwen3.5 GGUF with `nextn` tensors fits in 4 GB at Q8_0 | 4 |

**Rules for every experiment**

- The report is `results/8.x-<name>.md`. It uses plan.md's template (Question · Learn first · Prediction · Steps · Measurement · Done when · Stop if · Level and how checked · What I did not test) and adds **What this shows**.
- Commit the prediction before any timing run. Commit the frozen config to `results/8.x/frozen.json` before the test-set run.
- Every lossless claim goes through §7.4. Shadow-mode and exactness tests come before any speed measurement.
- plan.md's **Stop if** is final. "Stopped" is a valid result and gets a full report.
- Each experiment ends with a gate (§5). Its reviews are `reviews/exp-8.x-r<K>.md`.

**Notes for each experiment**

- **8.1:** the tensor-hash check is a script, `analysis/gguf_hash.py`, that compares the bytes of every tensor outside `blk.24` between the two GGUFs. Run each domain's sweep in its own bench session, with its own bracketing controls.
- **8.2:** before the validation plot, commit a prediction of how well coverage predicts acceptance, for example as a rank correlation. If the tool fails plan.md's stop rule, fix it before measuring anything else.
- **8.3:** the sanity check (argmax agrees in more than 99 % of positions) is a machine check. The analysis runs PyTorch on the GPU, so it holds the GPU lock. If 8.2 hasn't produced a ranking yet, 8.3 uses v1's and says so.
- **8.4:** plan.md's "10× more prompts" is `prompts/stress/`, built by a script from WikiText-2, standard-library code and the copy-heavy pattern. Never tune on it. Zero shadow-mode mismatches is a machine check.
- **8.7:** keep the minimal reproducer and every tensor-dump script. If the first op that differs is inside a cuBLAS call, try first a path that avoids cuBLAS, such as the single-token kernel for every column.
- **8.8:** the PyTorch forward pass must match llama.cpp's draft logits (max |Δ| of about 1e-2) before any training starts. That is a machine check. Training data comes only from dev-domain prompts and corpora, never test. The Qwen3.5-0.8B safetensors in the HF cache can cross-check the GGUF reader.
- **8.10:** commit the byte-count prediction before downloading the bigger model.

### Phase 9: Write-up (6 units) ★

plan.md Phase 9, adapted.

1. Each experiment already has its report.
2. **Claims ledger:** `results/claims.md`, generated by a script from a YAML ledger. Each claim lists its evidence file, its level (L1/L2/L3) and whether the test set confirmed it.
3. **Reproduction script:** `scripts/reproduce.sh` clones the repo fresh into `/tmp/ff-repro-<date>`, applies `patches/`, builds, and re-runs the headline configs. Run it once end to end and record the result.
4. **Independent check:** a fresh agent gets only `README.md` and reproduces one headline ratio. It passes if the ratio is within 2 × the noise floor of the published one, measured against the agent's own same-session control.
5. **Audit of v1:** `results/9-flashference-audit.md` marks each claim in v1's README *reproduced in WSL*, *not reproduced* or *not tested*, with evidence. `results/9-v1-readme.patch` is the proposed README change. It isn't applied (decision 2).
6. **`README.md` on `op`:** what was found, the generated headline table, how to reproduce it, and the limits.
7. **`docs/study-guide.md`** (decision 1) gives a reading order through NOTEBOOK.md and the reports. For each phase it has:
   - the 3 key facts, with links to the evidence;
   - the gate questions as a self-quiz, with the graded answers folded inside `<details>`;
   - "try it yourself" commands, with the expected range of output.
8. **Final review** (★) of the whole repo: the claims ledger against the evidence, and `check.py --all`. Then set `status: done-all` and push.

---

## 7. Measuring with nobody watching

### 7.1 Before each bench session

Preflight codes 10 and 20 must be clear. Record GPU temperature, SM and memory clocks, power draw and the driver version. Leave at least 2 minutes after any other GPU job.

### 7.2 Session shape

- A fresh server for each config, 1 warm-up pass, 3 timed passes, 256 tokens (plan.md).
- The control runs first and last. For an A/B comparison, interleave: control, A, B, A, B, control.
- **Validity:**
  - The two control runs must agree within 2 × the noise floor. If they don't, the session is invalid: keep the raw file, mark it `invalid` in its header, and re-run once.
  - A session is also invalid if any request takes more than 3× the median for its config, or if the timestamps show a gap (the laptop slept).
  - After 3 invalid sessions in a row, pause GPU work for 1 hour and do non-GPU units.

### 7.3 Comparisons

Compare only ratios against a control from the same session. v1's Windows numbers appear only as reference ratios, never inside a measured comparison.

### 7.4 Losslessness check (every speculative config)

1. Compare each output's token ids with the reference config's ids for the same prompt and pass. The reference is plain decoding unless the experiment names another.
2. For each mismatch, `bench/divergence.py`:
   - finds the first differing position;
   - starts a plain server **without `-bs`**;
   - sends the prompt plus the shared prefix as token ids, with `n_predict: 1, n_probs: 5`;
   - reads the top tokens at that position. Check once that these probabilities come from the raw logits (`post_sampling_probs` off).
3. A mismatch is **numerics**, which L2 allows, if the speculative token is one of plain's top 2 and the gap between those two, ln(p1/p2), is at most `tau_numerics`. Otherwise it's a **bug**. Until Phase 6 sets `tau_numerics`, every mismatch counts as unexplained.
4. Any bug, or any mismatch in a mode that claims L3, stops the experiment. In plan.md's words: "a lossless claim with one unexplained mismatch is not lossless".
5. Every report gives identical outputs as n/N, plus the counts of numerics mismatches and bugs.

---

## 8. Blocks

`BLOCKED.md` has one entry per block:

```markdown
## 2026-11-04 CUDA toolkit install failed
severity: soft            # hard | soft
what: ...
tried: ...
owner action: <exact command or decision needed, or "none: for information">
meanwhile: <what the agent works on instead>
```

- **Hard blocks stop the executor:**
  - a broken hard rule;
  - the GPU isn't visible in WSL, or the disk is full;
  - a third gate FAIL in Phases 0–6;
  - a bug-class mismatch in the Phase 6 reference;
  - a Phase 0–6 timebox past 1.5×;
  - nothing left in the queue that isn't blocked.
- **Soft blocks are logged, and work continues:**
  - the hybrid-build fallback;
  - no gdb;
  - an empty nsys trace, or Phase 7 ending with no decision;
  - a failed push;
  - any single experiment stopping.
- After fixing a hard block, the owner deletes its entry, or the whole file. The next session checks the fix.

---

## 9. Starting, watching, stopping

**Start.** Open a Claude Code or T3 Code session in `~/code-wsl/flashference` with permission prompts off (bypass or full-access mode, otherwise it will stop and wait at the first prompt), and say:

> You are the executor. Execute autoplan.md.

That is the owner's only action. The first unit is Bootstrap; after that the executor follows §4.0 until it's done or blocked. If the session ends for any reason (closed window, rate limit, crash, reboot), say the same sentence again in a new session: it resumes from STATE.md.

**Watch:** read STATE.md, `reviews/` and the commits on `origin/op`. Or ask, in another session, "summarize STATE.md".

**Stop:** create an empty file named `STOP` in the repo root (or just close the session). The executor finishes the current unit, deletes `STOP` and ends.

**While it runs:**
- Run only one executor at a time, and don't edit this checkout while it works.
- If you need the GPU, just use it: preflight waits until the GPU is idle.
- If Windows sleeps, the run pauses and resumes on wake. Any bench session caught by the sleep is thrown out (§7.2).
- With permission prompts off, the agent can run any command your user account can run. The hard rules and `guard.sh` reduce accidents, but they are not a sandbox.

**Budget:** the timeboxes add up to about 130 work units plus review rounds. STATE.md keeps a running unit count; the executor logs a hard block at 200 units. Expect tens of GPU hours, spread over several days.

---

## 10. Done means

- Phases 0–6 each have a gate PASS with its review files. Phase 7 has a PASS, or `Gate 7: no decision` with its soft block.
- Every Phase 8 item has a report with a final status: done, stopped (with the reason) or skipped (with the condition).
- The Phase 9 files exist, `scripts/reproduce.sh` was run from a fresh clone, the independent check passed, and the final review says PASS.
- `python autopilot/check.py --all` passes, STATE.md says `status: done-all`, and `origin/op` has every commit.
