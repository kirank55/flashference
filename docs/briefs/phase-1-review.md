# Gate 1 review brief (filled in from autopilot/review-brief.md)

This is the task text the executor gave the Gate 1 reviewer on 2026-11-01. The first attempt
(opencode/deepseek-v4.1-flash-free) ended at once with "Free usage limit reached" and wrote nothing.
Any reviewer can be started with this text as it stands. Change the report file name for later rounds.

---

You are the independent reviewer of one phase of an autonomous research run. Another agent (the
executor) did the work. Don't trust anything it wrote in NOTEBOOK.md until you have checked it
against raw files or source code. Your context is fresh on purpose.

Repo: /home/kiran/code-wsl/flashference, branch `op`, on WSL2 Ubuntu (RTX 3050 Laptop GPU, 8 GB RAM).
Start every shell command with `cd /home/kiran/code-wsl/flashference && source autopilot/env.sh && ...`
(env.sh puts .venv's python and the CUDA libraries on PATH and sets HF_HUB_OFFLINE=1).

## The brief (autopilot/review-brief.md, filled in)
Inputs:
- the plan.md section "## Phase 1: How an LLM produces one token" through its "### Gate 1";
- the autoplan.md section "### Phase 1: One token (4 units)", together with autoplan.md §1 (hard
  rules, especially science rules 10–16), §4.3–§4.4, §5 and §7;
- `git diff 9068467..HEAD` (phase_start_commit = 9068467);
- the raw files under results/1/;
- in NOTEBOOK.md: the gate answers (section "## Gate 1 answers"), the Phase 1 entries ("Phase 1
  predictions", "Phase 1.1", "1.2", "1.3–1.4", "1.5"), and the prediction log near the top
  (P-008..P-013).

Steps:
1. Run `python autopilot/check.py --phase 1` (about 2 minutes; it takes the GPU lock itself). Pick one
   check and confirm it fails on a deliberately broken input. Break a COPY, never a repo file:
   `mkdir -p /tmp/review-1 && cp -r results/1 /tmp/review-1/broken`, edit one value under
   "deterministic" in the copy, run `FF_P1_RESULTS=/tmp/review-1/broken python autopilot/check.py --phase 1`,
   and confirm the FAIL names that key.
2. Recompute every headline number from the raw files with your own short script, not the
   executor's: token counts and ratios, the greedy medians and ratios, parameter and byte counts,
   the embedding shares, the GGUF totals and per-block bytes, the chi-square result. Put your
   scripts in /tmp/review-1/.
3. Grade each of the 5 Gate 1 answers CORRECT, INCOMPLETE or WRONG, with the reason and the
   file:line or data that shows it.
4. Check every losslessness/equality claim; here that is 1.2's "three methods give identical ids".
5. Look for:
   - numbers with no source or no tag ([meas] / [derived] / [est] / [unv]), and [est] or [unv]
     numbers used as facts;
   - predictions written after their result: commit order, and check.py --predictions;
   - test-set use, and two changes measured as one;
   - deviations without a reason (STATE.md "Deviations");
   - a report missing its "What I did not test".

List BLOCKING and NON-BLOCKING issues. Don't fix anything. End the report with exactly one line:
`verdict: PASS` or `verdict: FAIL`.

## Phase-specific instructions
- autoplan.md requires this for Gate 1. Compare the deterministic numbers (GGUF byte counts,
  parameter counts, vocabulary size) with an earlier independent run of the same phase on branch v2:
  `git show v2:results/phase1/gguf.json` and `git show v2:results/phase1/params.json`. This is an
  independent replication. A mismatch is BLOCKING until it's explained (for example, different
  units or definitions). From v2, read only those two files and `git show v2 --stat`. The executor
  ran `git show v2 --stat | head -40 && git ls-tree -r --name-only v2 | head -50` at bootstrap and
  says that output showed file names and line counts but no measured numbers (NOTEBOOK.md, round-2
  entry). Confirm or refute that.
- The executor says no Phase 1 code ran before the prediction commit a2b4e25
  (2026-11-01T11:12:16Z). Check:
  - commit order;
  - the mtimes of the dev-run outputs in /tmp/ff-p1-dev, /tmp/ff-p1-dev0, /tmp/ff-p1-broken and
    /tmp/ff-p1-short;
  - the job record in results/_jobs/phase1-runs/.

  Optionally, the executor's session log at
  ~/.claude/projects/-home-kiran-code-wsl-flashference/61205b4c-6083-48e7-b0b0-0414f579b398.jsonl
  lists its tool calls.
- Phase 1's code was written by an opencode child, at the owner's request, and reviewed and changed
  by the executor (STATE.md Deviations; commit a6c843c). Review that code like any other. P-012(a)
  restates totals the executor had seen in Phase 0, and the entry says so. Judge whether the other
  predictions were really derived beforehand (toy/phase1_predict.py, committed with them).
- Timing context: 1.2 times with CLOCK_MONOTONIC_RAW because this machine's CLOCK_MONOTONIC runs
  fast (STATE.md Decisions, BLOCKED.md). Don't run new timings. If your review needs one, say so in
  the report.

## Hard rules for the reviewer
- The only file you may create or change is reviews/phase-1-r1.md. Never edit, move or delete any
  other file in the repo.
- No git write commands of any kind: no add, commit, restore, checkout, switch, reset, stash,
  clean, branch, tag, push, worktree or config. Read-only git (log, show, diff, status, ls-tree,
  cat-file) is fine.
- Anything that imports torch or transformers, or runs a llama.cpp binary, runs under the GPU lock:
  prefix it with `flock /tmp/flashference-gpu.lock`. Never run two at once. check.py takes the lock
  itself.
- Never open, list or print anything under prompts/test/. No sudo, no installs, no downloads.
  Never write under /mnt/c.
- Finish with `git status --short` showing only `?? reviews/phase-1-r1.md`.

When you finish, reply with the verdict line, then a short list of the BLOCKING issues (if any) and
the most important NON-BLOCKING ones, in under 40 lines. The full report goes in
reviews/phase-1-r1.md.

---

## Second reviewer (owner's rule, 2026-11-01)

A second, independent review of the same phase, by a different model family from both the author of
Phase 1's code (deepseek-v4.1-flash) and the first reviewer (deepseek-v4.1-flash). Everything
above applies, with these changes:
- Write your report to reviews/phase-1-r1-second.md (not reviews/phase-1-r1.md), and finish with
  `git status --short` showing only `?? reviews/phase-1-r1-second.md`.
- Use /tmp/review-1b/ for your scratch files, not /tmp/review-1/ (that directory is the first reviewer's).
- Don't read reviews/phase-1-r1.md or /tmp/review-1/ until your own grades and verdict are written.
  NOTEBOOK.md shows the first reviewer's grades ("Grade r1") and a "Gate 1 review round 1" entry.
  Ignore them and form your own judgment from the raw data. After your verdict, you may add a final
  section listing where you and the first review disagree.
- The executor has since corrected the text after re-checking it ("Gate 1: re-check of review r1"
  entry). Review the corrected text, and check that the corrections are themselves right.
- Be adversarial. The owner wants a rigorous review: check every number against the raw files, every
  sentence's claim against the data, every tag, and the code line by line. Re-derive results with
  your own scripts and compare them with the published tables, row by row.
- v2: read only `git show v2:results/phase1/gguf.json`, `git show v2:results/phase1/params.json` and
  `git show v2 --stat`. Nothing else from v2, not even its NOTEBOOK.md.
