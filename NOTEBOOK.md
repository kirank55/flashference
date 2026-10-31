# NOTEBOOK

Journal of branch `op`. An executor agent carries out plan.md under autoplan.md, with nobody watching. Entries are dated and in order, oldest first. This file is also the owner's study trail: each entry ends with **What this shows**, written for a reader who knows basic Python but no C++, CUDA or ML maths.

Every number in prose carries a tag and a source file:
**[meas]** read from a raw file · **[derived]** arithmetic on measured numbers, formula shown · **[est]** an estimate · **[unv]** seen but not recorded.

## Prediction log

One line per prediction (autoplan.md §5.2). Fields are separated by ` | `, so the prediction text never contains a `|`. A line is committed with `got __ | error __ | learned __` before the run that tests it, and `evidence:` names the exact file(s) that run will write. `python autopilot/check.py --predictions` checks the order from git history.

## Entries

## 2026-10-31 — Bootstrap
**Plan:** autoplan.md B1–B6: copy plan.md from `v2`, fix the `op` branch, recreate the v1 reference, write the autopilot tools, commit and push.
**Done:**
- B1: `plan.md` is `git show v2:plan.md`, byte for byte. Nothing else came from `v2`.
- B2: dropped the `llama.cpp` gitlink (`git rm --cached`; the checkout stays), deleted the stale `toy/__pycache__/`, and wrote the `.gitignore` from B2.
- B3: `ref/flashference` is a detached worktree of `v1.0` (commit a9bcb57). `prompts/v1/` holds v1's two prompt files, copied unchanged. `e-prompts.json` starts with a UTF-8 byte-order mark (v1 was written on Windows), so Python must read it with `encoding="utf-8-sig"`.
- B4: `autopilot/env.sh`, `preflight.sh`, `job.sh`, `check.py` (global and bootstrap checks) and `review-brief.md`.
- B5: `autopilot/guard.sh` is registered in `.claude/settings.json` as a `PreToolUse` hook on `Bash`. It splits each command into shell words, drops heredoc bodies, and checks only command positions, so `LLAMA_MTP_ADAPT=1` and the word "adapt" pass while `sudo` is blocked. `check.py --phase B` runs it against a table of allowed and blocked commands.
- `job.sh --no-lock` is an addition to autoplan.md §4.3. Downloads and CPU builds run detached like GPU jobs, but without holding the GPU lock, since they don't use the GPU.

**What this shows:** before any measuring starts, the run sets up the rules that keep its numbers honest. Every prediction is committed to git before the run that tests it, so the order can be proved later from history rather than taken on trust. Tables are produced by scripts and re-checked by `check.py`, so no number is typed by hand. The guard hook doesn't make the agent safe on its own; it catches accidents, such as a stray `git push --force` or a `sudo`, before they happen.
