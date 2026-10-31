# Review brief (autoplan.md §5.3)

The executor starts each reviewer with this text, filling in the fields in angle brackets.

> You're reviewing work another agent did. Don't trust anything in NOTEBOOK.md until you've checked it against raw files or source code.
> Inputs: the plan.md and autoplan.md sections for <phase>; `git diff <phase_start_commit>..HEAD`; the raw files under `results/<phase>/`; the gate answers in NOTEBOOK.md.
> 1. Run `python autopilot/check.py --phase <N>`. Pick one check and confirm it fails on a deliberately broken input.
> 2. Recompute every headline number from the raw files with your own short script, not the executor's.
> 3. Grade each gate answer CORRECT, INCOMPLETE or WRONG. Give the reason and the file:line or data that shows it.
> 4. Check every losslessness claim against the hash and divergence data.
> 5. Look for numbers with no source, [est] or [unv] numbers used as facts, predictions written after the result, test-set use, two changes measured as one, deviations without a reason, and a missing "What I did not test".
> List BLOCKING and NON-BLOCKING issues. Don't fix anything. End with one line: `verdict: PASS` or `verdict: FAIL`.

## Practical notes for the reviewer

- Repo: `~/code-wsl/flashference`, branch `op`. Run `source autopilot/env.sh` first: it puts the user-space CUDA toolkit and `.venv` on PATH and sets `HF_HUB_OFFLINE=1`.
- Write your report to `reviews/<report file>`. That is the only file you create. Don't edit, commit or push anything else. If you break a file on purpose for step 1, put it back (`git restore <file>`), and finish with `git status` showing only your report as new.
- Never open or print files in `prompts/test/`.
- Anything that uses the GPU holds the lock: `flock /tmp/flashference-gpu.lock <command>`.
- Your own scripts go in a scratch directory outside the repo (for example `/tmp/review-<phase>/`).
- Phase-specific instructions: <extra>
