# STATE
updated: 2026-10-31T09:25Z   session: 1
status: running              # running | gate-review | blocked | done-all
phase: 0   step: 0.2 CUDA toolkit
phase_start_commit: (the bootstrap commit; filled in by the first Phase 0 commit)
next: install the CUDA 12.8.1 toolkit into ~/opt/cuda-12.8 from the runfile (download job dl-cuda)
timebox: phase 0 used 0/3 units
units: 1 of the 200-unit budget (bootstrap)
running_jobs: dl-cuda (no GPU lock; ~/opt/dl/cuda_12.8.1_570.124.06_linux.run)
toolkit: (not installed yet)
noise_floor: (set by Gate 4, with its definition)
tau_numerics: (set in Phase 6, with its evidence file)
second_reviewer: off
## Done
- Bootstrap B1–B6 (this commit)
## Decisions
## Deviations
- 2026-10-31 `job.sh --no-lock`: §4.3 says job.sh takes the GPU lock. Downloads and CPU-only builds don't touch the GPU, so they run detached without it and GPU units aren't held up. GPU jobs still always take the lock.
- 2026-10-31 `preflight.sh --gpu`: the dirty-tree check (code 30) is for session start. Just before a GPU run, mid-unit, the tree is legitimately dirty, so `--gpu` skips that one check. The GPU-busy test uses the median of the 5 samples, so one blip from the Windows desktop doesn't count as busy.
## Open questions carried forward
- Does the guard hook load in this same session, or only in sessions started after `.claude/settings.json` exists? Checked by `sudo -n true` right after bootstrap (see NOTEBOOK).
