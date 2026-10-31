# NOTEBOOK

Journal of branch `op`. An executor agent carries out plan.md under autoplan.md, with nobody watching. Entries are dated and in order, oldest first. This file is also the owner's study trail: each entry ends with **What this shows**, written for a reader who knows basic Python but no C++, CUDA or ML maths.

Every number in prose carries a tag and a source file:
**[meas]** read from a raw file · **[derived]** arithmetic on measured numbers, formula shown · **[est]** an estimate · **[unv]** seen but not recorded.

## Prediction log

One line per prediction (autoplan.md §5.2). Fields are separated by ` | `, so the prediction text never contains a `|`. A line is committed with `got __ | error __ | learned __` before the run that tests it, and `evidence:` names the exact file(s) that run will write. `python autopilot/check.py --predictions` checks the order from git history.

P-001 | 2026-10-31 | 0.6 | llama-bench tg128 (bf16, -ngl 99 -fa on, Release build, CUDA 12.8) averages 68 tok/s (range 62–74) over the 3 runs, because v1's WSL smoke measured plain server decode at 63.7–64.9 tok/s and llama-bench skips the server's HTTP and sampling work (on Windows, v1 saw llama-bench about 71 against server 65–68) | evidence: results/0/llama-bench-1.json, results/0/llama-bench-2.json, results/0/llama-bench-3.json | got __ | error __ | learned __
P-002 | 2026-10-31 | 0.6 | the 3 runs' mean tok/s agree within 0–2 % (max/min − 1), because they run back to back on an idle GPU and v1 saw its ~5 % drift between sessions, not within one | evidence: results/0/llama-bench-1.json, results/0/llama-bench-2.json, results/0/llama-bench-3.json | got __ | error __ | learned __
P-003 | 2026-10-31 | 0.6 | within each run, the SD of the 5 repetitions is 0–1 % of that run's mean, because each repetition decodes the same 128 tokens from an empty context | evidence: results/0/llama-bench-1.json, results/0/llama-bench-2.json, results/0/llama-bench-3.json | got __ | error __ | learned __

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

## 2026-10-31 — Phase 0.1–0.4 Workbench: toolkit, Python, model
**Plan:** autoplan.md Phase 0 with plan.md 0.1–0.4. No sudo is available, so the CUDA toolkit goes into user space.
**Done:**
- 0.1 skipped (autoplan.md §0): `.wslconfig` already gives WSL 8 GB of RAM and 8 GB of swap.
- 0.2 CUDA 12.8.1 toolkit from NVIDIA's runfile, installed with `--toolkit --toolkitpath=$HOME/opt/cuda-12.8`. The runfile's md5 `e281945bf4ea42788db29457ebc1e6ae` matched NVIDIA's `md5sum.txt`, which I fetched again on 2026-10-31 and which agrees with autoplan.md. The installer only warned that it couldn't write to `/var/log` and `/usr/local`, which a non-root install doesn't need. `nvcc --version` reports release 12.8, V12.8.93. `toolkit: native`, so no fallback was needed. The toolkit includes Nsight Systems 2024.6.2 (`~/opt/cuda-12.8/bin/nsys`), which Phase 7 needs. `ninja` 1.13.2 came from `uv pip install ninja` into `.venv`.
- A 10-line CUDA test program, compiled with `nvcc -arch=sm_89` and run under WSL, ran a kernel without error. The driver reports the device as `cc 8.9, 14 SMs, bus 64 bits, mem clock 8001000 kHz` [unv: printed to the terminal, not saved; Phase 2.1 records the bus width with a cited source].
- 0.3 added `pytest matplotlib scikit-learn` to `.venv` (matplotlib was already there). The full list is in `results/env/pip-freeze.txt`: torch 2.15.0, transformers 5.22.0, gguf 0.19.0, datasets 5.1.0, numpy 2.5.4. With `autopilot/env.sh` loaded, torch still reports `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU`. Putting CUDA 12.8's `lib64` on `LD_LIBRARY_PATH` doesn't disturb torch, because torch's CUDA 13 libraries have different file names (`libcublas.so.13`).
- 0.4 the GGUF's sha256 is `ad1549ee…07bbc2` (`results/0/model-sha256.txt`). It matches plan.md's value exactly, so this is the same file v1 measured. `HuggingFaceTB/SmolLM2-135M-Instruct` is now in the HF cache: `model.safetensors` is 269,060,552 bytes [meas: `ls -L`, not saved]. The cache keeps weights behind two levels of symlinks, so `du` without `-L` reports only 3.5 MB.
- 0.5 configure: `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc` (`results/0/cmake-configure.log`). The log reports CUDA 12.8.93, CUDA graphs on and the FlashAttention kernels on.

Two departures from plan.md 0.5, both recorded under Deviations in STATE.md:
- **Release** instead of RelWithDebInfo (autoplan.md 0.5), so every measured number comes from binaries built the same way.
- `-DLLAMA_OPENSSL=OFF`. b11884 deprecates `LLAMA_CURL` and fetches over HTTPS with OpenSSL instead. This machine has no OpenSSL headers, and installing them needs apt. The flag only removes HTTPS model downloads from the binaries. It doesn't touch any compute code.

**What this shows:** the GPU can be driven from Linux without installing anything system-wide. On WSL2 the Windows driver provides the GPU, and the toolkit is just a compiler plus libraries, so it can live in a home directory. The model file is byte-identical to the one v1 measured on Windows (same sha256). Any speed difference later is therefore down to the platform or the code, not the model. The Python stack has a CUDA-enabled torch for the Phase 1–3 toy experiments, which run separately from llama.cpp.
