# NOTEBOOK

Journal of branch `op`. An executor agent carries out plan.md under autoplan.md, with nobody watching. Entries are dated and in order, oldest first. This file is also the owner's study trail: each entry ends with **What this shows**, written for a reader who knows basic Python but no C++, CUDA or ML maths.

Every number in prose carries a tag and a source file:
**[meas]** read from a raw file · **[derived]** arithmetic on measured numbers, formula shown · **[est]** an estimate · **[unv]** seen but not recorded.

## Prediction log

One line per prediction (autoplan.md §5.2). Fields are separated by ` | `, so the prediction text never contains a `|`. A line is committed with `got __ | error __ | learned __` before the run that tests it, and `evidence:` names the exact file(s) that run will write. `python autopilot/check.py --predictions` checks the order from git history.

P-001 | 2026-10-31 | 0.6 | llama-bench tg128 (bf16, -ngl 99 -fa on, Release build, CUDA 12.8) averages 68 tok/s (range 62–74) over the 3 runs, because v1's WSL smoke measured plain server decode at 63.7–64.9 tok/s and llama-bench skips the server's HTTP and sampling work (on Windows, v1 saw llama-bench about 71 against server 65–68) | evidence: results/0/llama-bench-1.json, results/0/llama-bench-2.json, results/0/llama-bench-3.json | got 70.60 tok/s raw mean of the 3 run means; 68.32 after screening out one clock-hit repetition | error +3.8 % raw, +0.5 % screened | learned v1's WSL server smoke was a good prior; both values sit inside the range, and the raw one only because one repetition was wrong
P-002 | 2026-10-31 | 0.6 | the 3 runs' mean tok/s agree within 0–2 % (max/min − 1), because they run back to back on an idle GPU and v1 saw its ~5 % drift between sessions, not within one | evidence: results/0/llama-bench-1.json, results/0/llama-bench-2.json, results/0/llama-bench-3.json | got 10.31 % raw (miss); 0.32 % screened | error miss raw, inside screened | learned run 2's 4th repetition was cut 0.622 s short by a wall-clock step (P-004); llama-bench on Linux times with the wall clock
P-003 | 2026-10-31 | 0.6 | within each run, the SD of the 5 repetitions is 0–1 % of that run's mean, because each repetition decodes the same 128 tokens from an empty context | evidence: results/0/llama-bench-1.json, results/0/llama-bench-2.json, results/0/llama-bench-3.json | got 0.32 %, 20.32 %, 0.18 % raw (run 2 a miss); worst 0.32 % screened | error miss for run 2 raw, inside screened | learned same cause as P-002; without the step, 5 repetitions agree within ~0.3 %
P-004 | 2026-10-31 | 0.6 | over 120 s, CLOCK_REALTIME − CLOCK_MONOTONIC changes only in 3–4 discrete steps spaced 31–32 s apart, each backward by 0.4–0.8 s, because systemd-resolved logs a clock change every ~31.5 s and run 2's 4th repetition came out 0.622 s short | evidence: results/0/clock-steps.json | got 4 steps in 123.6 s, each back 0.591–0.631 s; 32.19–32.30 s apart by the guest's monotonic clock | error count and size inside; spacing 0.2–0.3 s above the range by the guest clock, but ~31.6 s by the host clock [derived: 32.25 / 1.0194] | learned a duration needs a named clock; here guest seconds are ~1.9 % short
P-005 | 2026-10-31 | 0.6 | over the same 120 s the guest's CLOCK_MONOTONIC advances 1.3–2.5 % more than the Windows host clock, because steps of ~0.62 s every ~31.5 s mean the guest clock gains about 2 % and time sync pulls it back | evidence: results/0/clock-steps.json | got +1.94 % | error −0.06 points from the ~2 % I reasoned from | learned the cause is the kernel tick (10188–10191 µs instead of 10000, results/0/timesync-snapshot.txt), and CLOCK_MONOTONIC_RAW, which ignores the tick, keeps host time

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
- 0.4 the GGUF's sha256 is `ad1549ee…07bbc2` (`results/0/model-sha256.txt`). It matches plan.md's value exactly, so this is the same file v1 measured. `HuggingFaceTB/SmolLM2-135M-Instruct` is now in the HF cache: `model.safetensors` is 269,060,552 bytes [unv: `ls -L`, not saved]. The cache keeps weights behind two levels of symlinks, so `du` without `-L` reports only 3.5 MB.
- 0.5 configure: `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc` (`results/0/cmake-configure.log`). The log reports CUDA 12.8.93, CUDA graphs on and the FlashAttention kernels on.

Two departures from plan.md 0.5. The first is autoplan.md's own rule; the second is recorded under Deviations in STATE.md:
- **Release** instead of RelWithDebInfo (autoplan.md 0.5), so every measured number comes from binaries built the same way.
- `-DLLAMA_OPENSSL=OFF`. b11884 deprecates `LLAMA_CURL` and fetches over HTTPS with OpenSSL instead. This machine has no OpenSSL headers, and installing them needs apt. The flag only removes HTTPS model downloads from the binaries. It doesn't touch any compute code.

**What this shows:** the GPU can be driven from Linux without installing anything system-wide. On WSL2 the Windows driver provides the GPU, and the toolkit is just a compiler plus libraries, so it can live in a home directory. The model file is byte-identical to the one v1 measured on Windows (same sha256). Any speed difference later is therefore down to the platform or the code, not the model. The Python stack has a CUDA-enabled torch for the Phase 1–3 toy experiments, which run separately from llama.cpp.

## 2026-10-31 — Phase 0.5–0.6 Build and smoke test
**Plan:** Release build of stock b11884 (autoplan.md 0.5), then three `llama-bench -m models/Qwen_Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa on -p 0 -n 128 -r 5 -o json` runs (0.6), each bracketed by GPU readings.
**Prediction:** P-001–P-003, committed in 2e08d41 before the build existed.
**Build:** 387 steps in 13 min 12 s at `-j 4`, exit 0, no compiler warnings [meas: results/0/build-stock-job.txt]. `llama-server --version` prints `version: 0.7.0-dev (build 11884, commit 63412d873)` [meas: results/0/toolchain.txt]. In this release, the executables are 17.9 KB launchers. The code is in shared libraries next to them (`libllama-server-impl.so`, `libllama.so`, `libggml-cuda.so`, ...), so `toolchain.txt` records the sha256 of every library too. `libcuda.so.1` resolves to WSL's `/usr/lib/wsl/lib`, and cuBLAS/cudart to `~/opt/cuda-12.8`.
**Result:** the GPU was idle and on AC before every run (results/0/llama-bench-env.jsonl). The JSON shows `backends: CUDA`, `n_gpu_layers: 99`, `build_commit: 63412d873`.
<!-- gen: python bench/phase0_report.py -->
| run | test_time (UTC) | tok/s, all 5 reps | SD %, all 5 | clock changes in run window | excluded rep (tok/s) | tok/s, kept reps | SD %, kept |
|---|---|---:|---:|---|---|---:|---:|
| llama-bench-1.json | 2026-10-31T09:35:43Z | 68.40 | 0.32 | 09:35:43.111 | none | 68.40 | 0.32 |
| llama-bench-2.json | 2026-10-31T09:36:08Z | 75.21 | 20.32 | 09:36:14.740 | #4 (102.56) | 68.38 | 0.20 |
| llama-bench-3.json | 2026-10-31T09:36:33Z | 68.19 | 0.18 | none | none | 68.19 | 0.18 |
<!-- /gen -->
<!-- gen: python bench/phase0_report.py --summary -->
- raw, all 15 repetitions: mean of run means 70.60 tok/s; between-run spread (max/min − 1) 10.31 %; largest within-run SD 20.32 % of the run mean
- screened: mean of run means 68.32 tok/s; between-run spread (max/min − 1) 0.32 %; largest within-run SD 0.32 % of the run mean
- repetitions excluded: 1 of 15
<!-- /gen -->

**The outlier, and what caused it.** Repetition 4 of run 2 took 1.248 s; the other 14 took 1.865–1.881 s [meas: samples_ns in results/0/llama-bench-{1,2,3}.json]. Decode speed is set by memory bandwidth, and the memory clock read 8000 MHz before and after every run, so a real 50 % speed-up for one repetition isn't plausible. The machine's clock is the explanation:
- `llama-bench` times a repetition as the difference of two `std::chrono::high_resolution_clock` readings (`tools/llama-bench/llama-bench.cpp:40`, `get_time_ns`). In GCC's libstdc++ that clock is `system_clock`, the wall clock. ggml's own timer, which `llama-server`'s timings use, reads `CLOCK_MONOTONIC` (`ggml/src/ggml.c:576`, `ggml_time_us`), which is never stepped.
- `systemd-timesyncd` polls an NTP server every 32 s and finds the clock about 0.6 s ahead each time: offset −665.3 ms [meas: results/0/timesync-snapshot.txt], and −587.7 ms in an earlier reading [unv]. The wall clock steps back by that much on the same 32 s rhythm, so timesyncd is very likely what steps it. `systemd-resolved` logs every step; one is at 09:36:14.740, inside run 2 [meas: results/0/timesync-snapshot.txt].
- `test_time` is stamped after the model loads, and is followed by a 1-token warm-up and then the repetitions. Run 2 has `test_time` 09:36:08, so repetition 4 started 5.7–6.9 s later, roughly 09:36:13.7–09:36:14.9, and lasted ~1.87 s. On the pre-step clock, the step logged at 09:36:14.740 happened at about 09:36:15.35, inside repetition 4 and outside repetitions 3 and 5. The repetition lost 0.622 s [derived: 1.870 − 1.248], the same size as the steps measured below.
- I committed P-004 and P-005 before measuring the clock directly (`bench/clock_steps.py`):
<!-- gen: python bench/phase0_report.py --clock -->
| quantity | value |
|---|---:|
| window (guest CLOCK_MONOTONIC) | 123.6 s |
| wall-clock steps | 4 |
| step sizes | -0.591 s, -0.598 s, -0.631 s, -0.617 s |
| time between steps (guest monotonic) | 32.27 s, 32.30 s, 32.19 s |
| host elapsed / guest monotonic elapsed | 120.209 s / 122.544 s |
| guest CLOCK_MONOTONIC rate vs Windows clock | +1.94 % |
| CLOCK_MONOTONIC gain over CLOCK_MONOTONIC_RAW in the window | 2.362 s (+1.95 %) |
<!-- /gen -->
- The clock runs fast because the kernel's `tick` is 10188–10191 µs per 10 ms tick instead of 10000: tick 10189 and freq +91.5 ppm in results/0/timesync-snapshot.txt [meas]. A 70 s watch saw tick change 10188 → 10191 and freq swing between −95 and +91 ppm [unv: printed, not saved]. The tick alone makes the clock gain 1.88–1.91 % [derived: tick / 10000 − 1]. `CLOCK_MONOTONIC` and the wall clock both follow the tick; `CLOCK_MONOTONIC_RAW` doesn't. Over the host window, RAW agrees with the Windows clock to −0.005 % [derived: 122.544 s × (121.262 / 123.624) = 120.203 s RAW against 120.209 s host, assuming the RAW/MONOTONIC ratio holds steady through the window]. I don't know what set the tick; timesyncd's log doesn't say.

**What follows for the rest of the run** (recorded in STATE.md under Decisions):
1. Ratios measured in one session are safe: the same ~1.9 % rate error divides out. It changed by only ~0.03 % (tick 10188 → 10191) while I watched.
2. Absolute rates from `CLOCK_MONOTONIC` (Python's `time.perf_counter`, llama-server's `timings`) read ~1.9 % slow on this machine as it is today. Phase 2's bandwidth uses CUDA events, which time on the GPU's own clock. Gate 2's first-principles tok/s gets compared with Phase 4's server number corrected by the rate measured in that session.
3. `llama-bench` repetitions can be cut short by ~0.6 s. Any llama-bench number gets the screening rule in `bench/phase0_report.py`, and raw numbers are always shown next to screened ones.
4. The screened mean, 68.32 tok/s, is still a wall-clock figure. On host time it would be about 69.6 tok/s [est: × 1.0194, using the rate measured three hours after the runs].

**Prediction error:** P-001 inside its range (+0.5 % screened). P-002 and P-003 missed on the raw data. The miss is explained: one repetition hit by a clock step. P-004 was right on count and size but missed on spacing, because I didn't say which clock; P-005 inside. See the prediction log.
**What I now understand:** a timer is only as good as the clock behind it. On this WSL2 instance the wall clock jumps and even the monotonic clock runs ~1.9 % fast. Every timing tool used from here on must say which clock it reads.
**Open questions:** What set `tick` to 10188? Does it survive a WSL restart? Is v1's Windows-to-WSL gap (v1 measured WSL 1.4–6.6 % slower) partly this clock error? That needs a Windows-side check, out of scope here.

**What this shows:** a small, fully on-GPU benchmark gives very repeatable numbers on this laptop: 5 repetitions of 128 tokens agree within about 0.3 %, and three back-to-back runs agree within 0.32 %. The bf16 Qwen3.5-0.8B decodes about 68 tokens a second on this GPU under WSL, close to v1's ~71 on Windows. The one wild repetition wasn't the GPU at all. The Linux clock inside WSL is stepped back about 0.6 s every half minute and runs about 1.9 % fast in between, and `llama-bench` happened to use that clock. The lesson for every later phase: before trusting a number, ask which clock produced it, and keep the raw data so a bad reading can be found and explained rather than averaged in.

**What I did not test:**
- `llama-bench` with `-p` > 0 (prompt processing) or any other flag set; the Windows side; a RelWithDebInfo build.
- Whether the clock problem survives a WSL restart. That needs `wsl --shutdown`, which only the owner may run.
- Whether `CLOCK_MONOTONIC_RAW` is accurate against the GPU's timer. The Windows clock is the only reference used.
- Whether llama-server's own `timings` see the same ~1.9 % rate error. They read `CLOCK_MONOTONIC`, so they should; Phase 4 measures it.

## Gate 0 answers
plan.md's Gate 0 is a checklist, not questions. Each item, with its evidence (the reviewer's grade goes after each):
1. **`nvidia-smi`, `nvcc --version` and `torch.cuda.is_available()` all work in WSL.** Yes. In results/0/toolchain.txt, `nvcc` reports release 12.8, V12.8.93. `nvidia-smi` lists the NVIDIA GeForce RTX 3050 A Laptop GPU, compute capability 8.9, driver 592.82. torch prints `2.15.0+cu132 True NVIDIA GeForce RTX 3050 A Laptop GPU`. `python autopilot/check.py --phase 0` runs all three again, live.
2. **llama-bench runs the bf16 model on the GPU, and the number and the model's sha256 are written down.** Yes. The JSON reports `backends: CUDA`, `n_gpu_layers: 99` and `gpu_info: NVIDIA GeForce RTX 3050 A Laptop GPU` (results/0/llama-bench-1.json). tg128 is 68.32 tok/s after screening out one clock-hit repetition, and 70.60 tok/s raw [derived: `bench/phase0_report.py --summary` over results/0/llama-bench-{1,2,3}.json; table in the Phase 0.5–0.6 entry]. The sha256 is `ad1549eedc613064971dcbbbfab6c9b7990984d1c9ab38f792c6f2ec1207bbc2` (results/0/model-sha256.txt), the same as plan.md's value.
3. **`~/code-wsl/flashference` is a git repo with your first notebook entry committed.** Yes: branch `op`, with NOTEBOOK.md's Bootstrap entry in commit 03efb2f, pushed to `origin/op`.

autoplan.md's extra Phase 0 checks: `llama-server --version` reports `version: 0.7.0-dev (build 11884, commit 63412d873)` (results/0/toolchain.txt), and STATE.md says `toolkit: native`.
