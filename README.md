# slmspeed - fast code and copy-heavy generation for a 0.8B model

**Up to 1.9x faster code generation and ~2.8-3.0x faster copy-heavy generation for Qwen3.5-0.8B (bf16, not quantized) in llama.cpp, using the model's own multi-token-prediction (MTP) head as a speculative-decoding drafter. The full bf16 model verifies every token.**

Measured on a laptop GPU (RTX 3050 Laptop, 4 GB), single stream, greedy decoding, 256 new tokens. Absolute tokens/s are specific to that GPU; the **ratios** are what should carry over to other hardware (see [Hardware portability](#hardware-portability)).

## Headline results

| Workload | Config | Plain bf16 | With speculation | Speed-up | Draft acceptance |
|---|---|---|---|---|---|
| **Code** (4 prompts: two Python, one JavaScript, one SQL) | MTP drafter, output vocabulary trimmed to the 32k most common tokens, 2 drafts, `-bs` | 66.5 tok/s | **124.9 tok/s** (+/- 0.1) | **1.88x** | 68.5 % |
| **Copy-heavy** (6 prompts whose output mostly copies the prompt: spelling fix, JSON field edit, bug fix in a program, verbatim quotes, email rewrite, list conversion) | `--spec-type ngram-simple,draft-mtp` (n-gram lookup first, MTP second) | 67.6 tok/s | **201.1 tok/s** (+/- 1.0) | **2.97x** | 58.5 % |

Notes:

- "Plain" is the same bf16 model without speculation, measured in the same session on the *same prompt set* (plain speed drifts by roughly 5 % between sessions: 65-68 tok/s, so each ratio uses its own control).
- Copy-heavy with MTP alone (no n-gram drafter) is 146.7 tok/s = 2.17x; the n-gram drafter adds another 1.37x on top.
- A fresh re-run with `scripts/reproduce.ps1` on 2026-10-27 gave 124.3 tok/s (1.84x) for code and 189.2 tok/s (2.89x against that run's noisy plain control, 2.80x against the recorded one) for copy-heavy, i.e. the copy-heavy figure is **about 2.8-3.0x, 189-201 tok/s**, depending on machine state. See [`results/2026-10-27-reproduce-check.md`](results/2026-10-27-reproduce-check.md).
- Prompt sets are small (4 code, 6 copy-heavy prompts) and the model, GPU and OS are a single configuration. Treat the numbers as a well-measured case study, not a benchmark suite.
- Raw data: [`results/2026-10-27-bf16-E1-E5.md`](results/2026-10-27-bf16-E1-E5.md) (code and copy-heavy numbers), [`results/2026-10-26-bf16-b4-trim-draft.md`](results/2026-10-26-bf16-b4-trim-draft.md) (the draft-head trim), [`results/2026-10-26-bf16-b0-b3.md`](results/2026-10-26-bf16-b0-b3.md) (baseline and untrimmed MTP), [`results/e-refs.txt`](results/e-refs.txt), raw JSON next to each report.

How the code number depends on the trim size (same code prompts, same session):

| Draft head | tok/s | vs plain | Acceptance |
|---|---|---|---|
| untrimmed (248k tokens) | 93.7 | 1.41x | 74.1 % |
| trimmed to 8k | 98.5 | 1.48x | 40.8 % |
| trimmed to 8k + context-adaptive rows (experimental) | 114.0 | 1.72x | 58.1 % |
| **trimmed to 32k** | **124.9** | **1.88x** | 68.5 % |

## How it works

1. **Speculative decoding.** Instead of asking the big model for one token at a time, a cheap *drafter* guesses the next few tokens and the big model checks all of them in a single pass. Every accepted guess is a token the big model did not have to generate one by one. The final text is whatever the big model's own greedy choice is at each position - wrong guesses are simply discarded.
2. **The drafter is the model's own MTP head.** Qwen3.5 ships with a small multi-token-prediction block that predicts the next tokens from the model's hidden state. llama.cpp supports it (`--spec-type draft-mtp`). No separate draft model is needed and the bf16 GGUF already contains it.
3. **The expensive part of drafting is the vocabulary.** Qwen3.5 has a ~248k-token output vocabulary and the 0.8B model ties its output layer to the embedding table, so every drafted token re-reads a ~0.5 GB (bf16) matrix. Plain MTP therefore gives only 1.15x here. **Trimming the draft head's output layer to the K most common tokens** (we use a frequency ranking, `results/2026-10-26-bf16-b4-token-rank.txt`) cuts that cost by about 4 ms per drafted token. The big model's own output layer is untouched and still scores the full vocabulary, so the target still verifies every token. Small K is best for English prose (8k); code needs identifier-like tokens, so K = 32k is better for code.
4. **An n-gram drafter in front for copy-heavy text.** When the output repeats text that is already in the prompt, a plain n-gram lookup (`ngram-simple`, 4-token key, up to 8 tokens proposed) predicts long runs correctly. llama.cpp lets you chain drafters (`--spec-type ngram-simple,draft-mtp`: first non-empty draft wins), so the n-gram drafter proposes long copies and MTP covers everything else.
5. **The target verifies every token.** See [Limitations](#limitations-and-the-losslessness-caveat) for what that does and does not guarantee.

## When to use which config

| Your output looks like | Use | Measured (RTX 3050 Laptop) | Notes |
|---|---|---|---|
| **Code** | MTP, draft head trimmed to **32k**, `--spec-draft-n-max 2` | 124.9 tok/s, **1.88x** | 8k trim is worse on code (1.48x) |
| **Copy-heavy text** (rewrite/edit a given text, summarise with quotes, repeat records, RAG-style answers) | `ngram-simple,draft-mtp`, 8k trim, `--spec-draft-n-max 8` | 201.1 tok/s, **2.97x** (re-run: 189 tok/s, 2.89x) | Needs patch 2 for the exact config (see below) |
| **Ordinary English prose** (essays, stories, explanations) | MTP only, draft head trimmed to **8k**, n-max 2 | 107-109 tok/s, 1.57-1.67x (plain 65.1-68.3) | **Do not add the n-gram drafter**: 89.1 tok/s, which is still 1.37x plain but only 0.83x of MTP alone (n-gram guesses are mostly wrong and the long verify batch is wasteful). The 32k trim also works on prose (104.2 tok/s, 1.60x) |
| **Non-English** (es, de, zh, hi tested) | Not worth it with the headline configs | untrimmed MTP 1.00x; 8k trim 1.01x; 32k trim 1.04x | English-frequency trimming hurts acceptance. An experimental context-adaptive vocabulary (patch 2) reached 86.6 tok/s = 1.26x (acceptance 33 %) |

Also measured (prose only): with temperature 0.7 / top-p 0.9 the 8k-trim MTP config ran at 104.8 tok/s vs 67.3 plain (1.56x). Sampling was not tested on code or copy-heavy prompts.

Things that did **not** help on prose (details in [`results/2026-10-27-bf16-E1-E5.md`](results/2026-10-27-bf16-E1-E5.md) and [`docs/`](docs/README.md)): history-adaptive draft length (0.86x of fixed n-max 2), a draft confidence gate (`--spec-draft-p-min 0.5`, 0.77-0.79x), n-max 3 (a tie), and a separate Q4 draft model (0.505x of plain). Tree/multi-candidate drafts are not possible in this llama.cpp version.

## Quick start

You need the patched llama.cpp (see [Building llama.cpp](#building-the-patched-llamacpp)) and the bf16 GGUF `Qwen_Qwen3.5-0.8B-bf16.gguf` from [bartowski/Qwen_Qwen3.5-0.8B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF) (about 1.5 GB). Put the GGUF in `models/` and run from the repo root so the paths below resolve.

**Code (1.88x)** - PowerShell:

```powershell
$env:LLAMA_MTP_TRIM_FILE = "results\2026-10-26-bf16-b4-token-rank.txt"
$env:LLAMA_MTP_TRIM_K    = "32768"
llama-server -m models\Qwen_Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa on -np 1 -kvu -c 4096 `
  --spec-type draft-mtp --spec-draft-n-max 2 -bs
```

**Copy-heavy (about 2.8-3.0x)** - PowerShell:

```powershell
$env:LLAMA_MTP_TRIM_FILE  = "results\2026-10-26-bf16-b4-token-rank.txt"
$env:LLAMA_MTP_TRIM_K     = "8192"
$env:LLAMA_MTP_ADAPT      = "1"     # patch 2: MTP draft length adapts per sequence (starts at 2)
$env:LLAMA_MTP_ADAPT_INIT = "2"
llama-server -m models\Qwen_Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa on -np 1 -kvu -c 4096 `
  --spec-type ngram-simple,draft-mtp --spec-draft-n-max 8 -bs `
  --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8
```

Linux/macOS shells: use `export LLAMA_MTP_TRIM_FILE=results/2026-10-26-bf16-b4-token-rank.txt` etc. and `./llama-server` with the same flags.

Flags in short: `-ngl 99 -fa on` full GPU offload with flash attention; `-np 1 -kvu` one slot, unified KV; `-bs` backend (GPU) sampling, worth about +4-5 % here; `LLAMA_MTP_TRIM_*` select the trimmed draft head; `LLAMA_MTP_ADAPT*` is the adaptive MTP draft length from patch 2 (part of the measured copy-heavy config; on plain prose it did not help).

Query it the way the benchmarks do (chat template, thinking off, greedy):

```powershell
$body = @{ prompt = "<|im_start|>user`nWrite a Python function that merges overlapping intervals.<|im_end|>`n<|im_start|>assistant`n<think>`n`n</think>`n`n"; n_predict = 256; temperature = 0; cache_prompt = $false } | ConvertTo-Json
(Invoke-RestMethod http://127.0.0.1:8080/completion -Method Post -Body $body -ContentType "application/json").timings
```

The `timings` block reports `predicted_per_second`, `draft_n` and `draft_n_accepted`.

## Reproduce the numbers

One entry point builds or finds the patched llama.cpp, finds or downloads the model, and runs plain vs the code config vs the copy-heavy config (and MTP-alone as a control) on the prompt sets in [`prompts/`](prompts/) with one warm-up pass and three timed passes, each config in a fresh server, then prints tok/s, acceptance and ratios against the plain run of the same set:

```powershell
.\scripts\reproduce.ps1 -Smoke        # 1 pass, 64 tokens: a couple of minutes, checks that everything works
.\scripts\reproduce.ps1               # full: 3 passes x 256 tokens, about 4 minutes on the RTX 3050 laptop
.\scripts\reproduce.ps1 -IncludeProse # adds the ordinary-prose sets to show where the n-gram config loses
```

```bash
scripts/reproduce.sh --smoke          # Linux equivalent (CUDA); see the note below
scripts/reproduce.sh
```

Useful options: `-LlamaDir` / `--llama-dir` (use an existing patched build), `-Model` / `--model`, `-Passes`, `-Build` (force a rebuild), `-RequireIdle` (abort if the GPU is busy; by default it only warns). Output tables go to `reproduce-out/` (git-ignored). The default 3-pass run on the author's PC is recorded in [`results/2026-10-27-reproduce-check.md`](results/2026-10-27-reproduce-check.md).

Please close other GPU applications before a run; on a laptop also keep it on AC power. A GPU shared with another workload skews the numbers (during one measurement session an external Hyper-V worker was using the GPU; those runs were discarded and redone).

`reproduce.sh` has the same logic but has **not been run** (at most syntax-checked); the PowerShell script was run end to end on Windows 11 (smoke and full run).

### The original experiment harness

The numbers in the reports come from `scripts/bf16_bench.py` (sessions B1-B4) and `scripts/e_bench.py` + `scripts/e_cfg2.py` (E-series); they are kept as-is (they expect the dev-build folders `llama-cpp/`, `llama-cpp-b4/`, `llama-cpp-e/`). `reproduce_bench.py` is a simplified, self-contained version of the same method (same prompts, same request shape, same flags).

## Building the patched llama.cpp

The trimmed draft head is not in upstream llama.cpp. It is two small patches against **llama.cpp release `b11884` (commit `63412d873`)**:

| Patch | What it does | Needed for |
|---|---|---|
| [`patches/b4-trim-draft-head.patch`](patches/b4-trim-draft-head.patch) | Copies the output-layer rows of the K most frequent tokens (rank file, env `LLAMA_MTP_TRIM_FILE` / `LLAMA_MTP_TRIM_K`) into a small `[n_embd, K]` weight used only by the MTP drafter, and maps drafted ids back to token ids. Qwen3.5 dense with an MTP block only (anything else falls back to the stock path with a warning); no-op if the env vars are unset | both headline configs |
| [`patches/e2-e4-adaptive-n-dynamic-vocab.patch`](patches/e2-e4-adaptive-n-dynamic-vocab.patch) | Adaptive draft length (`LLAMA_MTP_ADAPT*`) and the experimental context-adaptive draft vocabulary (`LLAMA_MTP_TRIM_FIXED`, `LLAMA_MTP_DYN`). Applies on top of patch 1; no-op unless its env vars are set | the exact copy-heavy config (`LLAMA_MTP_ADAPT=1`, `LLAMA_MTP_ADAPT_INIT=2`; not separately ablated on the copy-heavy set) |

```bash
git clone --branch b11884 --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git apply ../slmspeed/patches/b4-trim-draft-head.patch
git apply ../slmspeed/patches/e2-e4-adaptive-n-dynamic-vocab.patch   # apply in this order
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-server
```

`scripts/reproduce.ps1 -Build` / `scripts/reproduce.sh --build` do exactly this (clone, verify the commit, apply both patches, build into `work/llama-patched/bin`).

What the measurements were actually run on: the CUDA toolkit was not installed on the author's PC, so the patched binary was a **hybrid build**: the patch only touches host-side code (`src/`, `common/`), so llama.cpp b11884 was built with MSVC and `-DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON`, and the official b11884 CUDA 12.4 Windows release DLLs (`ggml-cuda.dll`, cuBLAS) were dropped next to it. That hybrid build was measured to be as fast as the stock release (79.14 vs 79.23 tok/s on the same untrimmed MTP config; see `results/2026-10-26-bf16-b4-trim-draft.md`). `reproduce.ps1` falls back to this recipe when `nvcc` is not found and does a normal native CUDA build when it is. A patched binary with none of the env vars set behaves like stock b11884.

Which binary produced which number: code configs ran on the build with patch 1 (`llama-cpp-b4`), copy-heavy configs on the build with patches 1+2 (`llama-cpp-e`, whose source is identical to b11884 + both patches in `patches/`; verified by diffing); patch 2 does nothing without its env vars.

### Making the trimmed head ranking

The ranking file shipped in `results/` (`2026-10-26-bf16-b4-token-rank.txt`, one token id per line, most frequent first, special ids first) is what all measurements used. It was produced by `scripts/b4_rank_tokens.py` (it expects `llama-cpp-b4/llama-tokenize.exe`): tokenize a held-out corpus (WikiText-2 test split saved as `data/wiki.test.raw`, plus earlier generated text and the repo's markdown) with `llama-tokenize`, rank by frequency, append never-seen ids in id order. No new GGUF is written: the patch builds the trimmed head from the full one when the model loads. The corpus is English prose, which is why English prose likes a small K and non-English text suffers. Regenerating the file will give a slightly different ranking (the corpus includes this repo's own markdown); keep the shipped file if you want to match the published numbers.

## Hardware portability

- **Ratios are the thing that should transfer; raw tok/s will not.** All absolute numbers here are for an RTX 3050 Laptop GPU (4 GB; the driver reports "RTX 3050 A Laptop GPU"), i5-13420H, 16 GB RAM, plugged in, Windows 11 Balanced power plan (no system settings changed), NVIDIA driver 592.82, llama.cpp b11884 with CUDA 12.4 DLLs.
- **Why the gain should exist elsewhere** (reasoning, *not measured* on other hardware): a 0.8B model decodes memory-bandwidth-bound and launch-overhead-bound on small GPUs, so verifying 2-3 tokens in one batch costs little more than one, and the draft head's re-read of the 248k-row output matrix is a large share of drafting cost. On a much faster GPU the plain model gets faster, the fixed per-step overheads stay, so ratios may shrink; on slower memory they may grow. We have no data for either.
- **Replicating on your hardware:** use `scripts/reproduce.ps1` (Windows) or `scripts/reproduce.sh` (Linux) on any NVIDIA CUDA GPU with enough free VRAM for the 1.5 GB model plus a 4096-token context (it fits a 4 GB card), compare *ratios* to the tables above, and include `-IncludeProse`. If your plain speed differs a lot, report plain and speculative numbers together.
- **Tested:** NVIDIA RTX 3050 Laptop on Windows 11 only. CPU: the adaptive-length/dynamic-vocabulary patch was smoke-tested for functionality on CPU (`-dev none`, `scripts/e_smoke.py`); no CPU speed-up was measured. Vulkan, Metal, ROCm/HIP, SYCL and other backends are **untested**; the patch only uses generic ggml tensor copies so it may work, but nothing is claimed.
- The scripts assume `nvidia-smi` for the idle check; without it the check is skipped.

## Limitations and the losslessness caveat

- **Target verification, not bit-identical output.** The full bf16 target model scores and verifies every token, so the output is always something the target would pick greedily *given the batch it was verified in*. But speculative output is **not guaranteed byte-identical to plain greedy decoding**, not even with the untrimmed stock MTP head: verifying several tokens in one batch changes floating-point reduction order, and near-ties can flip. Measured: for code, 15/20 (prompt x pass) outputs equal the plain greedy output (the same 15/20 as untrimmed MTP); copy-heavy 18/18 equal plain; ordinary prose only 6/18 (10/30 in an earlier session), where outputs diverge late in the text. The trim itself changes nothing relative to untrimmed MTP: trimmed and untrimmed MTP outputs were identical in every pair compared (30/30 on prose); equality in these reports is measured against the same build's MTP output. If you need byte-exact plain-greedy output, do not use speculative decoding.
- **Sampling.** Only greedy decoding was measured for the headline numbers. With temperature 0.7 the technique still preserves the target's distribution (rejection sampling), but outputs can't be compared byte-wise, and it was only timed on prose.
- **Workload-dependent.** The code trim ranks tokens from an English-prose corpus (K = 32k compensates). Other languages get little or no benefit; the n-gram config is slower than MTP alone on ordinary prose.
- **Small sample.** 4 code prompts, 6 copy-heavy prompts, 256 new tokens, one machine. Speeds drift about 5 % between sessions, so ratios compare to a same-session plain run.
- **Server cache caveat.** `--spec-type ngram-mod` keeps a pool across requests and appears to give 7-9x on a replayed identical request; that is a cache hit, not generation speed, and is not used for any headline number (cold runs: 189-195 tok/s, about the same as `ngram-simple`).
- **Scope.** Qwen3.5-0.8B dense bf16 only, single stream. Quantized variants (Q4-Q8) were explored in the earlier experiments and are not part of this release's claims; the patch rejects models with a scaled draft head and falls back to the untrimmed path.
- **Upstream.** This is a patch against one llama.cpp release. A similar idea (FR-Spec-style draft vocabulary trimming for native MTP) is tracked upstream in ggml-org/llama.cpp issue #25187; this repo's mechanism differs and needs no GGUF changes. Newer llama.cpp versions may include or conflict with it.

## Repository layout

```
README.md                 this file
LICENSE                   MIT (this repo's code)
patches/                  the two llama.cpp patches (b11884)
prompts/                  prompts.json (copy-heavy "copy" C1-C6 + prose "free" F1-F6), e-prompts.json (code K1-K4, non-English)
scripts/reproduce.ps1     one-command reproduction (Windows)      reproduce.sh (Linux, untested)
scripts/reproduce_bench.py  the benchmark used by both
scripts/b4_*.py, e_*.py   draft-head ranking, original B/E-series harness and reports
results/                  headline reports + raw data (B0-B4, E1-E5), token-rank file, reproduce check
docs/                     index of the other experiments (docs/README.md)
docs/experiments/         write-ups of everything else (quant sweeps, skeleton-of-thought, parallel sub-tasks, ...)
```

The wider experiment log - quantization sweeps, speculative decoding with other drafters, skeleton-of-thought, parallel sub-tasks, free-text and non-English studies - is indexed in [`docs/README.md`](docs/README.md). It is archived material, not part of the release claims.

## License and credits

- This repository's code and docs: **MIT** ([LICENSE](LICENSE)).
- The patches modify [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, Copyright (c) 2023-2026 The ggml authors) and are distributed under the same MIT license; speculative decoding, the MTP drafter and the n-gram drafters are llama.cpp's work.
- Model: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) by the Qwen team, **Apache-2.0** per its model card (`license: apache-2.0`). The model is not redistributed here; GGUF conversion by [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF). The file measured was `Qwen_Qwen3.5-0.8B-bf16.gguf` (1.45 GiB, SHA-256 `AD1549EEDC613064971DCBBBFAB6C9B7990984D1C9AB38F792C6F2EC1207BBC2`); upstream files can be re-uploaded, so a fresh download may differ.
- The draft-vocabulary trimming idea follows FR-Spec (frequency-ranked speculative drafting vocabulary) and the discussion in llama.cpp issue #25187.
