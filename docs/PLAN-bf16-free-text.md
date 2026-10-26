# PLAN - bf16 (non-quantized) free-text single-stream speed-up, Qwen3.5-0.8B

Status as of 2026-10-26 (B4 added). Branch `bf16-free-text` (from main; `archive/0.8b-experiments` untouched). Goal: speed up ONE continuous free-text reply, single stream, bf16 target, no batching/parallel/sub-task methods. Reporting: speed-up ratios vs a same-session plain bf16 control, warm-up discarded, 5 runs mean +/- SD, control re-run at the end of every session, greedy, 256+ generated tokens, loops flagged. Background and estimates: the research note summarised in `results/2026-10-26-bf16-b0-b3.md`; estimates there are **[est]**, measurements **[measured]**. Harness: `scripts/bf16_bench.py`, report: `scripts/bf16_report.py`.

The phase names B0-B7 follow the research list (sections 3-4 of the research note); the task items map as: the run's items "B0/B1" = B0+B1, "B2" (backend sampling) = a B1 addition, "B3" (native MTP) = research-list B2, and the Q4 draft for the bf16 target = research-list B3.

| Phase | What | Status |
|---|---|---|
| B0 | Plain bf16 baseline, same-session control, bench vs server reconciliation | **done** [measured]: server 68-69.5 tok/s, bench 71.7; the old 51 tok/s did not reproduce |
| B1 | Overhead flags/env: CUDA graphs, fusion, graph reuse, GRAPH_OPT, fa, ctx, f16, length, `-np/-kvu`; backend sampling (`-bs`) | **done** [measured]: all defaults already optimal; `-bs` +4.0 % (70.94 vs 68.2) |
| B2 | Native MTP (`--spec-type draft-mtp`), n-max 1-3, p-min sweep, bf16 target | **done** [measured]: n-max 1 = 74.1 tok/s, 1.088x (accept 63 %); n-max 2 1.065x; n-max 3 0.951x; every p-min gate 0.84-0.94x |
| B3 | Q4_K_M same-model draft (`-md`, draft-simple) for the bf16 target; Q4 MTP file as a different target | **partly done** [measured]: one Q4-draft setting (n-max 2, p-min 0.6) = 0.505x; Q4 MTP file as target: MTP 0.99x / 0.96x / 0.85x of its own plain control (n-max 1/2/3). Other draft settings not tried |
| B4 | Output-layer trimming: lossless draft-only trimmed MTP head (FR-Spec, llama.cpp issue #25187) vs lossy 32k target head | **draft-only trim done** [measured]: patch `patches/b4-trim-draft-head.patch` (own hybrid build: MSVC + release CUDA DLL, no CUDA toolkit needed). MTP + 8k-token draft head + `-bs`: n-max 2 = 110.1 tok/s = **1.595x** (n-max 1 1.413x) vs previous best 1.147x; output identical to untrimmed MTP 30/30. Lossy target-head trim not done; no-rebuild GGUF route infeasible (head shape fixed to n_vocab). See `results/2026-10-26-bf16-b4-trim-draft.md` |
| B5 | Combine the winners (`-bs` + best MTP setting + graph defaults) and re-measure in one session | **first combination done** [measured]: bf16 + MTP n-max 1 + `-bs` = 79.18 +/- 0.07 tok/s = 1.147x vs plain bf16 (best so far); further combos not tried |; B4 supersedes the best: **bf16 + trimmed-8k-head MTP n-max 2 + `-bs` = 110.1 tok/s = 1.595x** (own build, env `LLAMA_MTP_TRIM_FILE`/`_K`)
| B6 | Overhead-floor probe: Luce megakernel as external reference, Nsight Systems/Compute of one decode token | **not started** |
| B7 | Code: fuse more GDN ops / custom head kernel in llama.cpp | **not started**; only worth it if B6 shows large overhead |

Open questions: why the 09:50 llama-bench run measured 51 tok/s (not reproducible; cause unknown); machine-state drift of 3-5 % within a day (use same-block controls).
