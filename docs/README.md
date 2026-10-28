# Other experiments (archive)

The [top-level README](../README.md) covers only the two results flashference is released for: fast **code** generation and fast **copy-heavy** generation for Qwen3.5-0.8B bf16 with lossless speculative decoding. Everything else that was tried on the way is archived here. These are working notes written during the experiments, not polished docs: they were written before the release, some used different models/quantizations (mostly Qwen3.5-0.8B Q4_K_M) and earlier builds, and some conclusions are superseded by the bf16 E-series (`../results/2026-10-27-bf16-E1-E5.md`). Treat numbers in them as belonging to their own session; nothing here is part of the release claims.

| Topic | Write-up | Scripts | Raw data (in `../results/`) |
|---|---|---|---|
| **Index of conclusions, all experiments, hardware notes** | [FINDINGS.md](experiments/FINDINGS.md) | - | - |
| Quantization sweep (decode tok/s per quant, llama-bench) | [2026-10-26-baseline-quants.md](experiments/2026-10-26-baseline-quants.md) (raw llama-bench table, not UTF-8) | `bench-quants.ps1`, `s1_quants.py`, `s1_report.py` | `2026-10-26-s1-quants-0.8b.raw.json` |
| Summary of the 0.8B scaling experiments S1-S3 (quants, speculative decoding, skeleton-of-thought) | [2026-10-26-scaling-0.8b.md](experiments/2026-10-26-scaling-0.8b.md) | - | - |
| Speculative decoding on the quantized 0.8B via llama-server | [2026-10-26-s2-spec-0.8b.md](experiments/2026-10-26-s2-spec-0.8b.md) | `s2_spec.py`, `s2_report.py` | `2026-10-26-s2-spec-0.8b.raw.json.gz` |
| Skeleton-of-thought for one reply | [2026-10-26-s3-sot-0.8b.md](experiments/2026-10-26-s3-sot-0.8b.md) | `s3_sot.py`, `s3_report.py`, `prompt_sets.py` (regenerates `prompts/prompts.json`) | `2026-10-26-s3-sot-0.8b.raw.json`, `2026-10-26-s3-samples.json` |
| Parallel sub-tasks (split one answer into concurrent requests) | [2026-10-26-parallel-subtasks.md](experiments/2026-10-26-parallel-subtasks.md) | `parallel_subtasks.py`, `make_report.py` | `2026-10-26-parallel-subtasks.raw.json`, `2026-10-26-parallel-subtasks-saturation.raw.json` |
| Plan for general single-reply speed-ups | [PLAN-single-reply.md](experiments/PLAN-single-reply.md) | - | - |
| Plan for the bf16 free-text work (led to B0-B4, E1-E5) | [PLAN-bf16-free-text.md](experiments/PLAN-bf16-free-text.md) | - | - |

## The bf16 series (kept in `results/`, because the release rests on it)

| Session | What | Report | Scripts |
|---|---|---|---|
| B0-B3 | bf16 baseline, per-request overhead, backend sampling (`-bs`), native MTP | [`../results/2026-10-26-bf16-b0-b3.md`](../results/2026-10-26-bf16-b0-b3.md) | `bf16_bench.py`, `bf16_report.py`, `b1_rerun.ps1`, `b2_run.ps1`, `b2_rerun.ps1`, `b3_run.ps1`, `b3_combo.ps1` |
| B4 | trimmed MTP draft head (patch 1) | [`../results/2026-10-26-bf16-b4-trim-draft.md`](../results/2026-10-26-bf16-b4-trim-draft.md) | `b4_build.cmd`, `b4_rebuild.cmd`, `b4_rank_tokens.py`, `b4_run1.ps1`-`b4_run3.ps1`, `b4_report.py` |
| E1-E5 | robustness, adaptive draft length, n-gram drafter, dynamic draft vocabulary, lossless-head study; code and copy-heavy headline numbers | [`../results/2026-10-27-bf16-E1-E5.md`](../results/2026-10-27-bf16-E1-E5.md) | `e_bench.py`, `e_cfg2.py`, `e_patch_apply.py` (how patch 2 was first applied; superseded by `patches/e2-e4-adaptive-n-dynamic-vocab.patch`), `e_report.py`, `e_smoke.py`, `e1_run*.ps1`, `e234_*.ps1`, `e5_collect.py`, `e5_analysis.py`, `gguf_read.py`, `gguf_tensors.py` |

Older scripts and the experiment harnesses expect the original dev layout (`llama-cpp/`, `llama-cpp-b4/`, `llama-cpp-e/`, `llama-src/`, `models/`, `data/` next to `scripts/`; these folders are git-ignored) and Windows paths relative to the repo root. `scripts/setup.ps1` and `scripts/download-models.ps1` set up the early (S1-S3) environment. For a clean one-command run of the release results use `scripts/reproduce.ps1` instead.
