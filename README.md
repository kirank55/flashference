# slmspeed

Experiments to maximize token generation speed for a small language model (Qwen3.5-0.8B) on a laptop.

**Hardware:** Windows 11, Intel i5-13420H (4P+4E cores), 16 GB RAM, RTX 3050 A Laptop GPU (4 GB VRAM).
**Runtime:** [llama.cpp](https://github.com/ggml-org/llama.cpp) build b11884, CUDA 12.4.

## Setup
```powershell
.\scripts\setup.ps1             # llama.cpp binaries -> llama-cpp/
.\scripts\download-models.ps1   # GGUF models -> models/
.\scripts\bench-quants.ps1      # baseline benchmark
```
Binaries and models are git-ignored.

## Docs
- [docs/FINDINGS.md](docs/FINDINGS.md) - everything learned so far (measured vs estimated), research sources, caveats, open questions.
- [docs/PLAN-single-reply.md](docs/PLAN-single-reply.md) - prioritized experiment plan for speeding up one continuous reply.

## Results
See `results/`. Baseline (2026-10-26, full GPU offload, flash attention on):

| Quant | Decode tok/s | Prompt tok/s |
|---|---|---|
| Q4_K_M | 114.5 | 6617 |
| Q5_K_M | 100.0 | 6849 |
| Q6_K | 94.5 | 6661 |
| Q8_0 | 75.5 | 7129 |
| BF16 | 51.5 | 4937 |

Run-to-run noise was about ±13 tok/s, so close comparisons need steadier power and thermal conditions.

### Parallel sub-tasks (2026-10-26, Q4_K_M, llama-server continuous batching)
Details: `results/2026-10-26-parallel-subtasks.md`; scripts: `scripts/parallel_subtasks.py`, `scripts/make_report.py`. Means of 5 timed runs after warm-up, laptop on AC, Balanced plan.

| Setup | Aggregate tok/s | Per-stream tok/s | Wall time |
|---|---|---|---|
| Baseline A: one ~400-word reply, single stream | 139 | 142 | 3.42 s (476 tok) |
| Exp 1, 16 tasks, sequential (np=1) | 127 | 140 | 10.3 s |
| Exp 1, 16 tasks, np=2 / 4 / 8 | 196 / 237 / 250 | 110 / 71 / 45 | 6.6 / 5.6 / 5.3 s |
| Exp 1, 16 tasks, np=16 | 288 | 27 | 4.5 s |
| Exp 1, 48 tasks, np=16 (saturation) | 295 (max run 309) | 25 | 13.3 s |
| Exp 2 skeleton-of-thought, best (np=4, long sections, 6 points) | 170 effective | 75 | 3.6 s (533 words) |

Parallel slots gave about 2.3x aggregate throughput at best; mean never exceeded 300 tok/s (only single runs did) and np=8/16/32 plateau around 270-300. Skeleton-of-thought was only about 1.2x faster per word than a sequential reply and read noticeably worse (repetitive, no transitions). Peak VRAM 0.7 GiB (np=1) to 1.4 GiB (np=16), 2.1 GiB at np=32.
## Planned experiments
Batch size, CPU threads, KV-cache quantization, speculative decoding (`--spec-type draft-mtp`), CPU-only comparison, power mode.
See [docs/PLAN-single-reply.md](docs/PLAN-single-reply.md) for the detailed, prioritized plan.
