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

## Planned experiments
Batch size, CPU threads, KV-cache quantization, speculative decoding (`--spec-type draft-mtp`), CPU-only comparison, power mode.
