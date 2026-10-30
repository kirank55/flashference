# NOTEBOOK

Dated entries, newest at the bottom. Template: plan.md Appendix B.

## Prediction log

<!-- P-001 | date | phase | prediction + why | got | error | learned -->

---

## 2026-10-30: Phase 0 workbench

**Machine (checked 2026-10-30):** RTX 3050 A Laptop 4 GB, compute capability 8.9, Windows driver 592.82, WSL2 Ubuntu 24.04, Python 3.12.3, uv 0.12.3. Windows RAM is 15.6 GB and `.wslconfig` currently sets `memory=8GB`. torch 2.15.0+cu132 sees the GPU.

**Model sha256:** ad1549ee…07bbc2, the same as v1
**llama.cpp:** b11884, `git rev-parse --short HEAD` = 63412d873 (branch `stock`)
**llama-bench tg128 (bf16, -ngl 99 -fa on):** __ tok/s (v1 measured about 71 on Windows)
**What I now understand:**
**Open questions:**
