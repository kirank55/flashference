# reproduce.ps1 check run (2026-10-27, RTX 3050 Laptop 4 GB, Windows 11)

Command: `.\scripts\reproduce.ps1` (3 timed passes x 256 tokens, 1 warm-up pass, fresh server per config, GPU idle before start, patched llama.cpp b11884 = `llama-cpp-e` dev build of patches/*.patch). Run 17:01-17:05 IST, nothing else on the GPU. Raw per-pass numbers: `reproduce-out/` (git-ignored).

| config | set | what | tok/s (mean +/- SD) | vs plain (same set) | acceptance | output == plain |
|---|---|---|---|---|---|---|
| code-plain | code | plain bf16 | 67.66 +/- 0.06 | 1.00x | - | 12/12 |
| code-mtp-trim32k | code | HEADLINE code: MTP, 32k draft head, 2 drafts, -bs | 124.28 +/- 0.35 | 1.84x | 68.5% | 9/12 |
| copy-plain | copy | plain bf16 | 65.45 +/- 2.07 | 1.00x | - | 18/18 |
| copy-mtp-trim8k | copy | MTP alone, 8k draft head, 2 drafts, -bs (control) | 146.74 +/- 0.20 | 2.24x | 84.4% | 18/18 |
| copy-ngram-mtp | copy | HEADLINE copy-heavy: ngram-simple,draft-mtp | 189.24 +/- 0.48 | 2.89x | 58.5% | 18/18 |

Comparison with the recorded E-series numbers (`2026-10-27-bf16-E1-E5.md`): code 124.28 vs 124.89 tok/s (the plain control this time was 67.66 vs 66.48, hence 1.84x vs 1.88x); MTP alone on copy-heavy 146.74 vs 146.74; ngram-simple+MTP on copy-heavy 189.24 vs 201.10 tok/s (acceptance identical, 58.5 %; plain control this run 65.45 +/- 2.07, noisy, vs 67.61 recorded: 2.89x vs this run's plain, 2.80x vs the recorded plain). So the copy-heavy headline reproduces at 2.8-3.0x; the exact figure depends on machine state.

Smoke mode (`-Smoke`: 1 pass, 64 tokens, code + copy sets): code 119.16 tok/s = 1.75x, copy MTP-only 150.97 = 2.21x, ngram-simple+MTP 220.94 = 3.24x (short outputs, indicative only).
