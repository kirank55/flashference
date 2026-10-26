# Parallel sub-tasks experiment (2026-10-26)

Goal: can parallel sub-tasks (llama-server slots / continuous batching) push aggregate decode throughput on a tiny model above 300 tok/s?

**Setup.** Qwen3.5-0.8B Q4_K_M, llama.cpp b11884 CUDA (llama-server), `-ngl 99 -fa on -c 2048*np --no-webui`, default continuous batching. Raw completions via `/completion` with Qwen chat format and an empty think block (thinking off), temperature 0, seed 1, `cache_prompt=false`. Python 3.13 stdlib client (threads); wall time measured client-side, per-stream tok/s from server `timings.predicted_per_second`. For each server config: one discarded warm-up pass of every workload, then 5 timed runs (3 for the saturation check). Servers were started fresh per `-np`; the np=1 config was repeated at the end as a drift check (it matched the first pass within about 1-2%).
**Machine state.** Laptop on AC (`Win32_Battery.BatteryStatus` = 2, 98% charge), power plan Balanced (not changed). GPU idle before every start (0 MiB used, 0% util, 47-60 C). GPU ran at about 2487 MHz SM clock throughout and 31-41 W mean power while busy (sampled); max temp 70 C. Note the laptop GPU has a ~37 W cap per `nvidia-smi`, so it is likely power/memory-bandwidth limited, not clock-throttled by temperature.
**Scripts.** `scripts/parallel_subtasks.py` (runner), `scripts/make_report.py` (tables), raw data in `results/2026-10-26-parallel-subtasks.raw.json` and `...-saturation.raw.json`.

Workloads:
- **Baseline A**: "Write a ~400-word explanation of how electric cars work." `n_predict` 700, single stream (finishes naturally at 476 tokens / 404 words).
- **Exp 1**: 8 or 16 independent short tasks (summarize x5, translate+explain x4, classify+justify x4, explain x3), `n_predict` 100; they average ~68-82 tokens because some end earlier. np=1 runs them sequentially; np>1 submits all at once (excess queues on the server). Same prompts and temperature 0, so total tokens are essentially identical across configs.
- **Exp 2** (skeleton-of-thought): step 1 outline of 6 points (<=80 tokens), step 2 expand each point in its own request (np=1: sequentially, np>1: concurrent), step 3 concatenate. Two section lengths ("short" ~4-5 sentences, "long" ~5-6 sentences) because the model does not follow word counts closely; Baseline A's 404 words falls between them (305-375 and 490-551 words). A words-normalised column is given for fair comparison.
Format: mean ┬▒ sd (minΓÇômax) over 5 timed runs.

### Baseline A: one ~400-word reply, single stream (n_predict 700)

| server | tokens | words | wall s | tok/s (tokens/wall) | decode tok/s (server timings) |
|---|---|---|---|---|---|
| np=1 | 476 | 404 | 3.42 ┬▒ 0.05 (3.37ΓÇô3.48) | 139.1 ┬▒ 2.1 (136.8ΓÇô141.2) | 141.7 ┬▒ 1.9 (139.3ΓÇô143.2) |
| np=2 | 476 | 404 | 3.45 ┬▒ 0.04 (3.41ΓÇô3.50) | 138.0 ┬▒ 1.8 (135.9ΓÇô139.7) | 141.0 ┬▒ 1.7 (139.2ΓÇô142.8) |
| np=4 | 476 | 404 | 3.46 ┬▒ 0.04 (3.41ΓÇô3.52) | 137.7 ┬▒ 1.6 (135.3ΓÇô139.4) | 141.6 ┬▒ 1.3 (140.1ΓÇô143.0) |
| np=8 | 476 | 404 | 3.52 ┬▒ 0.05 (3.45ΓÇô3.57) | 135.1 ┬▒ 1.9 (133.4ΓÇô138.1) | 140.7 ┬▒ 1.7 (138.6ΓÇô142.7) |
| np=16 | 476 | 404 | 3.48 ┬▒ 0.04 (3.44ΓÇô3.54) | 136.9 ┬▒ 1.7 (134.6ΓÇô138.5) | 141.9 ┬▒ 1.5 (139.8ΓÇô143.1) |
| np=1 (re-check at end) | 476 | 404 | 3.46 ┬▒ 0.03 (3.43ΓÇô3.51) | 137.5 ┬▒ 1.3 (135.7ΓÇô138.8) | 140.0 ┬▒ 1.2 (138.3ΓÇô141.1) |

### Experiment 1: 8 independent sub-tasks (n_predict 100 each); np=1 is sequential, np>1 all 8 requests submitted concurrently

| server | total tokens | job wall s | aggregate tok/s | per-stream decode tok/s | mean / max request latency s | speed-up vs np=1 |
|---|---|---|---|---|---|---|
| np=1 | 544 | 4.37 ┬▒ 0.12 (4.26ΓÇô4.56) | 124.5 ┬▒ 3.4 (119.4ΓÇô127.7) | 140.7 ┬▒ 2.6 (136.7ΓÇô143.0) | 0.55 / 0.79 | 1.00x |
| np=2 | 543 | 2.96 ┬▒ 0.07 (2.86ΓÇô3.03) | 183.9 ┬▒ 4.0 (179.4ΓÇô189.7) | 110.8 ┬▒ 2.8 (108.0ΓÇô114.6) | 1.60 / 2.95 | 1.48x |
| np=4 | 545 | 2.25 ┬▒ 0.25 (2.09ΓÇô2.69) | 244.7 ┬▒ 23.6 (203.2ΓÇô260.4) | 75.8 ┬▒ 6.0 (65.5ΓÇô79.5) | 1.48 / 2.25 | 1.94x |
| np=8 | 544 | 2.56 ┬▒ 0.21 (2.35ΓÇô2.91) | 213.4 ┬▒ 16.8 (187.3ΓÇô231.7) | 48.9 ┬▒ 2.0 (46.1ΓÇô51.3) | 1.72 / 2.56 | 1.70x |
| np=16 | 543 | 3.22 ┬▒ 0.40 (2.97ΓÇô3.92) | 170.7 ┬▒ 18.7 (138.9ΓÇô183.4) | 29.5 ┬▒ 4.5 (22.6ΓÇô33.7) | 2.54 / 3.22 | 1.36x |
| np=1 (re-check at end) | 544 | 4.42 ┬▒ 0.02 (4.39ΓÇô4.45) | 123.0 ┬▒ 0.7 (122.2ΓÇô123.8) | 138.9 ┬▒ 1.0 (137.7ΓÇô140.0) | 0.55 / 0.79 | 0.99x |

### Experiment 1: 16 independent sub-tasks (n_predict 100 each); np=1 is sequential, np>1 all 16 requests submitted concurrently

| server | total tokens | job wall s | aggregate tok/s | per-stream decode tok/s | mean / max request latency s | speed-up vs np=1 |
|---|---|---|---|---|---|---|
| np=1 | 1307 | 10.30 ┬▒ 0.23 (9.99ΓÇô10.58) | 126.9 ┬▒ 2.8 (123.5ΓÇô130.8) | 140.3 ┬▒ 2.2 (137.7ΓÇô143.1) | 0.64 / 0.80 | 1.00x |
| np=2 | 1294 | 6.60 ┬▒ 0.15 (6.37ΓÇô6.72) | 196.1 ┬▒ 4.3 (190.9ΓÇô200.7) | 110.1 ┬▒ 2.1 (107.5ΓÇô112.6) | 3.37 / 6.60 | 1.56x |
| np=4 | 1311 | 5.55 ┬▒ 0.28 (5.07ΓÇô5.74) | 237.0 ┬▒ 12.7 (225.2ΓÇô258.8) | 70.9 ┬▒ 2.2 (69.5ΓÇô74.7) | 3.00 / 5.54 | 1.86x |
| np=8 | 1305 | 5.27 ┬▒ 0.53 (4.86ΓÇô6.16) | 249.9 ┬▒ 24.1 (209.7ΓÇô270.0) | 45.2 ┬▒ 2.6 (41.4ΓÇô47.3) | 3.00 / 5.26 | 1.96x |
| np=16 | 1288 | 4.49 ┬▒ 0.26 (4.17ΓÇô4.81) | 287.6 ┬▒ 15.8 (268.3ΓÇô308.4) | 27.2 ┬▒ 1.1 (26.1ΓÇô28.6) | 3.59 / 4.49 | 2.29x |
| np=1 (re-check at end) | 1307 | 10.40 ┬▒ 0.10 (10.22ΓÇô10.47) | 125.7 ┬▒ 1.2 (124.8ΓÇô127.8) | 139.6 ┬▒ 1.0 (138.6ΓÇô141.2) | 0.65 / 0.80 | 0.99x |

Baseline A (np=1) = 0.85 s per 100 words.

### Experiment 2 (short sections): skeleton-of-thought, 6-point outline then parallel expansion (np=1 expands sequentially)

| server | points | outline s | expand s | total wall s | reply tokens | reply words | effective tok/s (reply tokens / total wall) | incl. outline tokens | per-stream decode tok/s | s per 100 reply words | words-normalised speed-up vs Baseline A (np=1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| np=1 | 6.0 | 0.55 | 2.78 | 3.33 ┬▒ 0.07 (3.24ΓÇô3.42) | 351 | 305 | 105.4 ┬▒ 2.1 (102.5ΓÇô108.2) | 126.7 | 141.9 | 1.09 | 0.78x |
| np=2 | 6.0 | 0.59 | 2.21 | 2.79 ┬▒ 0.08 (2.74ΓÇô2.93) | 428 | 375 | 153.4 ┬▒ 3.0 (150.6ΓÇô157.2) | 178.8 | 112.3 | 0.74 | 1.14x |
| np=4 | 6.0 | 0.61 | 1.86 | 2.47 ┬▒ 0.25 (2.21ΓÇô2.88) | 409 | 357 | 166.5 ┬▒ 11.1 (147.7ΓÇô176.4) | 195.4 | 79.7 | 0.69 | 1.23x |
| np=8 | 6.0 | 0.66 | 2.32 | 2.97 ┬▒ 0.30 (2.71ΓÇô3.48) | 376 | 330 | 127.4 ┬▒ 13.5 (114.6ΓÇô144.8) | 151.5 | 35.0 | 0.90 | 0.94x |
| np=16 | 6.0 | 0.73 | 2.69 | 3.41 ┬▒ 0.66 (2.61ΓÇô4.19) | 383 | 334 | 115.3 ┬▒ 20.2 (89.9ΓÇô136.2) | 136.7 | 31.6 | 1.02 | 0.83x |
| np=1 (re-check at end) | 6.0 | 0.55 | 2.82 | 3.37 ┬▒ 0.03 (3.32ΓÇô3.39) | 351 | 305 | 104.3 ┬▒ 0.9 (103.4ΓÇô105.8) | 125.4 | 139.6 | 1.10 | 0.77x |

### Experiment 2 (long sections): skeleton-of-thought, 6-point outline then parallel expansion (np=1 expands sequentially)

| server | points | outline s | expand s | total wall s | reply tokens | reply words | effective tok/s (reply tokens / total wall) | incl. outline tokens | per-stream decode tok/s | s per 100 reply words | words-normalised speed-up vs Baseline A (np=1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| np=1 | 6.0 | 0.55 | 4.52 | 5.08 ┬▒ 0.07 (5.00ΓÇô5.17) | 589 | 522 | 116.0 ┬▒ 1.7 (114.0ΓÇô117.7) | 130.0 | 140.9 | 0.97 | 0.87x |
| np=2 | 6.0 | 0.58 | 2.88 | 3.46 ┬▒ 0.20 (3.11ΓÇô3.60) | 552 | 490 | 159.6 ┬▒ 4.0 (154.5ΓÇô164.1) | 180.2 | 111.9 | 0.71 | 1.20x |
| np=4 | 6.0 | 0.61 | 3.01 | 3.62 ┬▒ 0.51 (3.05ΓÇô4.11) | 605 | 533 | 170.1 ┬▒ 26.5 (146.0ΓÇô198.9) | 190.1 | 74.8 | 0.68 | 1.25x |
| np=8 | 6.0 | 0.64 | 3.79 | 4.43 ┬▒ 0.40 (4.13ΓÇô5.11) | 604 | 534 | 136.9 ┬▒ 10.1 (122.4ΓÇô149.9) | 153.0 | 33.0 | 0.83 | 1.02x |
| np=16 | 6.0 | 0.63 | 4.40 | 5.03 ┬▒ 0.85 (4.20ΓÇô6.13) | 625 | 551 | 127.3 ┬▒ 21.8 (99.2ΓÇô147.1) | 141.7 | 29.9 | 0.91 | 0.93x |
| np=1 (re-check at end) | 6.0 | 0.57 | 4.58 | 5.16 ┬▒ 0.03 (5.13ΓÇô5.20) | 589 | 522 | 114.3 ┬▒ 0.6 (113.2ΓÇô114.7) | 128.0 | 139.6 | 0.99 | 0.86x |

### GPU during each server's whole run (sampled every ~0.4 s; includes warm-up)

| server | VRAM after warm-up MiB (whole GPU) | peak VRAM MiB | max temp C | mean power W (busy) | mean SM clock MHz (busy) | GPU idle before start (mem MiB / util % / temp C) |
|---|---|---|---|---|---|---|
| np=1 | 703 | 703 | 69 | 39.9 | 2488 | 0 / 0 / 47 |
| np=2 | 743 | 749 | 70 | 41.1 | 2487 | 0 / 0 / 60 |
| np=4 | 841 | 843 | 70 | 39.7 | 2487 | 0 / 0 / 60 |
| np=8 | 1027 | 1029 | 70 | 37.0 | 2487 | 0 / 0 / 59 |
| np=16 | 1385 | 1387 | 68 | 35.8 | 2487 | 0 / 0 / 59 |
| np=1 (re-check at end) | 703 | 703 | 69 | 40.4 | 2488 | 0 / 0 / 59 |

Server command (np=N): `-m models\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf -ngl 99 -fa on -np 1 -c 2048 --no-webui --host 127.0.0.1 --port 8089`  (-c = 2048 x np)

### Saturation check: 48 independent sub-tasks (cycled prompts, n_predict 100) submitted at once, 3 timed runs

| server | total tokens | job wall s | aggregate tok/s | per-stream decode tok/s | peak VRAM MiB | mean power W (busy) | max temp C |
|---|---|---|---|---|---|---|---|
| np=8 | 3898 | 14.23 ┬▒ 0.30 (14.03ΓÇô14.57) | 274.1 ┬▒ 6.3 (267.1ΓÇô279.2) | 42.4 ┬▒ 0.8 (41.8ΓÇô43.3) | 1029 | 36.9 | 67 |
| np=16 | 3912 | 13.28 ┬▒ 0.57 (12.62ΓÇô13.62) | 295.0 ┬▒ 12.1 (286.7ΓÇô308.9) | 24.6 ┬▒ 0.8 (23.9ΓÇô25.4) | 1387 | 31.0 | 66 |
| np=32 | 3906 | 13.90 ┬▒ 0.60 (13.27ΓÇô14.46) | 281.5 ┬▒ 12.7 (270.2ΓÇô295.3) | 14.5 ┬▒ 0.2 (14.2ΓÇô14.6) | 2115 | 32.0 | 67 |

## Quality notes (Exp 2, honest)
Qwen3.5-0.8B outline: "1. Electric motors convert electrical energy into motion. 2. Batteries store chemical energy for the vehicle. 3. The battery pack powers the electric motor. 4. The motor drives the wheels... 5. The battery is charged by sunlight or electricity. 6. The car uses the stored energy to travel." The points overlap heavily, so the stitched reply is repetitive (the battery -> motor -> wheels chain is restated in nearly every section), has no transitions or conclusion (each section is independent), the first section sometimes writes an intro-like paragraph despite instructions, and occasionally drifts (the 'long' variant produced a section on "solar charging" with DC charge controllers; one section said "the wheels drive the motor"). It reads as six loosely related paragraphs rather than one flowing explanation. The sequential Baseline A reply is more coherent and covers more distinct ground (induction, lithium-ion, efficiency, challenges), though it too contains inaccuracies (e.g. confusing friction/traction and kinetic energy statements). Outcome: skeleton-of-thought gives a modest speed-up but a clearly worse reply with this model; a larger model or a better outline prompt would be needed.

## Findings (measured)
- Single stream: ~139-142 tok/s decode (Baseline A), higher than the earlier llama-bench figure of 114 tok/s (the earlier note mentions noisy power conditions); `-np` up to 16 does not slow a single stream.
- Parallel slots help but saturate early: best sustained aggregate was about 280-290 tok/s (np=16, 16-48 tasks), i.e. about 2.0-2.3x the sequential 125-127 tok/s. Per-stream speed drops to ~27 tok/s at np=16.
- The 300 tok/s target was **not reached on average**. Only individual runs exceeded it (308.4 tok/s in the 16-task np=16 set, 308.9 tok/s in the 48-task np=16 saturation set); means were 287.6 +/- 15.8 and 295.0 respectively (see tables).
- np=8 and np=32 gave no gain over np=16; throughput plateaus at roughly 270-300 tok/s, so the GPU/server is saturated, probably by memory-bandwidth/power limits plus the model's hybrid recurrent layers and per-step sampling overhead (the last two are hypotheses, not measured).
- Small job sets suffer ramp-up and tail effects: with only 8 tasks the best was np=4 (244.7 tok/s); np=16 with 8 tasks leaves half the slots idle.
- Skeleton-of-thought: best was np=2-4 (~153-170 tok/s effective, 1.14-1.25x faster per word than Baseline A; sequential skeleton at np=1 is 0.78-0.87x, i.e. slower than just writing the reply, and np=8/16 are no better than Baseline A because the slowest slot and shared-batch slowdown dominate); it never approaches 300 tok/s because only 6 sections and an unavoidable sequential outline step limit the parallel part, and the longest section is the critical path.


