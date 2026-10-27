# FINDINGS - Qwen3.5-0.8B decode speed on an RTX 3050 A Laptop (4 GB)

Status as of 2026-10-26, repo commit f7ace2b. Everything is labelled **[measured]** (we ran it on this machine), **[reported]** (someone else's published number, not reproduced here) or **[estimate]** / **[hypothesis]** (reasoning, not tested). Raw data and scripts are in `results/` and `scripts/`. The follow-up experiment plan is in [PLAN-single-reply.md](PLAN-single-reply.md).

## 1. Hardware and software

| Item | Value |
|---|---|
| OS | Windows 11 |
| CPU | Intel i5-13420H (4 P + 4 E cores) |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3050 A Laptop, 4 GB VRAM. Laptop power cap about 37 W per `nvidia-smi` (see section 4) |
| Runtime | llama.cpp release b11884 (`63412d873`), CUDA 12.4 build, prebuilt binaries in `llama-cpp/` |
| Model | Qwen3.5-0.8B (hybrid: Gated DeltaNet linear-attention layers + a few full-attention layers; ~773 M params as reported by llama-bench), GGUFs from bartowski (`Qwen_Qwen3.5-0.8B-*.gguf`) |
| Common flags | `-ngl 99` (full offload), flash attention on (`-fa 1` / `-fa on`) |

Not recorded in the repo: the exact GPU memory bandwidth and the Windows power-mode settings during the first llama-bench run. See "What is still unknown".

## 2. Goal

**More than 300 tokens/s decode with Qwen3.5-0.8B on this laptop.** The goal was stated for a single continuous reply; the early experiments also checked aggregate throughput across several concurrent requests, because that is a much easier target. The current focus is a single reply (see section 8 and the plan).

## 3. Quant baseline (llama-bench) [measured]

Source: `docs/experiments/2026-10-26-baseline-quants.md`. `scripts/bench-quants.ps1`: `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3` (only 3 repetitions, so the SDs are crude). Mean +/- standard deviation as printed by llama-bench.

| Quant | Size | Decode tg128 (tok/s) | Prompt pp512 (tok/s) |
|---|---|---|---|
| Q4_K_M | 542 MiB | 114.5 +/- 12.9 | 6617 +/- 1080 |
| Q5_K_M | 606 MiB | 100.0 +/- 13.5 | 6849 +/- 593 |
| Q6_K | 649 MiB | 94.5 +/- 12.9 | 6661 +/- 591 |
| Q8_0 | 786 MiB | 75.5 +/- 1.5 | 7129 +/- 463 |
| BF16 | 1.44 GiB | 51.5 +/- 4.9 | 4937 +/- 1337 |

Reading it:
- Decode falls as the file gets bigger, roughly in line with "bytes streamed per token" (Q4_K_M to Q8_0 is 1.45x the bytes and 0.66x the speed). Prompt processing is flat at about 6600-7100 tok/s for the quantised files, i.e. compute-bound and not sensitive to bits per weight (BF16 is lower).
- The standard deviations on Q4_K_M, Q5_K_M and Q6_K (about 11-13%) are large. Differences of ~10-15 tok/s between those three are inside the noise of that run. Q8_0 happened to have a small SD. Treat the ordering Q4 > Q5 > Q6 > Q8 > BF16 as likely but the exact gaps as soft.
- These numbers were taken under unsteady power/thermal conditions (README note). They under-state what the machine does when steady (next section).

## 4. Server baseline: 139 tok/s single stream [measured]

Source: `docs/experiments/2026-10-26-parallel-subtasks.md`, "Baseline A". `llama-server` with Q4_K_M, `-ngl 99 -fa on -np 1 -c 2048 --no-webui`, one ~400-word reply ("explain how electric cars work", 476 tokens, 404 words), temperature 0, `cache_prompt=false`, 5 timed runs after a discarded warm-up.

| Metric | Value (mean +/- sd, 5 runs) |
|---|---|
| Wall time | 3.42 +/- 0.05 s |
| tokens / wall time | 139.1 +/- 2.1 tok/s (136.8-141.2) |
| Server-reported decode speed | 141.7 +/- 1.9 tok/s |
| Re-check at end of session | 137.5 / 140.0 tok/s (within ~1-2%) |
| Per-slot count | `-np 1/2/4/8/16` all gave 135-139 tok/s for a single request: extra idle slots cost <=3% |

Machine state during that run [measured]: laptop on AC (`Win32_Battery.BatteryStatus` = 2, 98% charge), power plan Balanced (unchanged), GPU idle before each start (0 MiB, 0% util, 47-60 C), SM clock about 2487 MHz throughout, mean 36-41 W while busy (sampled every ~0.4 s), max temp 70 C. `nvidia-smi` reports a power cap of about 37 W.

### Why 139 differs from llama-bench's 114.5 +/- 12.9

We did **not** re-run llama-bench under the steady conditions, so the cause is a **hypothesis**, not a result:
1. **Most likely [hypothesis]:** the first llama-bench pass ran under noisy power/thermal state (SDs of 11-13% say the machine was not steady; the README says so). The server runs were done plugged in with an idle, cool GPU at a steady ~2487 MHz and show an SD of only ~1.5%.
2. tg128 is a short test (128 tokens, ~1 s); clock ramp-up or a cold GPU matter more there than in a 476-token reply [hypothesis].
3. The harnesses differ (llama-bench synthetic tokens vs. real text via server with sampling at temperature 0). Sampling cost is small for this model but is not zero [hypothesis].
The honest statement is: with steady conditions this machine decodes Qwen3.5-0.8B Q4_K_M at about 139-142 tok/s through llama-server; llama-bench's 114.5 should be treated as a noisy, pessimistic first measurement. Controlling for this by tuning power/clock state is out of scope for the plan; instead all future comparisons are ratios against a same-session plain-decode control (see the plan's methodology).

The GPU at ~2487 MHz and ~37 W cap suggests the card is power-limited rather than temperature-limited [hypothesis; temperatures stayed <= 70 C].

## 5. Research conclusions (external sources + reasoning)

All URLs below were read while writing this document (or earlier in the research pass). Numbers from them are **[reported]**, not reproduced here. The model/hardware in each source differs from ours, so they are guidance, not predictions.

### 5.1 Why 300 tok/s single-stream is not reachable by tuning this 3050
- A **desktop RTX 5060** (Blackwell, 8 GB, public spec ~448 GB/s) is reported at **tg128 = 302.1 tok/s** with llama-bench and ~293-294 tok/s in the llama-server web UI, on a Qwen3.5-0.8B Q4_K_M file (497 MiB) - https://github.com/ggml-org/llama.cpp/issues/21984. That is a much larger, faster, higher-power GPU than a ~37 W 4 GB laptop 3050, and it only just reaches the target. Our 139 tok/s is 46% of that.
- Decode for a ~0.8B model is dominated by per-token fixed costs: many small kernels per token, launch/sync overhead, plus streaming ~0.57 GB of weights. At 139 tok/s that is ~79 GB/s of weight streaming alone **[estimate: 542 MiB x 139]**; 300 tok/s would need ~170 GB/s of effective streaming plus the same per-token overhead **[estimate]**. Whether this card's peak bandwidth even allows that is not recorded (see section 7).
- The qwen35 (Gated DeltaNet hybrid) decode path was profiled/optimised upstream in https://github.com/ggml-org/llama.cpp/pull/20391 (CUDA gated-delta-net kernel; merged 2026-03-11). It reports Qwen3.5-0.8B Q8_0 tg128 about 461 -> 466 tok/s (+1%) on a high-end GPU (not stated which; they mention RTX 5080/5090-class), i.e. the kernel itself is no longer the main cost and the model is already overhead-bound even on big GPUs. b11884 is newer than this PR, so we should already have it **[inference from dates]**.
- A long discussion on a related 27B qwen35 model shows per-token non-weight-streaming cost of ~3-4 ms/token on a data-centre Blackwell card under Linux, and that Windows desktop contention (compositor and other GPU apps) can silently cut decode by 1.5-1.6x - https://github.com/ggml-org/llama.cpp/issues/28196 (the author later retracted the "Windows is slower" claim as contention from a GPU-accelerated app; the lesson is: check `nvidia-smi` shows ~0% util before benchmarking, and do not drive benchmarks from a GPU-accelerated app on the same GPU). On the 27B model, `GGML_CUDA_DISABLE_GRAPHS=1` showed no benefit (single rep).
- Conclusion **[inference]**: >300 tok/s single-stream on this 3050 by tuning flags is **very unlikely**. The levers that could move it are algorithmic (speculative decoding, parallel sections); they are studied as scaling experiments in the plan.

### 5.2 Things that give ~0% on decode with a fully offloaded model
- **CPU threads (`-t`):** with all layers on the GPU, only one CPU thread submits work, so `--threads` does not help; maintainers state this explicitly and that extra threads "would just spin" - https://github.com/ggml-org/llama.cpp/issues/8684. A related thread found the busy-polling CPU core is a CUDA scheduling default (spin vs. blocking sync) that costs heat but not tok/s - https://github.com/ggml-org/llama.cpp/discussions/22238 (reported 30.2 vs 30.1 tok/s). **[reported]; not tested by us.**
- **Batch / ubatch size:** only affects prompt processing, not single-token decode **[reasoning; reported in general llama.cpp docs/discussions; not tested by us]**.
- **KV-cache quantisation (`-ctk/-ctv`):** KV is small for a 0.8B model at short context, and on the sister 27B model KV q8_0 vs f16 and FA on/off made no material difference - https://github.com/ggml-org/llama.cpp/issues/28196 **[reported]; not tested by us.**
- **Flash attention:** keep on. In the source above FA off was equal or slightly worse. Not A/B-tested by us; the baseline already uses `-fa 1`.
- **CPU-only:** the 13420H alone is far slower. A third-party benchmark for this CPU shows Llama 3.2 1B Q4_K_M generating at **36.5 tok/s** (prompt 135 tok/s) - https://www.localscore.ai/accelerator/1621 **[reported]**. For Qwen3.5-0.8B that suggests CPU-only is roughly **0.3-0.5x of our GPU speed [estimate; not run here]**. A CPU+GPU hybrid is not expected to help because the GPU already holds everything.

### 5.3 Speculative decoding (could change the picture, evidence is mixed)
- **MTP (`--spec-type draft-mtp`, `--spec-draft-n-max 1`):** Qwen3.5 GGUFs that carry an MTP head can self-draft. Reported results are contested: on a desktop RTX 3050 with **Qwen3.5-2B Q4_0**, n-max 1 gives **1.15x greedy / 1.09x sampled**, n-max 2 about 1.15x/1.05x, n-max 3 about 1.0x/0.92x, n-max 4 0.85x/0.78x, n-max 6 0.65x/0.60x - https://dev.to/gde/comparing-qwen-and-gemma-on-the-low-end-qwen-is-faster-without-drafting-gemma-has-better-quality-30bo. For a 9B MTP file the same thread reports ~1.16x at n-max 4 on a data-centre Blackwell card and a **0.57x slowdown on an Apple M4** - https://github.com/ggml-org/llama.cpp/issues/28196. A 27B model reaches 1.3-1.7x because the target is slower per token (more to hide); the smaller the model, the less there is to win. Our working range for 0.8B on this laptop: **0.9-1.2x [estimate]**, with n-max 1 the only setting likely to beat plain. **Measured later on this laptop (9.2): n-max 1 = 1.17x copy-heavy / 0.99x free text.** Each verify step runs the weights at ncols>1 where the kernels are less bandwidth-efficient, and the draft head itself costs a full vocab projection per drafted token (issue 28196's profiling).
- **Separate-model drafting:** the same tooling lists ten `--spec-type` modes in llama.cpp (draft-simple with a separate small model, draft-eagle3, draft-mtp, draft-dflash, draft-dspark and five n-gram methods) - https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g. It reports Gemma 4 E2B (a larger-than-0.8B model) gaining 1.40x on a desktop RTX 3050 and only 1.18x on a 40 W laptop GTX 1650 Ti, with the best draft length smaller on the slower card, and gains disappearing under concurrency. Takeaway: laptop-class GPUs gain less and prefer draft length 1.
- **n-gram speculation (`ngram-simple`, `ngram-map-*`, etc.):** needs no extra model, but only helps when the output **copies earlier text** (editing code, quoting the prompt, JSON/structured rewriting). Our working number is **up to ~1.7x on copy-heavy outputs, ~1.0x or slightly worse on free-form text [estimate from research notes; not in the sources above and not measured here]**. **Measured later (9.2): ngram-simple 1.66-1.95x copy-heavy, 0.92x free text.**
- **CUDA graphs on/off (`GGML_CUDA_DISABLE_GRAPHS=1`):** **untested on this machine and out of scope for the plan (hardware-specific).** Graphs should *help* tiny-model decode (less launch overhead) **[hypothesis]**; the only data point found (27B, 1 rep) showed no benefit from disabling them, which only tells us graphs were not hurting there.

### 5.4 Source list
| URL | Used for |
|---|---|
| https://github.com/ggml-org/llama.cpp/issues/21984 | RTX 5060 + Qwen3.5-0.8B Q4_K_M: 302 tok/s (bench), ~293 (server); Vulkan vs CUDA oddities |
| https://github.com/ggml-org/llama.cpp/issues/8684 | Threads do nothing with full GPU offload (maintainer explanation) |
| https://github.com/ggml-org/llama.cpp/discussions/22238 | One CPU core spins at 100% under full offload; blocking sync fixes heat, no t/s change |
| https://github.com/ggml-org/llama.cpp/pull/20391 | CUDA gated-delta-net kernel work; Qwen3.5-0.8B tg128 ~461->466 on a high-end GPU |
| https://github.com/ggml-org/llama.cpp/issues/28196 | qwen35 decode analysis, MTP depth sweeps, verify-step cost, Windows contention warning, graphs-off test |
| https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g | Drafters on small models/small GPUs, spec-type list, laptop vs desktop gains |
| https://dev.to/gde/comparing-qwen-and-gemma-on-the-low-end-qwen-is-faster-without-drafting-gemma-has-better-quality-30bo | Qwen3.5-2B on desktop RTX 3050: tg128 132.8, MTP n-max sweep |
| https://www.localscore.ai/accelerator/1621 | i5-13420H CPU-only reference (Llama 3.2 1B Q4_K_M: 36.5 tok/s) |

## 6. Parallel sub-task experiments [measured]

Full detail: `docs/experiments/2026-10-26-parallel-subtasks.md` (tables below are copied from it; "+/-" is sd, ranges omitted for brevity). Raw JSON: `results/2026-10-26-parallel-subtasks.raw.json`, `...-saturation.raw.json`. Scripts: `scripts/parallel_subtasks.py`, `scripts/make_report.py`.

Setup: Qwen3.5-0.8B Q4_K_M, llama-server, `-ngl 99 -fa on -c 2048*np --no-webui`, raw `/completion` with the Qwen chat format and an empty think block, temperature 0, seed 1, `cache_prompt=false`, Python stdlib threaded client. One discarded warm-up pass per server config, then 5 timed runs (3 for saturation). Fresh server per `-np`; np=1 repeated at the end as a drift check (within 1-2%). Laptop on AC, Balanced plan.

### 6.1 Independent short tasks (the "easy" parallel case)

**8 tasks** (n_predict 100; np=1 sequential, np>1 all submitted at once):

| server | total tokens | job wall s | aggregate tok/s | per-stream decode tok/s | speed-up vs np=1 |
|---|---|---|---|---|---|
| np=1 | 544 | 4.37 +/- 0.12 | 124.5 +/- 3.4 | 140.7 +/- 2.6 | 1.00x |
| np=2 | 543 | 2.96 +/- 0.07 | 183.9 +/- 4.0 | 110.8 +/- 2.8 | 1.48x |
| np=4 | 545 | 2.25 +/- 0.25 | 244.7 +/- 23.6 | 75.8 +/- 6.0 | 1.94x |
| np=8 | 544 | 2.56 +/- 0.21 | 213.4 +/- 16.8 | 48.9 +/- 2.0 | 1.70x |
| np=16 | 543 | 3.22 +/- 0.40 | 170.7 +/- 18.7 | 29.5 +/- 4.5 | 1.36x |
| np=1 (end re-check) | 544 | 4.42 +/- 0.02 | 123.0 +/- 0.7 | 138.9 +/- 1.0 | 0.99x |

**16 tasks:**

| server | total tokens | job wall s | aggregate tok/s | per-stream decode tok/s | speed-up vs np=1 |
|---|---|---|---|---|---|
| np=1 | 1307 | 10.30 +/- 0.23 | 126.9 +/- 2.8 | 140.3 +/- 2.2 | 1.00x |
| np=2 | 1294 | 6.60 +/- 0.15 | 196.1 +/- 4.3 | 110.1 +/- 2.1 | 1.56x |
| np=4 | 1311 | 5.55 +/- 0.28 | 237.0 +/- 12.7 | 70.9 +/- 2.2 | 1.86x |
| np=8 | 1305 | 5.27 +/- 0.53 | 249.9 +/- 24.1 | 45.2 +/- 2.6 | 1.96x |
| np=16 | 1288 | 4.49 +/- 0.26 | 287.6 +/- 15.8 (max run 308.4) | 27.2 +/- 1.1 | 2.29x |
| np=1 (end re-check) | 1307 | 10.40 +/- 0.10 | 125.7 +/- 1.2 | 139.6 +/- 1.0 | 0.99x |

**Saturation check, 48 tasks submitted at once, 3 timed runs:**

| server | total tokens | job wall s | aggregate tok/s | per-stream decode tok/s | peak VRAM MiB | mean power W (busy) |
|---|---|---|---|---|---|---|
| np=8 | 3898 | 14.23 +/- 0.30 | 274.1 +/- 6.3 | 42.4 +/- 0.8 | 1029 | 36.9 |
| np=16 | 3912 | 13.28 +/- 0.57 | 295.0 +/- 12.1 (max run 308.9) | 24.6 +/- 0.8 | 1387 | 31.0 |
| np=32 | 3906 | 13.90 +/- 0.60 | 281.5 +/- 12.7 | 14.5 +/- 0.2 | 2115 | 32.0 |

Takeaways [measured]:
- Independent tasks plateau at about **270-300 tok/s aggregate** (np=8/16/32 are the same within noise). Mean at np=16 is ~288-295 tok/s; only single runs (308.4 and 308.9) exceeded 300. That is about 2.0-2.3x the sequential 125-127 tok/s.
- Per-stream speed at np=16 is only ~25-27 tok/s, so each individual task takes ~5x longer. **This does not make one reply faster.**
- Small job sets suffer ramp/tail effects (8 tasks: best np=4; np=16 leaves half the slots idle).
- The likely reasons for the plateau (memory-bandwidth/power cap, hybrid recurrent layers, per-step sampling overhead) are **hypotheses, not measured**.

### 6.2 One reply: single stream vs skeleton-of-thought (SoT)

Baseline A (one ~400-word reply, np=1): 476 tokens, 404 words, 3.42 s, 139.1 tok/s, 0.85 s per 100 words.

Skeleton-of-thought: step 1 asks for a 6-point outline (<= 80 tokens), step 2 expands each point as its own request (np=1 sequentially, np>1 concurrently), step 3 concatenates. Section length varies because the model does not follow word counts closely.

SoT, **short sections** (columns trimmed from the report):

| server | outline s | expand s | total wall s | reply tokens | reply words | effective tok/s | per-stream tok/s | words-normalised speed-up vs Baseline A |
|---|---|---|---|---|---|---|---|---|
| np=1 | 0.55 | 2.78 | 3.33 +/- 0.07 | 351 | 305 | 105.4 +/- 2.1 | 141.9 | 0.78x |
| np=2 | 0.59 | 2.21 | 2.79 +/- 0.08 | 428 | 375 | 153.4 +/- 3.0 | 112.3 | 1.14x |
| np=4 | 0.61 | 1.86 | 2.47 +/- 0.25 | 409 | 357 | 166.5 +/- 11.1 | 79.7 | 1.23x |
| np=8 | 0.66 | 2.32 | 2.97 +/- 0.30 | 376 | 330 | 127.4 +/- 13.5 | 35.0 | 0.94x |
| np=16 | 0.73 | 2.69 | 3.41 +/- 0.66 | 383 | 334 | 115.3 +/- 20.2 | 31.6 | 0.83x |

SoT, **long sections**:

| server | outline s | expand s | total wall s | reply tokens | reply words | effective tok/s | per-stream tok/s | words-normalised speed-up vs Baseline A |
|---|---|---|---|---|---|---|---|---|
| np=1 | 0.55 | 4.52 | 5.08 +/- 0.07 | 589 | 522 | 116.0 +/- 1.7 | 140.9 | 0.87x |
| np=2 | 0.58 | 2.88 | 3.46 +/- 0.20 | 552 | 490 | 159.6 +/- 4.0 | 111.9 | 1.20x |
| np=4 | 0.61 | 3.01 | 3.62 +/- 0.51 | 605 | 533 | 170.1 +/- 26.5 | 74.8 | 1.25x |
| np=8 | 0.64 | 3.79 | 4.43 +/- 0.40 | 604 | 534 | 136.9 +/- 10.1 | 33.0 | 1.02x |
| np=16 | 0.63 | 4.40 | 5.03 +/- 0.85 | 625 | 551 | 127.3 +/- 21.8 | 29.9 | 0.93x |

Takeaways [measured]:
- SoT is at best **1.14-1.25x faster per word** (np=2/4) than just writing the reply; sequential SoT is slower (0.78-0.87x); np=8/16 give no gain (the slowest section and the shared-batch slowdown dominate).
- It never approaches 300 tok/s: only 6 parallel sections, an unavoidable sequential outline step, and the longest section is the critical path.
- **Quality was clearly worse** (see below). One outline from this model, quoted from the report: "1. Electric motors convert electrical energy into motion. 2. Batteries store chemical energy... 3. The battery pack powers the electric motor. 4. The motor drives the wheels... 5. The battery is charged by sunlight or electricity. 6. The car uses the stored energy to travel." Points overlap heavily, so the stitched reply restates "battery -> motor -> wheels" in almost every section, has no transitions or conclusion, sometimes drifts (a "solar charging / DC charge controller" section; one section said "the wheels drive the motor"). The sequential reply is more coherent and covers more distinct ground, though it also has inaccuracies. The quality assessment was by reading a single topic, **not** a rubric over several topics [limitation].

### 6.3 GPU state and VRAM during the runs [measured]

| server | peak VRAM MiB (whole GPU) | max temp C | mean power W (busy) | mean SM clock MHz | GPU idle before start (MiB / util % / temp C) |
|---|---|---|---|---|---|
| np=1 | 703 | 69 | 39.9 | 2488 | 0 / 0 / 47 |
| np=2 | 749 | 70 | 41.1 | 2487 | 0 / 0 / 60 |
| np=4 | 843 | 70 | 39.7 | 2487 | 0 / 0 / 60 |
| np=8 | 1029 | 70 | 37.0 | 2487 | 0 / 0 / 59 |
| np=16 | 1387 | 68 | 35.8 | 2487 | 0 / 0 / 59 |
| np=1 (end re-check) | 703 | 69 | 40.4 | 2488 | 0 / 0 / 59 |
| np=32 (saturation run) | 2115 | 67 | 32.0 | - | - |

VRAM is nowhere near the 4 GB limit (0.7 GiB at np=1, 1.4 GiB at np=16, 2.1 GiB at np=32). Capacity is not the limiter. Mean power per busy stretch fell slightly at higher np (40 W -> ~31-36 W), consistent with something other than raw power being the limiter [observation; cause unknown].

## 7. Caveats

- All speed numbers are for one laptop, one llama.cpp build (b11884), one model family, Q4_K_M unless stated. Nothing here was run on battery or on a different power plan.
- The llama-bench baseline had large run-to-run noise (SD up to ~13 tok/s); the server baseline is much tighter but was a single session and a single prompt (an explanatory "electric cars" reply).
- Sampling was temperature 0, thinking off. Real use with sampling or thinking on, or longer prompts/contexts, was not measured.
- `-c` was small (2048 per slot). Long-context behaviour is untested.
- Tokens/s depends on text: the server baseline's 476-token reply is one sample. Speculative-decoding gains are strongly text-dependent.
- The external numbers in section 5 are other people's results on different GPUs and models. Where they are used to predict ours the prediction is an **estimate**.
- Skeleton-of-thought quality was judged qualitatively on one topic.
- Aggregate "tok/s" from parallel runs is throughput across requests; it must not be reported as "the model runs at 300 tok/s".

## 8. What is still unknown

The plan was refocused on **general scaling behaviour** (hardware-independent metrics such as acceptance rate, tokens per verify step, speed-up ratio vs plain decode, quality scores), not on tuning this laptop's power/clock/runtime settings. Open questions, mapped to [PLAN-single-reply.md](PLAN-single-reply.md):

1. How decode speed scales with model size and bits per weight across a family (Qwen3.5 0.8B/2B/4B, Qwen3 0.6B/1.7B/4B), whether per-token time follows `a + bytes/bandwidth`, and the KLD/perplexity cost of each quant step (plan S1). So far we only have the noisy 0.8B quant table.
2. Acceptance rate, tokens per step and speed-up of n-gram speculation (n/m sweeps) and MTP (`draft-mtp`, n-max sweep) vs task type (copy-heavy vs free text) and vs target model size (plan S2). 0.8B answered in 9.2 (n-gram 1.66-1.95x copy-heavy / 0.92x free; MTP n-max 1 1.17x / 0.99x); the dependence on target size (2B/4B) is still open.
3. (Answered for 0.8B in 9.3: ~1.5x per word, quality not held.) Whether an improved skeleton-of-thought (distinct outline points, section context, shared prefix, intro/conclusion, rubric over >= 5 topics) reaches acceptable quality with a real speed-up, and how the speed-up scales with number of sections, section length and `-np` relative to the ideal critical-path bound (plan S3).
4. Whether lookahead/Jacobi decoding, a small compatible draft model or best-of-n gives anything for tiny models (plan S4); `llama-lookahead` is not in the b11884 package and a compatible sub-0.8B Qwen3.5 drafter probably does not exist.
5. Whether the single-reply >300 tok/s target is feasible on this machine: current evidence says no by tuning; speculation on copy-heavy output is the only plausible route, and the evidence on 0.8B is weak.

Deliberately **out of scope** (hardware-specific, see the plan's last section): power/clock tuning (including the 114 vs 139 tok/s question, treated as measurement noise), CUDA graphs on/off, and runtime micro-settings (`-c`, `--no-warmup`, `--no-mmap`, `-ub`, threads, KV-quant, Vulkan). This GPU's exact memory bandwidth also remains unrecorded.
## 9. Scaling experiments on Qwen3.5-0.8B (run 2026-10-26; 0.8B only, 2B/4B not yet run)

Hardware, methodology, tables and raw data: `docs/experiments/2026-10-26-scaling-0.8b.md` and the `*.raw.json` files next to it. Plan items filled in: [PLAN-single-reply.md](PLAN-single-reply.md) S1, S2a/S2b, S3. Tags: **[measured]** = run here in this session; **[estimate]** = reasoning; **[reported]** = other people's number. Every speed number here is for this laptop; ratios, acceptance and quality metrics are the parts that should transfer.

### 9.1 S1: speed vs bits per weight (14 quants), quality vs bf16

- **[measured]** llama-bench (-ngl 99 -fa 1, pp512/tg256, 5 reps, two passes within 0.6%): bf16 71.5, Q8_0 121, Q6_K 133, Q5_K_M 140, Q4_K_M 155, Q4_0 171, Q2_K 177 tok/s (SD < 1.2). **Q4_K_M = 1.28x Q8_0** and 2.17x bf16. Plan expectation was 1.3-1.5x for Q4_K_M vs Q8_0 [estimate]: slightly below, because the byte ratio (1.45x) is not fully converted into speed.
- **[measured]** Below Q4_K_M the gain is small and the quality price large: Q4_0 +10% speed for 2.4x the KLD (0.074 vs 0.030); Q2_K +14% speed for 14x the KLD (0.416; same-top-token 66.9%, PPL 29 vs 19). The plan's "<= 10% below Q4 for 0.8B" [estimate] is about right for 4-bit-class files and slightly exceeded by Q2_K.
- **[measured]** Speed is not a function of file size alone: Q3_K_M (484 MiB) is slower (149.5) than Q4_0 (502 MiB, 170.6) and IQ3_M (474 MiB, 164.1). Format-specific dequantisation cost matters; a `t = a + b*bytes` line fits all 14 files at R^2 0.98 only because bf16 anchors it (R^2 0.83 without bf16; intercept a ~2.3-3.1 ms/token, about 40-50% of a Q4_K_M token). Claim "per-token time is affine in model bytes" is therefore only partly supported at one model size; the layer-count part of the claim needs 2B/4B and is untested.
- **[measured]** Prefill is flat (7.7-9.2k tok/s) across quants; bf16 6.9k.
- **[measured]** KLD vs bf16 / same-top-token: Q8_0 0.0013 / 97.9%; Q6_K 0.0038 / 96.3%; Q5_K_M 0.0101 / 94.2%; Q4_K_M 0.0303 / 90.2%; IQ4_XS 0.050 / 87.8%; Q4_0 0.074 / 85.4%; Q3_K_M 0.122 / 81.9%; Q2_K 0.416 / 66.9%. Perplexity on ~10k tokens (se +/- 0.8) cannot separate the quants above Q4_K_M from bf16; KLD can.
- Practical reading for one reply on this laptop [estimate from the above]: Q4_K_M is the sensible default (1.28x Q8_0 at KLD 0.03); Q5_K_M if quality matters more (1.16x Q8_0, KLD 0.010).

### 9.2 S2: speculative decoding on the 0.8B (Q4_K_M, greedy, 6 copy-heavy + 6 free-text prompts, 5 reps; plain = ~140 tok/s)

Tables and raw data: `docs/experiments/2026-10-26-s2-spec-0.8b.md`; plan entry: [PLAN-single-reply.md](PLAN-single-reply.md) S2. 2B/4B (S2c) and temperature 0.8 were not run.

- **[measured] n-gram speculation helps only on copy-heavy output and costs on free text.** `ngram-simple` n=3 m=16: 1.95x on copy-heavy (273 tok/s; 59% acceptance, 4.2 tokens/step), but 1.66x when repeated later in the session (235 tok/s, same acceptance) and 0.92x (0.86x repeat) on free text. Best no-loss free-text setting n=4 m=4: 0.99x (copy 1.40x). Plan expectation was ~1.0x free and 1.3-1.7x copy [estimate]: copy-heavy is at or above it, free text is worse. `ngram-map-k` (best 1.03x copy / 0.95x free) and `ngram-map-k4v` (0.73x / 0.64x) are slower than plain because of per-step overhead; `ngram-mod` match=8 reaches 1.75x copy (cold ~1.7x) but its free-text 1.07x is a replay artifact of the warm-up (cold 0.97x).
- **[measured] MTP (draft-mtp) on 0.8B is a small win only on copy-heavy text.** n-max 1: 1.17x copy / 0.99x free (bartowski file with head: 1.18x / 0.99x), inside the plan's 0.95-1.2x. On free text every larger n-max is a loss (0.97x, 0.86x, 0.79x, 0.63x for n-max 2, 3, 4, 6; acceptance falls 64% -> 22%). On copy text speed-up rises to 1.54x at n-max 4 (218 tok/s) before falling at 6 - contradicting the plan's "peak at 1-2" for copy-heavy.
- **[measured] MTP + ngram-simple works** (copy-heavy 1.60-1.66x, 227-235 tok/s; free text 0.78-0.90x) but is no better than ngram-simple alone, so the combination does not stack on this machine.
- **[measured] >300 tok/s single reply:** reached only on individual copy-heavy replies (max 501 tok/s on one reply), not as a mean over the copy prompts (best 273; 235 on repeat) and not on free text (max 149 on one reply, means ~140). This supports 5.1 and the "free text stays near 140" expectation; the copy-heavy 300 route is a special-workload peak [estimate of practical meaning].
- **[measured] Acceptance and tokens/step are reproducible and machine-independent** (identical across the two passes); tok/s is not (MTP pass 1 ran in a slow machine state and is superseded). **Caveats:** small prompt set, copy-heavy prompts are best-case editing tasks, speculative free-text replies are not byte-identical to plain and no quality score was run.

### 9.3 S3: improved skeleton-of-thought for ONE reply (0.8B Q4_K_M, greedy, 6 topics main / 3 topics grid, 3 reps)

Tables, method and raw data: `docs/experiments/2026-10-26-s3-sot-0.8b.md`; plan entry: [PLAN-single-reply.md](PLAN-single-reply.md) S3. Speed is the ratio of words per second of the stitched reply to a same-session sequential reply of similar length; the 0.8B ignores word targets so lengths differ between cells.

- **[measured] Speed: 1.49x per word** for the improved variant (distinct outline + section context + intro + conclusion; N=6, 100-token sections, np=4; range 1.40-1.59 over 6 topics) vs **1.25x** for the naive outline (1.09-1.49; matches the earlier 1.14-1.25x). Best grid cell 1.72x (N=6, 200-token sections, np=4, T1-T3); best np 8 at 1.59x (np=4 1.50x on the same topics). Plan expectation was 1.3-1.8x per word at N=4-6, np=3-4 [estimate]: **met on speed**.
- **[measured] Scaling shape.** Speed-up grows with section length (1.0-1.1x at 50 tokens, 1.4-1.5x at 100, 1.6-1.7x at 200: the outline/intro cost is fixed), peaks at N=6 (N=8 is ~0.1x lower at every length, with more truncated/repeated sections) and saturates at np=4 (np=8 +0.0-0.1x); sequential SoT (np=1) is 0.83x. Realised/ideal is 0.4-0.55 at np>=4 because per-stream decode falls to 0.53x (np=4) / 0.35x (np=8) of single-stream while aggregate rises only to ~1.7x; so "speed-up follows ideal up to the knee" holds at np<=2-3 at best [claim only weakly supported]. `ideal` (from tokens) and the shape transfer; the ~1.7x aggregate ceiling is this laptop GPU.
- **[measured] Quality did not hold.** Auto rubric 1.33 vs 1.72 sequential (77%; plan asks within 10%), keyword coverage 0.67 vs 0.81, distinct-4gram 0.91 vs 0.92, repeated-sentence rate 0.017 vs 0.074 on average but 0.03-0.04 vs 0.00 in 3 of 6 topics. The distinct outline prompt does fix the naive failure (max title overlap 0.51 -> 0.17, repetition 0.079 -> 0.017, distinct-4gram 0.87 -> 0.91), supporting "quality depends on outline distinctness". Manual read of 2 topics: vaccines (b) good, deeper than sequential; electric cars (b) had a distinct-looking but wrong outline ("engine produces heat", "exhaust gases cool the engine"), copied into two sections and the intro/conclusion - **outline errors propagate and automatic metrics do not catch them**. No variant has section transitions. **Plan success criterion (>=1.4x AND rubric within 10% AND no worse repetition) is not met; the speed half is.**
- **[measured] Prefix reuse (cache_prompt with every slot primed) gives nothing** (1.29x vs 1.45x without on T1-T3; 438 of ~1130 prompt tokens cached, but prompts are ~100-150 tokens so prefill is negligible and priming adds a serial step) [only matters for long shared contexts - estimate].
- **Caveats:** 3 reps (plan: 5), 6 topics (3 in grid/np/prefix), one model, greedy; sequential control does not follow the word target and long sequential replies sometimes loop (T3, T5, N=8/L=200 cells), which lowers the sequential quality baseline in those cells (so SoT looks better there than it would against a loop-free control); the (b) vs (a) comparison changes outline prompt, context and intro/conclusion at once; the rubric is automatic and crude, the manual read is one non-blind reader on two topics. Not run: N 2/3/12, L=400, np 2/3/6/16, ablations of context and intro/conclusion, blind rubric.
- Practical reading [estimate]: a single-reply speed-up of ~1.5x on a 0.8B costs visible quality and needs an outline the model gets right; for short replies (< ~150 words) SoT gives ~1.0x. It cannot approach 300 tok/s (best single SoT reply: 268 tok/s effective, np=8).

## 10. bf16 free-text single stream (branch `bf16-free-text`, 2026-10-26)

Tables, method, raw data: `results/2026-10-26-bf16-b0-b3.md`; plan: [PLAN-bf16-free-text.md](PLAN-bf16-free-text.md). Non-quantized bf16 0.8B target, one greedy free-text reply (6 prompts, 256 tokens, 5 runs after a discarded warm-up, same-block plain controls). All speed-ups are ratios vs the plain bf16 server control (about 68-69.5 tok/s).

- **[measured] The old 51 tok/s (llama-bench) vs 71 tok/s (server) gap did not reproduce.** llama-bench today: 71.7 +/- 0.1 (also 71.5 with the old `-p 512 -n 128` style, 70.4 with `--no-warmup`); the server with identical flags: 69.45 +/- 0.20, i.e. bench is ~3 % (0.46 ms/token) faster. `-np`, `-kvu` make no difference; llama graph reuse is 99 % (no per-token rebuild). The old 51 +/- 4.9 was an unrepeatable slow-state outlier; cause not found. This also shows the "fixed overhead" model in the research note should not be calibrated on the 51.
- **[measured] No flag/env makes plain bf16 faster than default.** CUDA graphs on = +8 % (disable: 0.924x), fusion +3 % (0.971x), graph reuse +4.5 % (0.957x); `GGML_CUDA_GRAPH_OPT=1` 1.000x; `-fa off` 0.994x; ctx 2048/8192 1.00x; length 256 vs 512 1.01x; f16 weights 1.02x +/- 0.01 (= equal); `-t 8` = default.
- **[measured] `--backend-sampling` (`-bs`): 1.040x +/- 0.001** (70.94 vs 68.2 tok/s, byte-identical greedy output); also helps without `-kvu`.
- **[measured] Native MTP on the bf16 target** (the bf16 GGUF already contains the MTP block): n-max 1 = 1.088x (74.1 tok/s; accept 62.8 %, 1.62 tok/cycle, cycle 21.9 ms vs 14.7 ms plain), n-max 2 = 1.065x, n-max 3 = 0.951x; `--spec-draft-p-min` 0.5 / 0.75 lowers it to 0.84-0.94x in all cells. Each draft token costs ~5.3 ms (full 248k head). **Best so far: MTP n-max 1 + `-bs` = 79.18 +/- 0.07 tok/s = 1.147x.** Only 10/30 outputs are identical to plain greedy (speculative numerics); no loops seen in any config on these prompts.
- **[measured] Different target (Q4 MTP file):** plain 148 tok/s (2.16x only because it is 4-bit); MTP does not help it (0.99x / 0.96x / 0.85x). **Q4_K_M as draft for the bf16 target (one setting): 0.505x.**
- **[measured] B4: lossless trimmed draft vocabulary for MTP (FR-Spec idea, issue #25187): 1.595x.** Tables/patch: `results/2026-10-26-bf16-b4-trim-draft.md`, `patches/b4-trim-draft-head.patch`. The MTP draft step re-reads the tied 248k-row head (~4 ms of its ~5.3 ms); a patch to llama.cpp b11884 (qwen35 MTP graph + host id mapping) gathers the K most frequent head rows at load time and lets only the draft choose within them, while the target verifies over the full vocabulary. Built without the CUDA toolkit (not installed): MSVC build of the patched source + the release `ggml-cuda.dll` (the hybrid build matches the release speed: 79.14 vs 79.23 tok/s). Same-session ratios vs plain bf16 (69.0 tok/s): untrimmed MTP n-max 1 + `-bs` 1.148x (previous best), **8k-token draft head n-max 1 1.413x (97.6 tok/s), n-max 2 1.595x (110.1 tok/s)**, 4k/16k within 1 %, 32k 1.54x, 2k 1.56x; n-max 3 / 4 1.58x. Acceptance on these English prose prompts drops only 0.4-1 pp at 8-16k (62.8 -> 61.8 %); output is identical to untrimmed MTP in 30/30 cases (n-max 1, 2) and equals plain greedy in 10/30, as for untrimmed MTP (speculative numerics, not a trim effect). A GGUF-only route is not possible (the MTP head tensor must have n_vocab rows). Caveats: ranking corpus is English prose/repo text and K was tuned on the same 6 prompts; code/other languages and temperature > 0 untested; env-var-driven patch, not upstreamed.
- **Not done / caveats:** lossy target-head trim, megakernel probe and profiling (B6), kernel work (B7), temperature > 0, quality scoring of speculative output; machine-state drift of 1-4 % between blocks.

- **[measured] E1 (2026-10-27): B4 robustness.** Free text trimmed 8k = 1.64x (vs control at the start of the day; 1.57x vs the end control, machine drift 5 %). **Code:** untrimmed MTP 1.41x (acceptance 74 %), trimmed 8k 1.48x (98.5 tok/s, 40.8 %), trimmed 32k **1.88x** (124.9, 68.5 %). **Non-English (es/de/zh/hi):** untrimmed acceptance 39 % -> 1.00x, trimmed 12-20 % -> 1.01-1.05x. **T=0.7/top-p 0.9:** trimmed 8k 104.8 tok/s (1.56x), probabilistic draft sampling no better. Details `results/2026-10-27-bf16-E1-E5.md`.
- **[measured] E2a (adaptive draft length / confidence gate): negative.** On free text, history-adaptive n in [1,3]/[1,4] = 93 tok/s vs 109 for fixed n-max 2; `--spec-draft-p-min 0.5` raises acceptance to 70 % but is 0.77-0.79x; fixed n-max 3 ties n-max 2. E2b (tree drafts) infeasible (single linear draft per sequence in the server, recurrent state cannot be forked).
- **[measured] E3 (n-gram + MTP).** On copy-heavy prompts `--spec-type ngram-simple,draft-mtp` (N4/M8, n-max 8) = 201 tok/s = 2.97x plain (MTP alone 2.17x); ngram-mod,draft-mtp cold 190-195 tok/s. On free text both are 0.8x of MTP alone. ngram-mod keeps a cross-request pool, so a benchmark that replays identical prompts (warm-up pass) reports a fake 592 tok/s; measure it cold.
- **[measured] E4 (context-adaptive draft vocabulary, `LLAMA_MTP_TRIM_FIXED=4096 LLAMA_MTP_DYN=16`):** non-English 86.6 tok/s = 1.26x plain (trimmed 8k 1.01x), acceptance 12 -> 33 %; code 114.0 tok/s = 1.71x (trimmed 8k 1.48x); free text 105.6 (~3 % below the static head). Output byte-identical to trimmed 8k (42/42 pairs), target verifies every token. Open: dynamic rows on the 32k head for code.
- **[measured/est] E5 (exact trimmed target head).** CPU study: Q8 first pass + <=16 gathered bf16 rows certifies the exact argmax at 100 % of 2560 positions. GPU ceiling with a (lossy) Q8 head: plain 77.1 vs 65.1 tok/s, trim8k-MTP 121.3 vs 109.0 (+11 %); a real lossless implementation est. +5-8 %, not built.
