# PLAN - speeding up ONE continuous reply

Status: all experiments below are **planned** (nothing here has been run). Background and measured results: [FINDINGS.md](FINDINGS.md). Gains marked **[estimate]** are guesses from the research in FINDINGS.md section 5, not measurements; the purpose of the plan is to replace them with numbers.

## Goal and honest expectation

- **Stated goal:** more than 300 tok/s decode with Qwen3.5-0.8B on this laptop (i5-13420H, RTX 3050 A Laptop 4 GB, llama.cpp b11884, CUDA 12.4). **Current focus: a single continuous reply** (one user, one stream, readable text), not aggregate throughput across many requests.
- **Where we are [measured]:** 139 tok/s (server, Q4_K_M, steady state, plugged in). Parallel slots reach ~270-300 tok/s aggregate but each stream runs at only ~25 tok/s, so that does not help a single reply.
- **Honest expectation:** **>300 tok/s on one free-form reply is unlikely.** A desktop RTX 5060 is reported at ~302 tok/s on this model (llama.cpp issue 21984); this card does 139. Flag tuning (threads, batch, KV quant) is expected to give ~0%.
- **What could realistically get closest [all estimates]:**
  1. Clean power/clock state and CUDA graphs: 0-10% each at most (~140 -> ~145-155).
  2. MTP self-drafting at n-max 1: 0.9-1.2x; best case ~1.15x on predictable text (~160).
  3. n-gram speculation: ~1.0x on free text, up to ~1.7x when the reply copies the prompt (editing, quoting, JSON/structured rewriting) (~240 on such tasks).
  4. Stacking (3) + (2) + (1) on a copy-heavy task could in the very best case approach ~1.7 x 1.15 x 1.1 = ~2.1x, i.e. ~300. That is an optimistic upper bound for special workloads, not a general expectation.
  5. Improved skeleton-of-thought (parallel sections): measured naive version 1.14-1.25x with worse quality; a better version might reach 1.3-2x per word if quality holds [estimate], but it changes the output, so it counts only if the quality rubric passes.
  6. A smaller model (e.g. Qwen3-0.6B) will be faster per token but is a different, weaker model, so it is a trade-off, not an optimisation of this model.

## Experiment order (priority)

| # | Experiment | Why this order | Expected gain [estimate] |
|---|---|---|---|
| 0 | (j) Measurement protocol | Prerequisite; the first llama-bench had +/-11-13% noise | n/a |
| 1 | (a) GPU/power hygiene A/B | Cheap; may explain 114 vs 139 | 0-5% over the 139 server baseline; up to ~20% vs the noisy 114 |
| 2 | (b) CUDA graphs on/off | Cheap env var; tiny models are launch-overhead heavy | 0-10% |
| 3 | (c) MTP draft n-max 1/2 | Biggest plausible generic gain | 0.9-1.2x |
| 4 | (d) n-gram speculation | Big gain but only on copy-heavy text | 1.0x free text; up to ~1.7x copy-heavy |
| 5 | (h) smaller models | Cheap reference points | +10-40% tok/s for 0.6B-class (quality drops) |
| 6 | (e) Q4_0 / imatrix quant, with quality check | Small speed gains, need a quality gate | 0-10% speed |
| 7 | (i) n_ctx, --no-warmup, --no-mmap | Cheap sanity checks | 0-3% |
| 8 | (f) improved skeleton-of-thought | Only parallel idea that speeds one reply; needs quality work | 1.3-2.0x per word if quality holds |
| 9 | (g) other parallel-for-one-reply ideas | Speculative; mostly expected negative | 0.8-1.3x |

Run (j) first, then (a), then proceed in order; (c) and (d) should be run after (a)/(b) settle the baseline.

---

## (j) Measurement protocol (applies to every experiment)

**Hypothesis:** with a fixed protocol the run-to-run SD can be kept at about 1-2% (server runs reached ~1.5%), making differences >=5% detectable.

**Method (PowerShell, from repo root):**
1. Plugged in; note the Windows power mode (Settings > Power) and `powercfg /getactivescheme`. Close GPU-using apps (browsers with video, Electron apps); check idle first: `nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv` should read ~0% util, 0 MiB of our model, temp <= ~60 C (the lesson from llama.cpp issue 28196: other GPU use silently slows decode).
2. Start a fresh `llama-server` per configuration, e.g. baseline: `.\llama-cpp\llama-server.exe -m models\Qwen_Qwen3.5-0.8B-Q4_K_M.gguf -ngl 99 -fa on -np 1 -c 2048 --no-webui --host 127.0.0.1 --port 8089` (add the flags under test).
3. In a second window log GPU state every 0.5 s to a CSV for the whole run: `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv -lms 500 > logs\gpu-<exp>-<config>.csv`.
4. Send a **fixed prompt set** (temperature 0, seed 1, `cache_prompt=false`, thinking off, raw `/completion` with the Qwen chat format and an empty think block, same as `scripts/parallel_subtasks.py`): prompt P1 = "Write a ~400-word explanation of how electric cars work" (continuity with Baseline A), P2 = a free-form story/essay prompt, P3 = a copy-heavy prompt (e.g. "rewrite this 300-word text in a formal tone" / "fix the bug in this code and print the whole file"). Keep `n_predict` at 400-700 so the reply ends naturally or at the cap; record both.
5. **One discarded warm-up request**, then **5 timed runs** per prompt per configuration. Cool the GPU to <= 60 C between configurations.
6. Record per run: tokens, wall time, server `timings.predicted_per_second`, and for speculative modes `timings.draft_n` and `timings.draft_n_accepted` (acceptance rate). Record mean, SD, min, max, and the GPU CSV summary (mean SM clock, mean/max power, max temp, throttle-reason bits).
7. Re-run the baseline config at the end of every session as a drift check (accept <= 3% drift, otherwise repeat the session).
8. A result counts as a real difference only if it exceeds both 5% and 2 x the pooled SD.
9. Save data to `results/<date>-<experiment>.md` plus a `.raw.json`, like the existing results.

**Metric:** decode tok/s (server-reported) and wall tok/s (tokens / wall time), mean +/- SD over 5 runs. **Success criterion:** baseline SD <= 2% and end-of-session drift <= 3%. **Status:** planned.

A small new script `scripts/single_reply_bench.py` (single stream, N runs, writes json + md) should be added; it can reuse the client code in `scripts/parallel_subtasks.py`.

---

## (a) GPU / power hygiene A/B

**Hypothesis:** the 114.5 +/- 12.9 llama-bench number was depressed by unsteady power state/clock ramp; a controlled state gives ~139-142 tok/s, and a "Best performance" mode, a warm GPU or a clock lock may add a few more percent. [hypothesis]

**Method:** Same server baseline (j) under:
- A0: current state (Balanced, plugged in) - the 139 baseline repeated.
- A1: Windows power mode "Best performance" (Settings > System > Power, or `powercfg /overlaysetactive OVERLAY_SCHEME_MAX` - verify the exact overlay name on this Windows build; revert afterwards with the original overlay).
- A2: A1 + manufacturer "performance/turbo" mode if the laptop has one (vendor app; note what it is).
- A3: GPU clock lock, if the driver allows it: `nvidia-smi -lgc 2100,2100` (needs an elevated shell; often unsupported on laptop GPUs; reset with `nvidia-smi -rgc`). Check `nvidia-smi -q -d PERFORMANCE,CLOCK` for the throttle reasons first (power cap vs thermal). **Needs the user's go-ahead because it changes system settings.**
- A4: warm-up variants: 1 vs 5 discarded warm-up requests; a 30 s GPU-heating pre-run vs a cold start.
- A5 (diagnostic): re-run `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 10` for Q4_K_M under the A0 steady state, to see whether llama-bench itself reaches ~140 when the machine is steady.

**Metric:** decode tok/s, SM clock, power, throttle reasons. **Expected gain [estimate]:** 0-5% over the 139 server baseline; A5 explains the gap if it shows 130-145. **Success criterion:** a configuration beating A0 by >5% and >2 SD, reproducible on re-run. **Status:** planned.

Results to fill in:

| Config | decode tok/s mean | SD | wall tok/s | mean SM MHz | mean W | max C | throttle reasons | vs A0 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| A0 Balanced, AC | | | | | | | | 1.00x | |
| A1 Best performance | | | | | | | | | |
| A2 vendor turbo | | | | | | | | | |
| A3 clock lock | | | | | | | | | |
| A4 warm-up variants | | | | | | | | | |
| A5 llama-bench steady | | | | | | | | n/a | |

---

## (b) CUDA graphs on/off A/B

**Hypothesis:** ggml's CUDA graphs cut per-token launch overhead; with a tiny model that overhead is a large share of each token, so graphs on (the default) is faster than `GGML_CUDA_DISABLE_GRAPHS=1`. The only external data point (27B, 1 rep) showed no benefit from disabling graphs. [hypothesis]

**Method:** Same prompts and protocol (j), alternate configurations ABAB so drift is visible:
- B0: default: `llama-server ... -ngl 99 -fa on -np 1`.
- B1: `$env:GGML_CUDA_DISABLE_GRAPHS = "1"` set in the shell before launching the same command (remove with `Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS`).
- Check the server log at start for graph-related lines to confirm graphs are active/disabled. Also try `-np 1` vs default `-np` (auto slots may disable graph capture for some batch shapes - check log).
- Optional: also record `nvidia-smi` GPU util % (if graphs help, util should rise).

**Metric:** decode tok/s. **Expected gain [estimate]:** B0 over B1 of 0-10%. This tells us whether launch overhead is the bottleneck: if B1 is much slower, overhead matters and more overhead reduction (speculation) is promising; if equal, we are bandwidth/power-bound. **Success criterion:** difference measured with SD < 2%. **Status:** planned.

| Config | decode tok/s mean | SD | util % | log shows graphs? | vs B0 | Notes |
|---|---|---|---|---|---|---|
| B0 default | | | | | 1.00x | |
| B1 DISABLE_GRAPHS=1 | | | | | | |
| B0 repeat | | | | | | |
| B1 repeat | | | | | | |

---

## (c) MTP self-drafting vs plain

**Hypothesis:** the MTP head can verify 1-2 drafted tokens per step; at n-max 1 it may give ~1.1x, at n-max 2 about the same or worse on a 0.8B model; acceptance is higher on code/structured text than prose. Draft cost (full-vocab output projection per drafted token, verify at ncols>1) may eat the gain on this small model. [hypothesis from reported 1.15x on Qwen3.5-2B desktop 3050 and 0.57x on an M4 for 9B]

**Method:**
- Models: the MTP-carrying GGUF `unsloth/Qwen3.5-0.8B-MTP-GGUF` (Q4_K_M; download to `models\`; **the bartowski file may not carry the MTP head**; check the server log: "creating MTP draft context"). Plain control: the same unsloth MTP file **without** `--spec-type` (to isolate the speculation cost from file differences) and the bartowski Q4_K_M baseline.
- Flags: `llama-server -m models\<mtp>.gguf -ngl 99 -fa on -np 1 -c 2048 --no-webui --spec-type draft-mtp --spec-draft-n-max 1` and again with `--spec-draft-n-max 2` (optionally 3 for the curve). Confirm flag names with `.\llama-cpp\llama-server.exe --help` (spec section) for b11884.
- Prompts P1-P3 from (j), greedy (temperature 0) and sampled (temperature 0.8) variants. Note: reported greedy output may differ slightly between plain and speculative (different kernels); compare quality by eye on a few outputs.
- Record acceptance (`draft_n_accepted / draft_n`), tokens per step.

**Metric:** decode tok/s, acceptance rate, speed-up vs plain (same file). **Expected gain [estimate]:** n-max 1: 0.95-1.2x; n-max 2: 0.9-1.15x. **Success criterion:** >=1.10x speed-up on P1 (free text) with SD < 3% and no visible quality change; any config <1.0x is dropped. **Status:** planned.

| Config | P1 tok/s | P2 tok/s | P3 tok/s | acceptance P1/P2/P3 | vs plain (same file) | VRAM MiB | Notes |
|---|---|---|---|---|---|---|---|
| bartowski Q4_K_M plain | | | | n/a | | | |
| MTP file plain | | | | n/a | 1.00x | | |
| MTP n-max 1 | | | | | | | |
| MTP n-max 2 | | | | | | | |
| MTP n-max 3 (optional) | | | | | | | |

---

## (d) n-gram speculation on copy-heavy tasks vs free text

**Hypothesis:** n-gram (prompt-lookup) drafting costs almost nothing per step and accelerates only when the reply repeats earlier text. Expect ~1.0x (or slightly below, from failed drafts) on free text, up to ~1.7x on copy-heavy outputs. [estimate; research notes, not yet measured by us]

**Method:**
- List the available types with `.\llama-cpp\llama-server.exe --help` (b11884; llama.cpp offers five n-gram `--spec-type` modes, e.g. `ngram-simple` and `ngram-map-*` variants; the exact names and the n/m size flags - likely `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-draft-n-max` - must be confirmed from `--help`).
- Sweep: for each mode, n in {2, 3, 4} (lookup n-gram size) x m in {8, 16, 32} (draft length) [lowest-risk subset first: n=3, m=16], on the plain Q4_K_M file.
- Prompt sets: **copy-heavy** (C1 "rewrite this paragraph with fixed typos", C2 "output this JSON with field X changed", C3 "fix a bug in this 100-line file and print the entire file", C4 "summarise by quoting key sentences") vs **free text** (P1, P2 from (j)). Greedy.
- Record acceptance and tok/s; also the `--spec-draft-n-max` sensitivity.

**Metric:** decode tok/s, acceptance, speed-up vs plain by prompt type. **Expected gain [estimate]:** copy-heavy 1.3-1.7x (best case); free text 0.95-1.02x. **Success criterion:** >=1.3x on copy-heavy prompts and no worse than 0.97x on free text. **Status:** planned.

| Mode / n / m | C1 tok/s | C2 tok/s | C3 tok/s | C4 tok/s | P1 tok/s | P2 tok/s | acceptance (copy / free) | vs plain (copy / free) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| plain | | | | | | | n/a | 1.00x / 1.00x | |
| ngram-simple n=3 m=16 | | | | | | | | | |
| ngram-simple n=2 m=8 | | | | | | | | | |
| ngram-simple n=4 m=32 | | | | | | | | | |
| ngram-map (variant 1) n/m | | | | | | | | | |
| ngram-map (variant 2) n/m | | | | | | | | | |
| ngram + MTP combined (if allowed) | | | | | | | | | |

---

## (e) Lower quant / Q4_0 / imatrix: speed vs quality

**Hypothesis:** smaller bytes per weight raise decode speed modestly (Q4_K_M 114.5 vs Q8_0 75.5 in the noisy baseline), but 0.8B models lose quality quickly below ~4 bits; an imatrix-calibrated quant improves quality at the same speed. [hypothesis]

**Method:**
- Candidates: Q4_K_M (baseline), Q4_0, IQ4_XS, Q3_K_M, and an imatrix Q4_K_M built by us: `llama-imatrix.exe -m models\Qwen_Qwen3.5-0.8B-bf16.gguf -f <calibration.txt> -o imatrix.dat -ngl 99`, then `llama-quantize.exe --imatrix imatrix.dat models\...-bf16.gguf models\...-Q4_K_M-imat.gguf Q4_K_M` (llama-imatrix and llama-quantize are in `llama-cpp\`). Use a calibration text of ~100-200 KB of general English plus a little code.
- Speed: protocol (j) on each file.
- Quality gate with `llama-perplexity.exe`: (1) perplexity on a fixed text (e.g. wikitext-2 test split or a ~200 KB sample), `-c 512 -ngl 99`; (2) KL divergence against the bf16 logits: `llama-perplexity -m bf16.gguf -f text.txt --kl-divergence-base logits.bin`, then for each quant `--kl-divergence-base logits.bin --kl-divergence`; report mean KLD and same-top-token %. Add a small manual check: the 5-topic prompt set from (f) read side by side.
- Note: the logits file for bf16 at 0.8B with a 200 KB text is large (hundreds of MB to GBs); keep it out of git (`logs/` is ignored).

**Metric:** decode tok/s, perplexity, mean KLD, top-token agreement. **Expected gain [estimate]:** Q4_0 vs Q4_K_M within +/-5%; Q3_K_M / IQ4_XS up to +5-10%; imatrix: no speed change, KLD improvement. **Success criterion:** keep a quant only if speed gain >= 5% and mean KLD < ~0.05 and perplexity increase < ~5% vs Q4_K_M [thresholds are a proposal]. **Status:** planned.

| Quant | file MiB | tok/s mean +/- SD | vs Q4_K_M | perplexity | mean KLD vs bf16 | same-top % | Notes |
|---|---|---|---|---|---|---|---|
| Q4_K_M | 542 | | 1.00x | | | | |
| Q4_K_M + imatrix | | | | | | | |
| Q4_0 | | | | | | | |
| IQ4_XS | | | | | | | |
| Q3_K_M | | | | | | | |
| Q8_0 (reference) | 786 | | | | | | |

---

## (f) Improved skeleton-of-thought (the one parallel idea that speeds a single reply)

**Hypothesis:** naive SoT gave 1.14-1.25x per word with clearly worse text because the outline points overlapped and sections had no context. A better design might recover quality: distinct non-overlapping points, shared prefix reuse, an intro/conclusion step, and the right slot count. [hypothesis; the gain is an estimate]

**Method (all with a fixed set of >= 5 topics, e.g. electric cars, how vaccines work, how a bill becomes law, photosynthesis, how the internet routes packets, plus a code-explanation topic):**
1. **Outline prompt:** ask for exactly N points, each a distinct sub-topic with a 3-6 word title and one-line scope, "no overlap between points", optionally with a few-shot example; use temperature 0. Check overlap automatically (e.g. word-set Jaccard between points) and re-roll if too similar.
2. **Shared prefix / prompt cache:** put the instruction + topic + full outline in a common prefix and add only "Now write section k:" at the end; enable `cache_prompt=true` so the llama-server slot cache reuses the prefix (measure the prompt-eval saving; with `-np` > 1 check slot assignment/`--cache-reuse`).
3. **Slot count:** expand sections with `-np` in {2, 3, 4, 6}, with section count = np vs section count > np.
4. **Intro and conclusion:** generate an intro (after the outline) and a conclusion (after the sections) as extra steps; conclusion runs sequentially (cost) or in parallel with the last section using the outline only (variant).
5. **Section context:** give each section its neighbours' titles ("previous: X, next: Y, do not repeat earlier points").
6. **Quality rubric (score each reply 0-2; blind to the method; ideally by the user or a stronger model):** coverage of the main aspects, repetition (repeated claims/phrases across sections), factual errors, coherence/transitions. Also automatic: distinct-4gram ratio, repeated-sentence count, word count.
7. Baselines per topic: sequential single reply of the same word count (np=1) and naive SoT from `scripts/parallel_subtasks.py`.

**Metric:** words per second end to end (outline + sections + intro/conclusion), plus rubric score and automatic repetition metrics. **Expected gain [estimate]:** 1.3-2.0x per word if quality parity can be reached (the measured naive best is 1.25x). **Success criterion:** >= 1.4x words-normalised speed-up AND average rubric within 10% of sequential on all topics AND no topic with a worse repetition score. **Status:** planned.

| Variant | topic | total wall s | words | words/s | vs sequential | coverage | repetition | factual | coherence | distinct-4gram | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sequential (np=1) | T1..T5 | | | | 1.00x | | | | | | |
| naive SoT (np=4) | T1..T5 | | | | | | | | | | |
| distinct outline (np=4) | T1..T5 | | | | | | | | | | |
| + prefix cache | T1..T5 | | | | | | | | | | |
| + intro/conclusion | T1..T5 | | | | | | | | | | |
| np=2 / 3 / 6 sweep | T1..T5 | | | | | | | | | | |

---

## (g) Other parallel-for-one-reply ideas

**Hypothesis:** extra GPU parallelism could be converted into single-stream speed only by speculation-like schemes; most of these are expected to be neutral or negative. [hypothesis]

Sub-experiments:
1. **Parallel sampling / best-of-n with early cut:** run n=4 candidates at np=4 (~75 tok/s each, measured at 4 slots) and cut all at the first sentence/paragraph where a cheap scorer (e.g. mean logprob or a repetition check) says they diverge, keep the best, continue. This improves quality at equal latency, it does not make one reply faster. **Expected speed gain [estimate]: ~0.5-0.8x** (slower per stream). Success: only if quality improves enough to allow a smaller-model/lower-quant trade. Low priority.
2. **Speculative parallel decoding by sections / Jacobi / lookahead decoding:** check whether the b11884 package contains `llama-lookahead` (it does **not** appear in `llama-cpp\` - the folder has llama-bench, -cli, -server, -perplexity, -imatrix, -quantize and others, but no lookahead or speculative binary) - if missing, build llama.cpp from source with CUDA (needs the CUDA 12.4 toolkit and MSVC; ~30-60 min) or check `llama-cli --help` for a lookahead mode. Run `llama-lookahead -m <model> -ngl 99 -p <prompt> -n 256` (flags `-W`/`-N`/`-G` window/ngram sizes may apply) and compare tok/s with plain. **Expected [estimate]:** 0.9-1.3x; lookahead is reported to help mainly when there is spare compute (big GPU, small batch); the 3050 is likely already near its limit. Only worth building if (c)/(d) disappoint.
3. **Draft with a smaller separate model (`--spec-type draft-simple -md <small.gguf> -ngld all`):** requires an **identical or compatible vocabulary** with the target. Qwen3.5 uses a ~248k-token vocabulary (Qwen3.5-2B is listed with 248,320), which differs from Qwen2.5/Qwen3 (~152k) and from tiny models such as SmolLM2-135M/360M (~49k). Therefore a 135M-360M drafter with a compatible vocab **probably does not exist**; Qwen3.5-0.8B may already be the smallest in its family. Step 1: search Hugging Face for a Qwen3.5-family GGUF smaller than 0.8B (or a distilled/pruned draft with the same tokenizer); step 2: if found, test `-md` with `--spec-draft-n-max 4/8`. If the vocab differs, llama.cpp will refuse or produce garbage - do not force it. **Expected [estimate]:** only worthwhile if the drafter is <10% of the target's per-token cost, which is unlikely at this scale (the target is already 0.8B); likely 0.8-1.2x.
4. **EAGLE3 / DFlash-style drafters:** the spec-type list also includes draft-eagle3/draft-dflash/draft-dspark; they need a matching drafter checkpoint for Qwen3.5-0.8B, which probably does not exist. Only check availability.

**Metric:** tok/s vs plain; quality as in (f). **Success criterion:** >=1.15x on free text with no quality change. **Status:** planned (all sub-items), low priority.

| Sub-experiment | prompt | tok/s | vs plain | quality notes | blocker / availability |
|---|---|---|---|---|---|
| g1 best-of-4 + early cut | | | | | |
| g2 llama-lookahead | | | | | binary missing in b11884 zip? |
| g3 draft-simple small model | | | | | compatible-vocab model found? |
| g4 eagle3 / dflash | | | | | drafter exists? |

---

## (h) Smaller / different model comparison (single-stream tok/s)

**Hypothesis:** tok/s rises as parameters and bytes fall, but not proportionally because per-token fixed overhead dominates; Qwen3-0.6B (non-hybrid, standard attention) may be faster or slower than the hybrid Qwen3.5-0.8B despite fewer parameters. [hypothesis]

**Method:** protocol (j), Q4_K_M files, same prompts: Qwen3.5-0.8B (reference), Qwen3-0.6B, and optionally Llama-3.2-1B-Instruct, SmolLM2-360M, Gemma-3-270M or Qwen2.5-0.5B (pick 2-3; confirm GGUFs exist on Hugging Face, e.g. bartowski/unsloth). Also record `llama-bench -p 512 -n 128 -r 5` under steady state and the model's quality on the (f) topics by eye (smaller models will be visibly worse; this is a trade-off, not a free gain). Reference point: Qwen3.5-2B Q4_0 reported at tg128 ~133 tok/s on a desktop RTX 3050 (dev.to article in FINDINGS.md) - useful for judging whether our laptop is behind.

**Metric:** decode tok/s, VRAM, one-line quality note. **Expected gain [estimate]:** Qwen3-0.6B +10-40% vs Qwen3.5-0.8B; tiny (135-270M) models could exceed 250-300 tok/s but with strongly reduced quality. **Success criterion:** report the speed-vs-quality curve; flag any model >300 tok/s that still passes a minimal quality check (e.g. coherent 400-word reply on all 5 (f) topics). **Status:** planned.

| Model (Q4_K_M) | params | file MiB | tok/s mean +/- SD | vs Qwen3.5-0.8B | quality note | Notes |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.77 B | 542 | | 1.00x | | |
| Qwen3-0.6B | | | | | | |
| model 3 | | | | | | |
| model 4 | | | | | | |

---

## (i) Context size, warm-up and loading flags

**Hypothesis:** at short context, `-c` (KV size) has negligible effect on decode; `--no-warmup` only changes first-request latency; `--no-mmap` can matter on some systems (reported on ROCm in issue 8684) but should be ~0% on CUDA with all weights on the GPU. [hypothesis]

**Method:** protocol (j) with P1 and one long-context prompt (~2k and ~8k tokens of input):
- `-c` in {512, 2048, 8192, 32768} (note VRAM; the hybrid's recurrent state does not grow with context, KV only on full-attention layers).
- `--no-warmup` vs default (record first-request time and later-request speed).
- `--no-mmap` vs default.
- `-ub` / `-b` (512 default vs 128 vs 2048): expect ~0% on decode, record prompt-processing t/s too.
- Optional: `-t 1/4/8` and `--threads-batch` to confirm the "threads do nothing" claim; `--poll 0`; `-ctk q8_0 -ctv q8_0` vs f16; `-fa off` (to confirm keep-on).
- Optional: Vulkan backend (`llama-server -dev Vulkan0`, needs a Vulkan build) - one report (issue 21984) saw Vulkan decode ~338 tok/s vs CUDA 302 in llama-bench on a 5060 but only ~35 tok/s in the server, so it is inconsistent; only worth it if a Vulkan build is cheap to get.

**Metric:** decode tok/s and first-request latency. **Expected gain [estimate]:** 0-3% each; most expected to be noise. **Success criterion:** any flag with >5% and >2 SD effect is promoted to the default and re-verified. **Status:** planned.

| Flag | value | decode tok/s | SD | first-request s | VRAM MiB | vs default | Notes |
|---|---|---|---|---|---|---|---|
| default | -c 2048 | | | | | 1.00x | |
| -c | 512 / 8192 / 32768 | | | | | | |
| --no-warmup | on | | | | | | |
| --no-mmap | on | | | | | | |
| -ub / -b | 128 / 2048 | | | | | | |
| -t | 1 / 4 / 8 | | | | | | |
| KV type | q8_0 | | | | | | |
| -fa | off | | | | | | |

---

## After the experiments

1. Fill the tables, write `results/<date>-<experiment>.md` per experiment (same style as existing results), update FINDINGS.md section 8 ("what is still unknown") and the README results table.
2. Combine the winners into one "best single-reply config" and run the full (j) protocol on it for a final number, with P1-P3, including a worst-case free-text prompt, so the headline is not just the best-case copy-heavy figure.
3. Report free-text and copy-heavy results **separately**; do not report aggregate parallel throughput as single-stream speed.