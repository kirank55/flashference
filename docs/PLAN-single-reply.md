# PLAN - general scaling experiments for speeding up ONE continuous reply

Status: all experiments below are **planned** (nothing here has been run). Background and measured results: [FINDINGS.md](FINDINGS.md). Gains marked **[estimate]** are guesses from the research in FINDINGS.md section 5, not measurements; the point of the plan is to replace them with numbers.

## Scope (decided 2026-10-26)

The experiments aim at **scaling behaviour that can be shown to hold generally**: algorithmic and model-level results (how speed, acceptance and quality change with model size, bits per weight, draft length, task type, number of parallel sections) that should transfer to other GPUs and machines. They are **not** about tuning this laptop. Power/clock settings, CUDA graphs, context-size and other runtime micro-settings are out of scope (last section).

**Focus:** a single continuous reply (one user, one stream, readable text), not aggregate throughput across many requests.

**Original goal, restated honestly:** >300 tok/s on one reply with Qwen3.5-0.8B on this laptop (RTX 3050 A Laptop 4 GB, llama.cpp b11884). We measured 139 tok/s [measured]; a desktop RTX 5060 is reported at ~302 tok/s (llama.cpp issue 21984). **>300 tok/s on one free-form reply is unlikely here.** The scaling studies are meant to tell us which algorithmic levers have headroom in general and how big they are; where a lever could reach 300 tok/s it will be a special workload (e.g. copy-heavy output), and the results will say so.

### Hardware-independent vs hardware-dependent metrics

- **Hardware-independent (what generalization claims rest on):** acceptance rate (accepted / drafted tokens), mean tokens accepted per verify step, quality scores (KLD vs reference, perplexity, rubric), model size in bytes and bits per weight, outline-overlap score, ideal parallel speedup computed from token counts (see S3), draft-length and task-type effects on those.
- **Hardware-dependent but comparable as ratios:** speed-up vs plain decode **on the same machine in the same session**, and the cost ratio `r = t_step(spec) / t_step(plain)` (time of one verify/draft step relative to one plain decode step). A speed-up from speculation is approximately `(mean tokens per step) / r` [model, to be validated]; acceptance and tokens/step transfer across hardware, `r` is the machine-specific constant, so the plan reports both so a reader can predict the speed-up elsewhere from their own `r`.
- Absolute tok/s are recorded for reference only and are never the headline.

### Generalization claims we hope to test (each is a hypothesis)

| Claim | Study |
|---|---|
| Decode time per token is affine in model bytes: `t = a + bytes/BW`, with the fixed part `a` growing with layer count; hence bit-width savings flatten out for tiny models | S1 |
| For a given family, lower-bit quants cost KLD in a predictable, model-size-dependent way (smaller models degrade faster per bit) | S1 |
| Acceptance rate and tokens/step depend mainly on task type (copy-heavy >> free text) and draft length, and fall with larger drafts; gain shrinks as the target model gets smaller | S2 |
| Skeleton-of-thought speed-up is bounded by the critical path (outline + longest section) and saturates at the batch-scaling knee; quality depends on outline distinctness | S3 |
| Parallel-for-one-reply schemes beyond speculation/SoT give little for tiny models | S4 |

## Methodology (minimal; not an experiment)

- Record hardware once per session (GPU model, driver, llama.cpp build, OS, `nvidia-smi` idle utilisation ~0% before starting) in each result file. Do not tune or vary power/clock state; just do not benchmark while other GPU apps are running.
- Fresh `llama-server` (or `llama-bench`) per configuration; **one discarded warm-up**, then **5 timed runs** per prompt; report **mean and SD** (and min/max). Greedy decoding (temperature 0, seed 1, `cache_prompt=false`, thinking off, raw `/completion` with the Qwen chat format and an empty think block - same as `scripts/parallel_subtasks.py`).
- **Always report ratios against a plain-decode control measured in the same session on the same machine**, and repeat the control at the end (drift <= 3%, else redo).
- For speculative modes record `timings.draft_n`, `timings.draft_n_accepted`, and tokens per step = `predicted_n / (predicted_n - draft_n_accepted)`; confirm flag names with `.\llama-cpp\llama-server.exe --help` for b11884.
- A difference counts only if it exceeds 2 x pooled SD.
- Fixed prompt sets (reuse everywhere; store them in `prompts/`): **free text** F1-F3 (explain electric cars, a short story, an essay on a given topic), **copy-heavy** C1-C4 (rewrite a paragraph with typos fixed, output JSON with one field changed, fix a bug and print the whole file, quote key sentences from a passage), n_predict 400-700.
- Save to `results/<date>-<study>.md` + `.raw.json`, like the existing results. A small `scripts/single_reply_bench.py` (single stream, N runs, md + json) should be added; it can reuse the client in `scripts/parallel_subtasks.py`.

---

## S1. Speed vs model size and bits per weight, with the quality trade-off curve

**Question / generalization claim:** how does single-stream decode speed scale with model size (parameters, layers, bytes) and bits per weight across one family, and what quality does each step down in bits cost? Claim to test: per-token time is `a + bytes/BW` with `a` roughly proportional to layer count (fixed per-token overhead); so quantising a tiny model buys less than quantising a large one, while the KLD cost of low bits is larger for smaller models. [hypotheses]

**Method:**
- Families (pick one, then the other if time allows; confirm GGUFs exist on Hugging Face, e.g. bartowski/unsloth): **Qwen3.5 0.8B / 2B / 4B** (same hybrid architecture - the cleanest scaling axis) and **Qwen3 0.6B / 1.7B / 4B** (standard attention - tests whether the hybrid matters).
- Quants per model: Q2_K or IQ2/IQ3 (if usable), Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, and bf16 where it fits (bf16 only for the smaller sizes; the 4 GB card will not hold 4B bf16 - use Q8_0 or a CPU/partial-offload run as the KLD reference for 4B and say so). Optionally one imatrix variant at Q4_K_M and at the lowest quant, built with `llama-imatrix.exe` + `llama-quantize.exe --imatrix` (both in `llama-cpp\`), to show how much imatrix recovers at low bits.
- Speed: single-stream decode, methodology above (`llama-server` or `llama-bench -p 0 -n 256 -r 5`). Record file bytes and bits per weight (`bytes*8 / params`).
- Quality: `llama-perplexity.exe` on a fixed text (~200 KB, e.g. wikitext-2 test), and KL divergence against the highest-precision model of the same size: `llama-perplexity -m ref.gguf -f text.txt --kl-divergence-base logits.bin`, then per quant `--kl-divergence-base logits.bin --kl-divergence`; report mean KLD and same-top-token %. Keep logits files out of git (`logs\` is ignored).
- Analysis: plot tok/s vs file bytes per model and fit `t_token = a + b * bytes` per family; plot `a` vs layer count; plot KLD vs bits per weight per model size; plot speed-up vs the Q8_0 (or bf16) of the same model.

**Metrics:** decode tok/s (and ratio to the same model's Q8_0), fitted `a` and effective bandwidth `1/b`, perplexity, mean KLD, same-top %. **Expected [estimate]:** Q4_K_M vs Q8_0 ~1.3-1.5x for 0.8B (measured noisy 1.52x), larger for 4B (closer to the byte ratio ~1.8x); going below Q4 adds <= 10% for 0.8B while KLD rises several-fold; smaller models show a larger `a`-to-total ratio. **Generalization claim supported if:** the affine fit has R^2 > 0.95 within each family and `a` scales with layer count across both families, and the KLD-vs-bits ordering by model size is the same in both families. **Success criterion:** at least 3 model sizes x 5 quants measured in one family with SD < 3%, and the fit parameters reported with confidence intervals. **Status:** planned.

Results to fill in (one table per model; speed-up is vs that model's Q8_0):

| Model | Quant | file MiB | bits/weight | tok/s mean +/- SD | vs Q8_0 | perplexity | mean KLD | same-top % | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | Q8_0 | 786 | | | 1.00x | | | | |
| Qwen3.5-0.8B | Q6_K | 649 | | | | | | | |
| Qwen3.5-0.8B | Q5_K_M | 606 | | | | | | | |
| Qwen3.5-0.8B | Q4_K_M | 542 | | | | | | | |
| Qwen3.5-0.8B | Q4_0 | | | | | | | | |
| Qwen3.5-0.8B | Q3_K_M | | | | | | | | |
| Qwen3.5-2B | (same rows) | | | | | | | | |
| Qwen3.5-4B | (same rows) | | | | | | | | |
| Qwen3 0.6B / 1.7B / 4B | (same rows) | | | | | | | | |

Fit summary:

| Family | a (ms/token) | effective BW (GB/s) | R^2 | layers | a vs layers note |
|---|---|---|---|---|---|
| Qwen3.5 | | | | | |
| Qwen3 | | | | | |

---

## S2. Speculative decoding scaling: acceptance and speed-up vs task type, draft settings and model size

**Question / generalization claim:** how do acceptance rate, tokens accepted per verify step, and the speed-up ratio change with (i) task type (copy-heavy vs free text), (ii) draft length (MTP `--spec-draft-n-max`; ngram `n` lookup size and `m` draft length), and (iii) target model size? Claims: acceptance is a property of the text and model pair (hardware-independent); n-gram acceptance is high only when the output copies earlier text; MTP gain shrinks as the target gets smaller because the verify/draft step costs a larger share of a plain step (reported: ~1.15x on Qwen3.5-2B, ~1.16x on 9B, ~1.3-1.8x on 27B; a 0.57x slowdown on an M4 for 9B). [hypotheses; reported figures are from other machines]

**Method:**
- **S2a - n-gram speculation (no extra model):** list the modes with `llama-server --help` (b11884 offers five n-gram `--spec-type` modes, e.g. `ngram-simple` and `ngram-map-*`; the n/m flags are probably `--spec-ngram-size-n` / `--spec-ngram-size-m` plus `--spec-draft-n-max` - confirm). Sweep n in {2, 3, 4} x m in {4, 8, 16, 32} per mode on the Qwen3.5-0.8B Q4_K_M file; all of C1-C4 and F1-F3; plus a "copy fraction" covariate (fraction of generated 4-grams that occur earlier in prompt+output) to show acceptance vs copy fraction.
- **S2b - MTP self-drafting:** model file carrying the MTP head (`unsloth/Qwen3.5-0.8B-MTP-GGUF`; check for the log line "creating MTP draft context"; the bartowski files may not have the head). Plain control = the same MTP file without `--spec-type`. Sweep `--spec-type draft-mtp --spec-draft-n-max` in {1, 2, 3, 4, 6} on F1-F3 and C1-C4, greedy and temperature 0.8.
- **S2c - vs model size:** repeat the best S2a and S2b settings (plus n-max 1 and 2) on the 2B and 4B Qwen3.5 (MTP files if published, e.g. unsloth Qwen3.5-2B/4B-MTP-GGUF; confirm). This gives the speed-up ratio vs target size curve on one machine, and acceptance vs size independent of the machine.
- **S2d - combined:** if llama.cpp allows MTP + n-gram together, measure the combination on copy-heavy text.
- Record for every cell: acceptance rate, tokens per step, speed-up vs same-file plain, and the step-cost ratio `r = tokens_per_step / speedup` [derived].
- Quality check: greedy output is not guaranteed byte-identical with speculation (reported); compare a few outputs and the rubric from S3 on a subset.

**Metrics:** acceptance rate, tokens per step, speed-up ratio vs plain, `r`. **Expected [estimate]:** n-gram: ~1.0x (0.95-1.02x) on free text, 1.3-1.7x on copy-heavy at best settings, acceptance strongly tracking copy fraction; MTP n-max 1: 0.95-1.2x on 0.8B, rising with model size (2B ~1.1-1.2x, 4B ~1.2-1.4x), peak at n-max 1-2 and falling below 1.0x by n-max 4-6; acceptance higher on code/structured text than prose. **Generalization claim supported if:** acceptance and tokens/step curves are stable across prompts of the same type and ordered the same way across 3 model sizes, and speed-up is predicted by `tokens_per_step / r` to within ~5%. **Success criterion:** complete n/m and n-max sweeps on at least 2 task types with SD < 3% on speed-up; report which setting is best per task type and per size. **Status:** planned.

Results to fill in:

| Mode | n / m or n-max | Model size | Task type | acceptance | tokens/step | r | speed-up vs plain mean +/- SD | Notes |
|---|---|---|---|---|---|---|---|---|
| plain | - | 0.8B | free / copy | n/a | 1.00 | 1.00 | 1.00x | |
| ngram-simple | n=3 m=16 | 0.8B | free | | | | | |
| ngram-simple | n=3 m=16 | 0.8B | copy | | | | | |
| ngram-simple | (full n x m grid) | 0.8B | free / copy | | | | | |
| ngram-map-* | (grid) | 0.8B | free / copy | | | | | |
| draft-mtp | n-max 1 | 0.8B | free / copy | | | | | |
| draft-mtp | n-max 2, 3, 4, 6 | 0.8B | free / copy | | | | | |
| draft-mtp | n-max 1, 2 | 2B | free / copy | | | | | |
| draft-mtp | n-max 1, 2 | 4B | free / copy | | | | | |
| ngram + MTP | best of each | 0.8B | copy | | | | | |

Copy-fraction check:

| Prompt | copy fraction (4-gram) | ngram acceptance | speed-up |
|---|---|---|---|
| C1..C4, F1..F3 | | | |

---

## S3. Skeleton-of-thought for ONE reply: speed-up vs sections, section length and outline quality

**Question / generalization claim:** can parallel section expansion speed up one reply at acceptable quality, and how does the speed-up scale with the number of sections, section length and outline quality? Claims: (i) speed-up is bounded by the critical path, `ideal = reply tokens / (outline tokens + max section tokens)`, a hardware-independent number computed from token counts; (ii) real speed-up follows `ideal` until the number of concurrent sections exceeds the machine's batch-scaling knee (hardware-dependent: measured here at np ~4 for this model, with per-stream speed falling as np grows), after which more sections only help aggregate throughput, not the single reply; (iii) quality depends on outline distinctness and on giving each section context, not on np. Baseline measured so far: naive SoT 1.14-1.25x per word at np=2/4 with clearly worse text. [hypotheses]

**Method (always with a fixed set of >= 5 topics, e.g. electric cars, how vaccines work, how a bill becomes law, photosynthesis, how internet packets are routed, plus one code-explanation topic; temperature 0):**
1. **Outline quality:** compare (a) the naive outline prompt, (b) "exactly N distinct points, 3-6 word title and one-line scope, no overlap" (optionally with a few-shot example), (c) (b) plus automatic re-roll when pairwise word-set Jaccard overlap between points exceeds a threshold. Record the outline overlap score.
2. **Section context and structure:** each section prompt gets the full outline and its neighbours' titles ("do not repeat other points"); with/without an intro and a conclusion step (intro after the outline; conclusion after sections or in parallel from the outline).
3. **Shared prefix:** common prefix = instruction + topic + outline, section k request appended; test `cache_prompt=true` with `-np` slots (prefix prompt-eval saving; check slot assignment / `--cache-reuse`).
4. **Scaling sweeps:** number of sections N in {2, 3, 4, 6, 8, 12} at fixed total target length; section length in {50, 100, 200, 400 tokens} at fixed N; `-np` in {1, 2, 3, 4, 6, 8, 16} for fixed N=8 (the parallelism-vs-throughput curve). Express every point as a **ratio to the sequential (np=1) reply of the same words**, and next to `ideal` from token counts.
5. **Aggregate-vs-single curve:** from the same runs plot aggregate tok/s / sequential tok/s and per-stream tok/s / sequential per-stream tok/s vs np, to show the trade-off between aggregate throughput and the latency of one reply (we already measured aggregate plateau 270-300 and per-stream ~25 at np=16 for independent tasks).
6. **Quality rubric (score each reply 0-2, blind to the method; ideally by the user or a stronger model):** coverage of main aspects, repetition across sections, factual errors, coherence/transitions; plus automatic distinct-4gram ratio, repeated-sentence count, word count. Baselines per topic: sequential single reply of equal length, and naive SoT.

**Metrics:** words-normalised speed-up vs sequential, `ideal` bound and realised/ideal fraction, rubric scores and repetition metrics per topic. **Expected [estimate]:** with a distinct outline and context, 1.3-1.8x per word at N=4-6, np=3-4 on this machine; the realised/ideal fraction falls as np exceeds the batch knee; outline overlap explains most of the quality loss; adding an intro/conclusion costs ~10-20% of the speed-up. **Generalization claim supported if:** realised speed-up tracks `ideal` up to the knee for all 5+ topics and the quality-vs-outline-overlap relationship is consistent across topics. **Success criterion:** >= 1.4x words-normalised speed-up AND mean rubric within 10% of sequential on every topic AND no topic with a worse repetition score; otherwise report that SoT with this model does not pay and why. **Status:** planned.

Results to fill in:

| Variant | N sections | section tokens | np | topic | total wall s | words | ideal speed-up (from tokens) | realised speed-up vs sequential | realised/ideal | outline overlap | coverage | repetition | factual | coherence | distinct-4gram | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sequential | - | - | 1 | T1..T5 | | | 1.00 | 1.00x | | | | | | | | |
| naive SoT | 6 | ~100 | 4 | T1..T5 | | | | | | | | | | | | |
| distinct outline | 6 | ~100 | 4 | T1..T5 | | | | | | | | | | | | |
| + context | 6 | ~100 | 4 | T1..T5 | | | | | | | | | | | | |
| + intro/conclusion | 6 | ~100 | 4 | T1..T5 | | | | | | | | | | | | |
| + prefix cache | 6 | ~100 | 4 | T1..T5 | | | | | | | | | | | | |

N and section-length sweeps:

| N | section tokens | np | ideal | realised | realised/ideal | rubric mean |
|---|---|---|---|---|---|---|
| 2, 3, 4, 6, 8, 12 | fixed | = N or 4 | | | | |
| 6 | 50, 100, 200, 400 | 4 | | | | |

np sweep (N = 8):

| np | aggregate tok/s / sequential | per-stream tok/s / sequential | single-reply speed-up | Notes |
|---|---|---|---|---|
| 1, 2, 3, 4, 6, 8, 16 | | | | |

---

## S4. Other algorithmic parallel-for-one-reply ideas (low priority)

**Question / generalization claim:** does any other algorithm turn spare parallelism into single-stream speed for a tiny model? Claim to test: methods that need spare compute per step (lookahead/Jacobi) or a much cheaper drafter pay off only when the target is large relative to the overhead; for sub-1B models they are neutral or negative. Caveats kept from before. [hypothesis]

**Sub-experiments:**
- **S4a - Lookahead / Jacobi decoding:** `llama-lookahead` is **not** in the b11884 `llama-cpp\` folder. Check `llama-cli --help` for a lookahead mode; otherwise build llama.cpp from source with CUDA (CUDA toolkit + MSVC) only if S2 results suggest headroom. Sweep window W, ngram N, guess G (`-W`/`-N`/`-G`) and compare with plain on F1-F3 and C1-C4; metric = tokens per step and speed-up ratio; repeat on 0.8B and 2B/4B to get the size trend. **Expected [estimate]:** 0.9-1.3x, trending up with model size.
- **S4b - Separate small draft model (`--spec-type draft-simple -md <small.gguf>`):** requires a compatible vocabulary. Qwen3.5 uses a ~248k-token vocabulary (Qwen3.5-2B is listed with 248,320), unlike Qwen2.5/Qwen3 (~152k) or SmolLM2-135M/360M (~49k), so a 135M-360M drafter with a compatible vocab **probably does not exist**; Qwen3.5-0.8B may be the smallest of its family. Check Hugging Face first. If a compatible drafter exists, measure acceptance and the speed-up for **0.8B/2B/4B targets**: the interesting generalization is how the drafter-size/target-size ratio sets the gain (reported on Gemma 4: 1.4-1.7x with a 59 MB drafter). If not, test the family pair 0.8B draft -> 4B target (same vocabulary) as the size-ratio study. **Expected [estimate]:** 0.8B->0.8B-class target: <= 1.0x; 0.8B->4B: 1.3-1.8x.
- **S4c - Parallel sampling / best-of-n with early cut:** n candidates in n slots, cut at the first divergence checkpoint using a cheap scorer (mean logprob, repetition check), continue the best. This improves quality at about equal latency; it does not speed up one reply (per-stream speed drops with n, measured). Use as a quality tool: could a lower-bit/smaller model with best-of-n reach the quality of a larger one at the same latency? **Expected speed [estimate]:** 0.5-0.8x of plain per stream; quality gain per n to be measured with the S3 rubric.
- **S4d - EAGLE3 / DFlash-style drafters:** only check whether a matching drafter checkpoint exists for the Qwen3.5 sizes (probably not).

**Metrics:** tokens per step, speed-up ratio, rubric (S4c). **Success criterion:** >= 1.15x on free text with no quality change for any size, else documented as negative result with the size trend. **Status:** planned.

| Sub-experiment | target size | task type | tokens/step | speed-up vs plain | quality notes | blocker / availability |
|---|---|---|---|---|---|---|
| S4a lookahead | 0.8B / 2B / 4B | free / copy | | | | binary missing in b11884 zip |
| S4b draft-simple, compatible drafter | 0.8B / 2B / 4B | free / copy | | | | compatible-vocab model found? |
| S4c best-of-n + early cut | 0.8B | free | n/a | | | |
| S4d eagle3 / dflash | | | | | | drafter exists? |

---

## Priority and expected gains

| Priority | Study | Why in this order | Expected result [estimate] |
|---|---|---|---|
| 1 | S1 size x bits scaling + KLD curve | Cheap (llama-bench and llama-perplexity only), gives the size axis that S2 needs, yields the most portable fit (`a + bytes/BW`) | Q4_K_M vs Q8_0 ~1.3-1.5x at 0.8B, larger at 4B; < 10% more below Q4 at 0.8B with several-fold KLD |
| 2 | S2a/S2b n-gram and MTP sweeps (0.8B) | Biggest plausible algorithmic gain; acceptance/tokens-per-step are hardware-independent | n-gram 1.0x free text, 1.3-1.7x copy-heavy; MTP n-max 1: 0.95-1.2x |
| 3 | S2c speculative vs model size | Turns S2 into a scaling law (needs S1 model files) | MTP gain rises from ~1.1x (0.8B) to ~1.2-1.4x (4B) |
| 4 | S3 improved skeleton-of-thought | The only parallel idea that speeds one reply; needs prompt work and a rubric | 1.3-1.8x per word at N=4-6 if quality holds (naive measured: 1.14-1.25x) |
| 5 | S4a/S4b lookahead and small drafter | Possibly infeasible (no binary, vocab mismatch); do only if S2 shows headroom | 0.9-1.3x (lookahead); 0.8B->4B draft 1.3-1.8x if the pair is usable |
| 6 | S4c best-of-n | Quality tool, not a speed-up | 0.5-0.8x speed |

Run S1 first (it produces the model files and the size axis), then S2, then S3; S4 only if there is time or S2 shows headroom.

## What could realistically get closest to 300 tok/s (single reply, this laptop) [estimates]

1. n-gram speculation on copy-heavy output (up to ~1.7x of 139 = ~240 tok/s).
2. MTP n-max 1 on predictable text (up to ~1.15-1.2x), stackable with (1) if the combination is supported: best case ~2x = ~280-300 on a copy-heavy task. This is an upper bound for special workloads, not the free-text expectation.
3. Skeleton-of-thought at 1.3-1.8x per word, but it changes the text, so it counts only if the rubric passes.
4. Quantising lower than Q4_K_M: <= ~10% for 0.8B, with a quality cost.
5. Switching to a smaller model is a different model, not a speed-up of this one; S1 will show how fast that gets in general.

Free-text single-reply speed above ~200 tok/s on this machine is improbable under any of these; we will report free-text and copy-heavy results separately and never report parallel aggregate throughput as single-stream speed.

## After the studies

1. Fill the tables, write `results/<date>-<study>.md` per study (same style as existing results), update FINDINGS.md section 8 ("what is still unknown") and the README results table.
2. Summarise each generalization claim as supported / not supported / inconclusive with the evidence and the hardware it was measured on, stating which parts are ratios or acceptance statistics that should transfer and which (batch-scaling knee, `r`) are machine-specific.

## Out of scope (hardware-specific; dropped on purpose)

These were in the first version of this plan. They tune this laptop rather than reveal general scaling behaviour, so they are **not planned**. They may be revisited only if the user asks.

- GPU power/clock hygiene: Windows power mode (Balanced vs Best performance), vendor turbo mode, GPU clock lock (`nvidia-smi -lgc`), warm-up and thermal variants. Note: the 114 vs 139 tok/s discrepancy is treated as measurement noise (FINDINGS.md section 4); the methodology above (warm-up discarded, 5 runs, mean and SD, ratios vs a same-session plain control) handles it without tuning.
- CUDA graphs on/off (`GGML_CUDA_DISABLE_GRAPHS=1`).
- Runtime micro-settings: `-c` / n_ctx, `--no-warmup`, `--no-mmap`, `-ub` / `-b`, `-t` / `--threads-batch` / `--poll`, KV-cache quantisation (`-ctk/-ctv`), `-fa` on/off, and a Vulkan backend comparison. Existing evidence says these are ~0% on decode with a fully offloaded model (FINDINGS.md section 5.2).
- Any experiment whose result is only an absolute tok/s on this machine.