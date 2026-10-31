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

---

## 2026-10-30: Phase 1 measurements

Raw output is in `results/phase1/*.json`. Scripts and commands are in `toy/README.md`. Every number below comes from those files. The Gate 1 answers and the 1.2 "why is cached faster" explanation are in the next entry.

**Setup:** RTX 3050 A Laptop 4 GB, torch 2.15.0+cu132, transformers 5.22.0, gguf 0.19.0. Target model `HuggingFaceTB/SmolLM2-360M-Instruct`, bf16 on CUDA (it fit, so CPU fp32 was not needed). The 135M model was not downloaded.

**Commands run (repo root):**
```
.venv/bin/python toy/phase1_tokens.py
.venv/bin/python toy/phase1_greedy.py          # n=32, reps=9 interleaved, cuda bf16
.venv/bin/python toy/phase1_greedy.py --long   # same, with a 1240-token prompt
.venv/bin/python toy/phase1_forward.py         # one forward pass vs sequence length, reps=30
.venv/bin/python toy/phase1_params.py
.venv/bin/python toy/phase1_gguf.py
.venv/bin/python toy/phase1_sample.py          # run 3 times; sample.json byte-identical across runs
```

**1.1 tokens (SmolLM2 tokenizer, vocab 49152):** English 44 chars → 10 tokens. Code 38 → 15. Emoji 29 → 12 (6 ids decode to a partial character alone). Hindi 25 chars / 67 UTF-8 bytes → 30 tokens (18 ids are partial characters). All 4 round-trip exactly.

**1.2 greedy (32 new tokens, median of 9 timed runs after 1 warmup, methods interleaved; re-run by the reviewer 2026-10-30, `HF_HUB_OFFLINE=1`):**

| prompt | no-cache | cached | `generate` | ids match (all 3) |
|---|---|---|---|---|
| short, 40 tok | 867.8 ms (27.1 ms/tok) | 851.1 ms (26.6) | 963.5 ms (30.1) | yes (32/32) |
| long, 1240 tok | 2993.5 ms (93.5 ms/tok) | 1044.3 ms (32.6) | 1122.2 ms (35.1) | yes (32/32) |

Short-prompt ranges: no-cache 720–925, cached 708–1075, generate 776–1239 ms. Long: no-cache 2960–3038, cached 952–1330, generate 941–1423 ms. The first version (3 runs per method, run as separate blocks) gave short 1577 / 1713 / 1285 ms and long 3499 / 1207 / 1435 ms in an earlier session (kept as `results/phase1/greedy_cuda_bf16*_blocks3.json`). The ids were the same; the times were not. The same work took up to 1.8× longer in one session than in another.

**1.2 one forward pass vs length (`phase1_forward.py`, median of 30, ms):** 1 tok 29.0, 40 tok 30.1, 72 tok 24.7, 150 tok 26.1, 300 tok 33.5, 600 tok 46.6, 1240 tok 90.8. One cached 1-token step: 41.2 after a 40-token prompt, 35.1 after 1200 (min 20.1 / 21.2). Single calls range from about 20 to 80 ms.

**1.3 params (SmolLM2-360M, bf16):** `tie_word_embeddings=True`, and `lm_head.weight` is the same tensor as `embed_tokens.weight`. Total 361,821,120 params = 723,642,240 bytes. Embedding (= head) 94,371,840 B (13.04%), attention 157,286,400 B (21.74%), MLP 471,859,200 B (65.21%), norms 124,800 B (0.02%), lm_head 0 B counted separately.

**1.4 Qwen3.5-0.8B bf16 GGUF:** 335 tensors, 1,546,700,032 tensor bytes (file 1,557,662,528 B). `token_embd` is BF16, 508,559,360 B = 32.88% of tensor bytes, or 33.77% if the MTP block is excluded. There is no separate `output.weight`, so the head is tied. Vocab 248320, width 1024. `block_count=25`, `full_attention_interval=4`, `nextn_predict_layers=1`.
- Attention blocks (have `attn_q/k/v`): 3, 7, 11, 15, 19, 23. Each is 36,710,400 B.
- Recurrent blocks (have `ssm_*`; they also contain a tensor named `attn_qkv`): 0–2, 4–6, 8–10, 12–14, 16–18, 20–22. Each is 43,164,288 B.
- MTP block: blk.24 (attention-type layer plus `nextn.*` tensors), 40,916,992 B (2.65%).

**1.5 sampling:** seed 1234, T=0.7, top-p 0.9, 32 tokens. The sampler is written by hand on logits from the cached loop. Sample 0 matched greedy exactly (32/32). Samples 1–4 left greedy after 2, 5, 2 and 2 tokens. Texts are in `results/phase1/sample.json`.

**Not tested:** the 135M model; CPU or fp32 timings; n>32; any prompt besides the "sky is blue" one; whether the bf16 id match still holds on longer generations.

---

## 2026-10-30: Gate 1 answers (reviewer-as-human)

Numbers come from `results/phase1/*.json` unless marked [derived] (arithmetic on those numbers or on GGUF metadata, not measured).

**1.2 step 4: why the cached loop is faster, and why it wasn't on the short prompt.**
Without a cache, step i runs the whole sequence again (prompt + i tokens), so the work per step grows with length and most of it repeats the last step. With a cache, the prompt runs once (prefill) and each later step runs one token, reading the stored keys/values for the past. On the long prompt this is clear: no-cache 93.5 ms/token, cached 32.6 ms/token (2993.5 vs 1044.3 ms). That matches `phase1_forward.py`: a 1240-token forward is 90.8 ms, so 32 of them is about 2.9 s [derived], while cached is about one 91 ms prefill plus 31 steps of about 30 ms.
On the short prompt the saving is real but too small to see. A forward pass here has a floor of about 25–30 ms that hardly depends on length: 1 token 29.0 ms, 72 tokens 24.7, 150 tokens 26.1. Only from about 300 tokens (33.5) does it grow. The short no-cache loop never runs more than 72 tokens, so each of its steps costs the same as a cached 1-token step: 27.1 vs 26.6 ms/token. The floor is most likely fixed cost per forward (Python, HF, kernel launches), not arithmetic [unv, see grade]. The first measurement (3 runs per method in separate blocks, 1.1 s spread) even showed cached as slower. That was noise and drift, not a real effect.

**Q1. What is a logit? What does the LM head compute, and why does its cost grow with vocabulary size?**
A logit is one raw score per vocabulary token for "this is the next token". Softmax turns the scores into probabilities; argmax picks the biggest. The LM head is one matrix-vector product: final hidden state (width d) times a [vocab × d] matrix gives vocab logits. Every row has to be read and dotted with the hidden state, so cost is vocab × d multiply-adds and vocab × d × 2 bytes (bf16) read per token. It grows linearly with vocab. Qwen3.5-0.8B: 248320 × 1024 = 254M multiply-adds and 508,559,360 B per token [derived]. SmolLM2-360M: 49152 × 960, 94,371,840 B.

**Q2. Why can't the model produce token t+1 before token t?**
Token t is part of the input that predicts t+1. Until the forward pass for t is done and its argmax (or sample) is picked, the input for the next pass doesn't exist. So 32 tokens means 32 forward passes one after another, each paying the full per-pass cost (about 27 ms here, `greedy_cuda_bf16.json`). The forward-vs-length numbers show the opening that speculation uses: a pass over 72 tokens costs no more than a pass over 1 (24.7 vs 29.0 ms). If someone guesses the next few tokens, one pass can check all of them for about the price of one.

**Q3. What does a KV cache store? What does a recurrent layer store instead? Why is "undo 3 tokens" easy for one and hard for the other?**
A KV cache stores, for each attention layer and each past token, that token's key and value vectors. It grows by one entry per token. SmolLM2: 32 layers × 5 KV heads × 64 × 2 (K and V) = 20,480 values per token [derived from config]. Qwen3.5: only the 6 attention blocks have one: 6 × 2 × 512 = 6,144 values per token [derived from GGUF shapes].
A Gated DeltaNet layer stores a fixed-size state instead: a small matrix per head (GGUF says `ssm.state_size=128`, `ssm.group_count=16`, so about 16 × 128 × 128 values per layer [derived, not checked against llama.cpp]), plus a short conv buffer (`conv_kernel=4`). Each token updates the state in place: decay it, then write the new token into it.
Undo on a KV cache is just "forget the last 3 entries", because each token's entry is separate and nothing else changed. `phase1_forward.py` does exactly this with `past.crop(-1)` between timed steps. The recurrent state has no per-token entries. The last 3 tokens are mixed into the same matrix, the older contents were decayed and overwritten, and the update can't be run backwards exactly. To undo, you must have saved a copy of the state from before those tokens, or recompute from an earlier copy.

**Q4. What fraction of Qwen3.5-0.8B's bytes is the head? Do you agree with "about one third"?**
`token_embd` (= the head; there is no `output.weight`) is 508,559,360 B. Over all tensor bytes (1,546,700,032) that is 32.88%. Over the bytes a plain decode step uses, which leave out the MTP block (1,546,700,032 − 40,916,992 = 1,505,783,040 B), it is 33.77%. I use the second denominator because the question is about decode cost. The input-embedding lookup reads only one 2 KB row; the head reads all 508.6 MB. Yes, I agree with about one third, **of the bytes**. flashference actually wrote "~1/3 of per-token *time*" (`ref/.../2026-10-27-bf16-E1-E5.md`). That only follows if decode is limited by memory bandwidth, and Phase 1 didn't measure that. My HF numbers show a different limit: a 1-token SmolLM2 step costs about the same as a 150-token one. So for time I'd mark it [est] until Phase 7.

**Q5. Why did tied embeddings plus a 248k vocabulary make flashference's MTP draft expensive?**
The MTP block's own weights are small: 40,916,992 B (2.65%). It has no head of its own (blk.24 has `nextn.shared_head_norm` but no head matrix), so to turn its hidden state into a draft token it uses the shared 248k × 1024 `token_embd`: 508,559,360 B, 12.4× the block itself. One drafted token therefore reads about 549 MB, 36.5% of a full plain step, and 92.6% of that is the head [derived, bytes not time]. Because the head is tied to the input embedding, it has to keep every row for every token. With a big vocabulary that makes each draft cost about a third of a real step, so a draft only pays off if many tokens are accepted. That is why flashference trimmed the draft head to the most frequent rows.

## Gate 1 grade (reviewer grading the answers above, Appendix A style)

What's wrong or missing:
- Rule 1 broken: no prediction was written before the 1.2 timing. The "≈ 32 × 91 ms" check was done afterwards.
- Step 4 / Q2: "the floor is fixed per-forward cost (Python, kernel launches)" is a likely explanation, but no harness measured it. A one-off profiler run (not saved) showed about 1180 kernel launches and about 16 ms of GPU kernel time per 1-token forward. Treat that as [unv].
- Q3: the recurrent state size comes from metadata, not from llama.cpp's code. Undo was only tested on a KV cache (SmolLM2), never on a recurrent state.
- Q4: the time claim is not tested. I gave no bandwidth figure, so I can't say how far 27 ms/step is from the memory floor.
- Q5: the argument is in bytes. "MTP uses `token_embd` as its head" is an inference from a missing tensor; it should be confirmed in the llama.cpp source (Phase 5).
- Q1 leaves out that softmax over 248k entries, and the top-p sort in 1.5, also cost time. That part is small next to the matrix read, but still not zero.

Open questions to answer before Phase 2:
- Why does a 1-token forward of a 724 MB model take about 25 ms? What is this GPU's real memory bandwidth, and what floor does it give? (Phase 2.)
- Why does forward cost start to grow between 150 and 300 tokens?
- Why do single forwards vary 20–80 ms, and whole runs up to 1.8× across sessions? What noise floor does that leave for Phase 4?
- Why is `generate()` a little slower than the hand-written cached loop (30.1 vs 26.6 ms/token short; 35.1 vs 32.6 long)?
- Sample 0 at T=0.7 matched greedy on all 32 tokens. How peaked is the distribution? Print top-1 probabilities per step to see.
- Explain `keep = (cumsum − p) < top_p` in `phase1_sample.py` line by line.
- bf16: no-cache and cached gave the same ids for 32 tokens. They run different kernel shapes, so would they still match at 256 tokens? (L2 vs L3, preview of 8.7.)
- Phase 0 is not closed: the llama-bench tg128 line is still `__ tok/s`, so Gate 0 hasn't passed.
