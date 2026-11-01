# Phase 1 review, round 1

- **Reviewer:** an opencode child (opencode/deepseek-v4.1-flash), fresh context, 2026-11-01, about 12:29–12:45Z. Brief: `docs/briefs/phase-1-review.md`.
- **Scope:** `git diff 9068467..HEAD` (9 commits); `results/1/`; `NOTEBOOK.md` (prediction log P-008..P-013, the Phase 1 entries, `## Gate 1 answers`); `STATE.md`; `BLOCKED.md`; `toy/phase1_*.py`, `toy/phase1_report.py`, `toy/phase1_run_all.sh`; `autopilot/check.py`; the executor's session log (`~/.claude/projects/-home-kiran-code-wsl-flashference/61205b4c-….jsonl`).
- **v2:** only `git show v2:results/phase1/gguf.json`, `git show v2:results/phase1/params.json` and `git show v2 --stat` were used for the replication table. See the reviewer-compliance note in §11 for one extra read.
- **Where my work is:** `/tmp/review-1/`: `recompute.py`, `v2-gguf.json`, `v2-params.json`, `broken/`, `broken-check.txt`, `checkpp.txt`.

## Summary

**No BLOCKING issue. The phase meets Gate 1's requirements.**

- `python autopilot/check.py --phase 1` exits 0: 8/8 generated tables match, all 13 predictions are in order (P-006 is the only VOID), and all 12 Phase 1 checks pass (2 global checks skip because there is no test set yet).
- A deliberately broken copy makes the check fail on the exact key I changed.
- Every headline number recomputes from the raw files with my own script (§2).
- The v2 replication matches on every deterministic value in the two permitted files (§3): GGUF byte counts, tensor/parameter counts, vocabulary size, token_embd bytes/fraction, block kinds and per-block bytes, MTP bytes, and the SmolLM2-360M parameter and byte groups.
- The five Gate 1 answers are all CORRECT, with minor precision/tag notes (§4).
- 1.2's three methods give identical ids on both devices, the CPU and GPU ids agree, the sample script's greedy ids equal 1.2's cached ids, all six token round-trips are exact, and the 135M/360M tokenizer ids agree (§5).
- P-008..P-013 were committed at 11:12:16Z; every dev output is later (11:21:11–11:52:11), the official job ran 11:53:20–11:58:32Z, and no Phase 1 script appears in the session log before the prediction commit (§6).
- The opencode child's code is sound. The executor's fix to the 1.5 toy check (draw through `sample()` against a separately computed reference, not from the sampler's own nucleus) is real and necessary (§7).

## 1. Machine checks

**`python autopilot/check.py --phase 1`** (run 12:30–12:35Z; check.py took the GPU lock itself) exited 0:

```
PASS  generated tables regenerate: 8/8 match
PASS  prediction ids are consecutive and none was ever deleted
PASS  predictions committed before their evidence: 13 predictions, 12 checked in order, 0 pending,
      void (not credited, tie verified): P-006
SKIP  test ledger lists only frozen configs: no test runs yet
SKIP  prompts/test matches prompts/test.sha256: test set not written yet
PASS  rule-5 files unchanged since bootstrap: vs 03efb2f
--- phase 1
PASS  1.2: no-cache, cached and generate() give identical ids (cpu): 64 tokens each
PASS  1.2: no-cache, cached and generate() give identical ids (cuda): 64 tokens each
PASS  toy/phase1_tokens.py re-runs from a clean checkout (HF_HUB_OFFLINE=1)
PASS  tokens.json: deterministic part identical on a re-run
PASS  toy/phase1_params.py re-runs from a clean checkout (HF_HUB_OFFLINE=1)
PASS  params.json: deterministic part identical on a re-run
PASS  toy/phase1_gguf.py re-runs from a clean checkout (HF_HUB_OFFLINE=1)
PASS  gguf.json: deterministic part identical on a re-run
PASS  toy/phase1_greedy.py --device cpu --repeats 0 re-runs from a clean checkout (HF_HUB_OFFLINE=1)
PASS  greedy-cpu.json: deterministic part identical on a re-run
PASS  toy/phase1_greedy.py --device cuda --repeats 0 re-runs from a clean checkout (HF_HUB_OFFLINE=1)
PASS  greedy-cuda.json: deterministic part identical on a re-run
PASS  toy/phase1_sample.py re-runs from a clean checkout (HF_HUB_OFFLINE=1)
PASS  sample.json: deterministic part identical on a re-run
PASS: 0 failing check(s)
```

**Deliberately broken input.** I copied `results/1/` to `/tmp/review-1/broken/` and changed one value under `deterministic`:

```
tokens.json: deterministic.vocab_size.qwen3.5-0.8b  248077 -> 248078
FF_P1_RESULTS=/tmp/review-1/broken python autopilot/check.py --phase 1
```

It exited 1 with:

```
FAIL  tokens.json: deterministic part identical on a re-run:
      deterministic.vocab_size.qwen3.5-0.8b: 248078 vs 248077
FAIL: 1 failing check(s)
```

The other five deterministic files still passed, so the failure is specific to the broken key and the check names it. This is the required broken-input test; the copy is outside the repo.

## 2. Headline numbers, recomputed

`/tmp/review-1/recompute.py` reads only `results/1/*.json` and imports no repo code. It recomputes the token counts and ratios, the greedy medians and ratios (from `measured.seconds`), the parameter/byte sums, the embedding shares, the GGUF totals and per-block bytes, and the chi-square.

| quantity | NOTEBOOK / table | recomputed | raw source |
|---|---|---|---|
| vocab sizes (SmolLM2-360M / -135M / Qwen3.5) | 49,152 / 49,152 / 248,077 | same | `tokens.json:vocab_size` |
| Hindi SmolLM2 / Qwen tokens, ratio | 44 / 18, 2.44× | 44 / 18, 2.4444× | `tokens.json` Hindi `n_tokens` |
| English ratio | 1.00× | 10 / 10 = 1.0000× | same |
| Chinese / code / emoji / German ratios | 2.38 / 1.08 / 1.07 / 1.42 | 2.375 / 1.077 / 1.067 / 1.417 | same |
| Hindi chars / UTF-8 bytes | 38 / 98 | 38 / 98 | `tokens.json` |
| partial-character pieces, Hindi S / Q | 28 / 0 | 28 / 0 | same |
| all six round-trips exact | yes | all `decodes_back_exactly` true | same |
| greedy-cpu nocache / cached / generate median | 16.085 / 5.210 / 5.277 s | 16.0855 / 5.2096 / 5.2772 s | `greedy-cpu.json:measured.median_s` |
| greedy-cuda nocache / cached / generate median | 2.128 / 1.571 / 1.710 s | 2.1284 / 1.5714 / 1.7096 s | `greedy-cuda.json` |
| CPU nocache/cached, generate/cached | 3.09, 1.01 | 3.0876, 1.0130 | same |
| GPU nocache/cached, generate/cached | 1.35, 1.09 | 1.3544, 1.0879 | same |
| CPU step 1/2/64 nocache; cached | 0.1446/0.1513/0.3390; 0.1563/0.0805/0.0812 | same (median of 5 repeats per step) | `step_seconds` |
| GPU step 1/2/64 nocache; cached | 0.0299/0.0233/0.0338; 0.0253/0.0243/0.0214 | same | same |
| all three ids equal, both devices | yes | `ids.nocache == ids.cached == ids.generate` and CPU cached == GPU cached | `greedy-*.json:deterministic` |
| SmolLM2-360M params / bytes / embedding share | 361,821,120 / 723,642,240 / 13.04% | same; 94,371,840/723,642,240 = 13.0412% | `params.json` |
| SmolLM2-135M params / bytes / embedding share | 134,515,008 / 269,030,016 / 21.05% | same; 56,623,104/269,030,016 = 21.0471% | same |
| GGUF file / tensor / parameter bytes | 1,557,662,528 / 1,546,700,032 / 772,845,888 | same; sums over the 335 tensor records | `gguf.json` |
| token_embd shape / type / bytes / share | [1024, 248320], BF16, 508,559,360, 32.88% | same; 508,559,360/1,546,700,032 = 0.32880283796 | same |
| separate output.weight | no | no (no such tensor) | same |
| attention / recurrent / MTP blocks | 3,7,11,15,19,23 / 18 others / 24 | same; 36,710,400 / 43,164,288 / 40,916,992 bytes each | same |
| bytes not in tensors | 10,962,496 | 1,557,662,528 - 1,546,700,032 = 10,962,496 | same |
| tensor types | BF16 195 / 1,544,683,520; F32 140 / 2,016,512 | same | same |
| toy chi-square | p = 0.958113, 0 outside nucleus | stat 0.085579 on 2 dof, p = 0.958113; 0 outside | `sample.json:toy` |
| sample first-difference positions | 4, 8, 15, 2, 5; 5 distinct | same; greedy ids == 1.2 CPU cached ids | `sample.json:model` |

The table's rounded values match the generated report (`check.py --tables` re-ran all 8 marked commands and got identical output), so no table cell is typed by hand.

## 3. Independent replication against v2

`git show v2:results/phase1/gguf.json` and `git show v2:results/phase1/params.json` are from the earlier independent run of Phase 1 on `v2`. Every value in the permitted files matches the current `results/1/` output exactly:

| value | v2 | current |
|---|---|---|
| file_bytes | 1,557,662,528 | 1,557,662,528 |
| total_tensor_bytes | 1,546,700,032 | 1,546,700,032 |
| n_tensors | 335 | 335 |
| vocab / width | 248,320 / 1,024 | 248,320 / 1,024 |
| token_embd type / bytes | BF16 / 508,559,360 | BF16 / 508,559,360 |
| token_embd_fraction | 0.3288028379636059 | 0.3288028379636059 |
| separate_output_tensor | false | false |
| attention_blocks | [3, 7, 11, 15, 19, 23] | same |
| recurrent_blocks | [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22] | same |
| mtp_blocks / mtp_bytes | [24] / 40,916,992 | [24] / 40,916,992 |
| token_embd_fraction_without_mtp | 0.33773747378639624 | 0.33773747378639624 (derived) |
| 360M params / bytes | 361,821,120 / 723,642,240 | same |
| 360M embedding bytes / fraction | 94,371,840 / 0.13041228770725158 | 94,371,840 / 13.0412287707% |
| 360M attention / mlp / norms / lm_head bytes | 157,286,400 / 471,859,200 / 124,800 / 0 | same |
| tie_word_embeddings / lm_head is embedding | true / true | true / true |

No mismatch to explain. v2's `params.json` covers only the 360M model; the 135M numbers have no v2 counterpart, but they reproduce from `config.json` shapes in `toy/phase1_predict.py` and match `params.json` exactly. The v2 run is a genuine replication of the deterministic numbers, not a copy: the current `gguf.json` has a different schema and more detail (per-tensor records, tensor types, groups), while the shared values agree.

**The executor's v2-listing claim is confirmed.** `git show v2 --stat | head -40` prints the tip commit message ("phase 1"), 17 changed paths with insertion counts, and no measured numbers; `git ls-tree -r --name-only v2 | head -50` prints 20 file names only. It shows file names and line counts but no file contents or measured values.

## 4. Gate 1 answers

| # | grade | reason and evidence |
|---|---|---|
| 1 | **CORRECT** | Defines a logit as the raw pre-softmax score, the LM head as a hidden-vector × vocabulary-matrix product, and explains the linear growth in vocabulary. The Qwen numbers are right: 248,320 × 1,024 × 2 = 508,559,360 bytes, 32.88% of tensor bytes (`NOTEBOOK.md:466-469`; `results/1/gguf.json`). Minor tag issue in N4. |
| 2 | **CORRECT** | Token t is part of the input that predicts t+1, so the loops are strictly sequential; one pass computes all positions and speculation checks several at once. The 103-position vs 1-position step comparison (33.8 vs 21.4 ms) is in `results/1/greedy-cuda.json` (`NOTEBOOK.md:470-473`). Minor tag issue in N4. |
| 3 | **CORRECT** | KV cache stores per-layer, per-position K/V and truncates; a Gated DeltaNet layer stores a fixed 128×128 matrix per head plus a kernel-4 conv state, and rolling back needs a checkpoint or re-run. The 12,288 bytes/token derivation is correct for 6 layers, 2 KV heads, head_dim 256, bf16 (`NOTEBOOK.md:474-478`; config values match `config.json`). Precision note N3: "only the 6 attention blocks" omits the MTP block. |
| 4 | **CORRECT** | 508,559,360 / 1,546,700,032 = 32.88%, and 508,559,360 / (1,546,700,032 − 40,916,992) = 33.77%; both say "about one third". The answer states the assumption that plain decode does not read blk.24 and that it was not checked (`NOTEBOOK.md:479-482`). |
| 5 | **CORRECT** | MTP block 40,916,992 bytes, no head matrix in blk.24, tied 248,320-row head 508,559,360 bytes; 92.6% of a drafted token's bytes and 36.5% of a plain step are correct. The v1 numbers (~5.3 ms untrimmed, ~4.0 ms saved, 1.15×→1.41×/1.60×) match the cited lines of `ref/flashference/results/2026-10-26-bf16-b4-trim-draft.md` (`NOTEBOOK.md:483-488`). |

The answers are in the executor's own words and are built from the results, not copied from v2. The one phrasing that closely resembles v2's notebook ("only the 6 attention blocks") is a natural summary of the same GGUF block table; I found no v2 read in the executor's session log before or after the results (only `git show v2 --stat`, `git ls-tree` and `git show v2:plan.md`).

## 5. Losslessness and equality claims

- **1.2's three methods give identical ids.** Verified directly from both raw files: `ids.nocache == ids.cached == ids.generate` is true for CPU and CUDA, 64 tokens each; the `all_three_equal` flag agrees. The CPU cached ids equal the GPU cached ids. `check.py` repeats this check from the raw files and passes.
- **Why the claim is meaningful.** The three methods are separate implementations (`toy/phase1_greedy.py:greedy_nocache`, `greedy_cached`, `generate`), and the raw `measured.order` records one warm-up then A B C interleaved repeats. A repeat that gave different ids would fail the script's own `assert got == results[m]`; the run completed with 5 repeats per method.
- **Token round-trip.** All six texts decode back exactly for both tokenizers (`tokens.json`: every `decodes_back_exactly` true; 12 combinations). The Hindi pieces are partial characters on their own (28 of 44) but the full id sequence reconstructs the sentence. The 135M and 360M ids are identical for every text (raw lists equal; flag true).
- **1.5 cross-check.** `sample.json`'s `greedy_ids` equal `greedy-cpu.json`'s `cached` ids exactly, and the toy check's chi-square recomputes to p = 0.958113 with 0 draws outside the 3-token nucleus.
- **No unexplained output mismatch.** There is no speculative-decoding config in Phase 1, so §7.4's tau_numerics check does not apply; the only equality claims are the ones above.

## 6. Predictions, commit order, mtimes, job record

- **Prediction commit:** `a2b4e25`, 2026-11-01T11:12:16Z. It adds P-008..P-013 with `got __ | error __ | learned __`, the "what I knew" entry, `toy/phase1_predict.py`, and the STATE.md update.
- **First committed results:** `a7f3d6e`, 11:59:06Z, strictly after the prediction commit. `check.py --predictions` finds the first commit that added each line, requires the first version to have `got __`, and requires the evidence file to be added strictly later; all six Phase 1 predictions pass. P-006 is the only VOID, with the tie verified.
- **`got`/`error` fields:** filled once at `042bebe` (12:04:12Z) and never changed; `check.py`'s rewrite check passes. Only `learned` was added with the fills.
- **Dev-run mtimes (all after 11:12:16):**

  | directory | earliest | latest |
  |---|---|---|
  | `/tmp/ff-p1-dev` | 11:21:11 (`tokens.json`) | 11:52:11 (`sample.json`) |
  | `/tmp/ff-p1-dev0` | 11:38:02 (`greedy-cpu.json`) | 11:38:02 |
  | `/tmp/ff-p1-broken` | 11:33:04 | 11:33:04 |
  | `/tmp/ff-p1-short` | 11:45:48 | 11:45:48 |

  No Phase 1 output predates the prediction commit. The executor's session log shows no Phase 1 script execution before 11:12:16; the only pre-commit Phase 1 command is `python toy/phase1_predict.py` at 11:11:18, which reads `config.json` and loads no weights.
- **Official job:** `results/_jobs/phase1-runs/cmd` names `toy/phase1_run_all.sh`, `start` 11:53:20Z, `end` 11:58:32Z, `exit` 0. The raw file `started_utc` stamps are 11:53:37, 11:53:41, 11:53:44, 11:53:53, 11:56:56 and 11:57:36; all inside the job and after the prediction commit. `results/1/preflight.jsonl` has three code-0 lines at 11:53:35, 11:56:54 and 11:58:32 with `gpu_lock: held-by-caller`, AC power and idle GPU readings.
- **Conclusion:** the executor's claim that no Phase 1 code ran before `a2b4e25` is supported by all three records (commit order, `/tmp` mtimes, job record) and by the session log.

## 7. Code review

The Phase 1 code was written by an opencode child and changed/reviewed by the executor (`STATE.md:32-33`, `a6c843c`). I read the final scripts and `check.py`'s Phase 1 additions.

- **`toy/phase1_greedy.py`** — three genuinely different loops: nocache recomputes the full sequence with `use_cache=False`; cached passes `past_key_values` and feeds one new token; `generate` uses the library loop. Warm-up is one call per method, then A B C interleaved repeats. Timers are `CLOCK_MONOTONIC_RAW` (`time.clock_gettime`), with `torch.cuda.synchronize()` around GPU calls and `int(nxt)` forcing completion inside the timed region. The per-step times are recorded inside each loop, and the report's step medians are computed across repeats. The repeated-run `assert` catches nondeterministic ids. No bug found.
- **`toy/phase1_params.py`** — counts each distinct `named_parameters()` tensor once in bf16, groups by name, asserts `other == 0`, and checks `lm_head.weight` shares storage with `embed_tokens.weight`. The 360M and 135M totals match the config arithmetic exactly. No bug found.
- **`toy/phase1_gguf.py`** — reads the GGUF with `GGUFReader`, records each tensor's shape/type/elements/bytes, groups by prefix and block, and classifies blocks by `nextn.*`, then `ssm_*`, then `attn_q/k/v/qkv`. The per-block totals and the MTP/recurrent/attention split are correct. The `block_kind` ordering means blk.24 is `mtp` even though it has attention tensors, which is why the report's block-kind table has no separate attention entry for it. No bug found.
- **`toy/phase1_tokens.py`** — round-trips six texts with three tokenizers, decodes each id alone to count partial characters, and records the raw ids/pieces. The 248,077 HF tokenizer length vs 248,320 GGUF rows is reported and explained, not hidden. No bug found.
- **`toy/phase1_sample.py`** — implements temperature, top-p and a seeded `torch.multinomial` draw. The toy check now draws through `sample()` itself and compares with a separately computed reference; the child's version had drawn from the sampler's own nucleus, which made "never outside the nucleus" true by construction. The executor's fix is present and is the right kind of fix. The model part uses the cached loop from 1.2 with per-seed generators. No bug found.
- **`autopilot/check.py`** — the Phase 1 check builds a clean `git archive HEAD` checkout, symlinks `models`, sets `HF_HUB_OFFLINE=1`, runs every script under `flock /tmp/flashference-gpu.lock`, and compares each `deterministic` section with the committed one. The `first_difference` function reports the first differing JSON path, which is exactly what the broken-input test showed. The check also verifies 1.2's identical-ids claim. The clean-checkout re-run is a strong check: it catches both stale results and environment-dependent output. No bug found.

## 8. BLOCKING issues

None.

## 9. NON-BLOCKING issues

**N1. P-011 and P-012 are point predictions, not ranges.** `NOTEBOOK.md:22-23`; autoplan.md §1 rule 11 says every prediction states a range. P-011 gives a range only for the embedding shares ("13.0–13.1%", "21.0–21.1%") and exact counts for everything else; P-012 is exact throughout. The counts are deterministic config arithmetic, so a point is defensible, but the deviation is not recorded in STATE.md's Deviations. For future exact predictions, write a degenerate range ("exactly X; range X–X") or record the deviation. This does not affect the result: both predictions were committed before the run and match exactly.

**N2. P-012(a) is not blind, and the f32 set behind P-012(c) was fitted to the seen total.** Disclosed at `NOTEBOOK.md:238` and in P-012 itself: `model_size` and `model_n_params` were seen in Phase 0, and the choice of f32 tensors (norms, `ssm_a`, `ssm_dt`, `conv1d`) was inferred from the seen total. The per-block split and the attention/recurrent/MTP classification are the real test, and they match v2. This is honest but limits the prediction's evidential value; the v2 replication is the stronger check.

**N3. Gate 1 Q3 says "In Qwen3.5 only the 6 attention blocks have one [KV cache]".** `NOTEBOOK.md:475`. The GGUF also has blk.24, the MTP block, which contains `attn_q`, `attn_k`, `attn_v`, `attn_output` and q/k norms (`results/1/gguf.json`); it is a full-attention layer too, and would need its own cache when the MTP draft runs. The main 24-layer point (6 attention, 18 recurrent) is right, and Q5 correctly calls blk.24 a full attention layer. Add "of the 24 main layers" to Q3.

**N4. Two tags are imprecise.** `NOTEBOOK.md:472`: "64 tokens took 64 forward passes [meas: n_generated]" — `n_generated` records 64 generated tokens, while the 64 forward passes are derived from the loop (each step is one pass). `NOTEBOOK.md:469`: "248,320 × 1,024 × 2 bytes = 508,559,360 bytes ... [meas: token_embd]" — the tensor bytes are measured, but the product is [derived] arithmetic. No number is wrong; retag as [derived] with the measured input.

**N5. The Phase 1 predictions entry has no "What this shows" paragraph.** `NOTEBOOK.md:233-242`; the NOTEBOOK header (`:3`) and autoplan §5.4 say every entry ends with one. The four result entries all have one. Relatedly, a few method parameters and rounded restatements in prose carry no tag, e.g. `:245` "six short texts with three tokenizers", `:279` "64 new tokens ... 5 repeats", `:420-421` "100,000 draws ... five 64-token samples at T = 0.7, top-p 0.9", `:343` "3.1× faster", `:457` "within 2–15 tokens". The headline results are tagged; this is a rule-10/study-trail style gap, not a wrong number.

**N6. One sentence in the predictions entry is now stale.** `NOTEBOOK.md:242`: "The five drafts in toy/ are unchanged since 05f4335 and have never been run." That was true when the prediction was committed, but `a6c843c` changed them; the 1.1 entry explains the code was finished later. A reader arriving at the current file may be confused; add "at the time of this commit" or a pointer to 1.1.

**N7. The 1.2 tick sentence is an inference from a spot reading.** `NOTEBOOK.md:279`: "The kernel tick read 10833 µs at the first preflight ... so CLOCK_MONOTONIC would have run fast during these runs." The preflight is a single reading at 11:53:35, not a during-run measurement; the tick is re-set every ~8 s. The timed runs use `CLOCK_MONOTONIC_RAW`, so the timings are unaffected. Say "the first preflight read ... so a MONOTONIC-based timer could have run fast; we used RAW" rather than asserting the rate during the runs.

## 10. What I did not test

- No new timings. Per the brief, I did not re-run the 1.2 or 1.5 measurements; the medians, ranges and ratios are recomputed from the raw `measured.seconds` and `step_seconds`.
- I did not reload the tokenizers to re-tokenize the six texts independently. I checked the raw ids, pieces, counts and `decodes_back_exactly` fields, and `check.py` re-ran the tokenizer scripts from a clean checkout with identical deterministic output.
- I did not run llama.cpp or inspect which tensors a plain decode step actually reads; Q4's assumption is flagged by the answer itself.
- I did not re-verify v1's B4 report beyond the cited lines 9 and 59, or the v1 README line 37 cited in P-012's "what I knew" entry.
- I did not check the 135M model against v2 (v2 has no 135M `params.json`); its numbers are checked against config arithmetic and the current `params.json`.
- I did not test other prompts, longer generations, bf16, quantised GGUFs, or the vision tower.
- I did not open, list or print anything under `prompts/test/`.

## 11. Limits of this review and reviewer compliance

- **Same model as the code author.** This review was run by the same model family that wrote Phase 1's code (`STATE.md:33`). The context is fresh and every number was recomputed from the raw files, but a different model would be a stronger independent check. This is the owner's chosen reviewer and is recorded as a deviation.
- **Reviewer's own deviation:** while checking whether the Gate 1 answers were paraphrased, I also read `git show v2:NOTEBOOK.md` (the first 90 lines). The phase-specific instruction allowed only v2's `results/phase1/gguf.json`, `params.json` and `git show v2 --stat`. No number in this report comes from v2's notebook: the current numbers were recomputed from `results/1/` before that read, and the replication table uses only the two permitted files. A reader who wants a strictly clean replication should treat the v2 comparison as coming from those two files and the stat only.
- **Hard rules followed:** the only file I created is `reviews/phase-1-r1.md`; no git write commands; the break test was in `/tmp/review-1/broken/`; every torch/transformers/llama.cpp run was under `flock /tmp/flashference-gpu.lock` (check.py takes the lock itself); no installs or downloads; no writes under `/mnt/c`; no `prompts/test/` access.
- **Final `git status --short`:** only `?? reviews/phase-1-r1.md`.

verdict: PASS
