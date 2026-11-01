# Phase 1 review, round 1 — second independent reviewer

- **Reviewer:** opencode-go/glm-5.3, fresh context, 2026-11-01, about 13:18–13:55Z. Brief: `docs/briefs/phase-1-review.md`, including the "Second reviewer (owner's rule, 2026-11-01)" section, which overrides the earlier text where they differ.
- **Independence:** my grades and verdict were fixed and written to `/tmp/review-1b/my-grades.txt` **before** I read `reviews/phase-1-r1.md` or `/tmp/review-1/`. I ignored the "Grade r1" marks in NOTEBOOK.md while grading. Section 13 lists where the first review and I differ.
- **Scope:** plan.md "Phase 1" + Gate 1 (plan.md:176–228); autoplan.md §1 (rules 10–16), §4.3–4.4, §5, §7, "Phase 1: One token" (autoplan.md:348–358); `git diff 9068467..HEAD`; the raw files under `results/1/`; NOTEBOOK.md's prediction log (P-008..P-013, :19–24), Phase 1 entries (:233–465) and `## Gate 1 answers` (:467–496, the corrected text); STATE.md; BLOCKED.md; `toy/phase1_*.py` (all seven), `autopilot/check.py`; `results/_jobs/phase1-runs/`; the executor's session log; the three HF `config.json` files; `ref/flashference/results/2026-10-26-bf16-b4-trim-draft.md` and README line 37 (both cited by Gate 1 answers).
- **v2:** only `git show v2:results/phase1/gguf.json`, `git show v2:results/phase1/params.json`, and `git show v2 --stat`. Nothing else from v2, per my rules.
- **Where my work is:** `/tmp/review-1b/`: `recompute.py` (my own script, no repo code imported), `recompute-out.txt` (213 checks), `v2-replication.txt` (83 compared values), `broken/` + `check-broken.log`, `check-phase1.log`, `my-grades.txt`.

## Summary

**No BLOCKING issue. The phase meets Gate 1's requirements, and the corrected text's corrections are themselves right.**

- `python autopilot/check.py --phase 1` exits 0 (all global and Phase 1 checks PASS); a deliberately broken copy fails on exactly the key I changed.
- Every headline number recomputes from the raw files with my own script: **213/213 match** the published tables, row by row.
- The v2 replication: **83 deterministic values compared, 82 exact matches**, and the one difference is a labeling convention (`mtp(attn)` vs `mtp` for blk.24), not a number.
- All five Gate 1 answers: **CORRECT**, each re-checked against raw data, config.json, and (for answer 5) v1's B4 report.
- 1.2's identical-ids claim (and the other equality claims) hold in the raw data.
- The predictions were committed before any run: commit order, `/tmp` mtimes, the job record, and the executor's session log all agree, with one wording nit (§7).
- The code (written by an opencode child, reviewed and changed by the executor) is sound; the executor's fix to the 1.5 toy check removed a real circularity.
- The two corrections made after review r1 (step-table medians; the 1.5 "same explanation" claim) are both right, and I confirmed the executor's critique of r1's own faulty recomputation table.

## 1. Machine checks

**`python autopilot/check.py --phase 1`** (run 13:19–13:21Z; check.py takes the GPU lock itself for its re-runs) exited 0 with the same output the first reviewer reports: 8/8 generated tables regenerate; 13 predictions, 12 checked in order, P-006 the only VOID (tie verified); test-ledger and test-prompt checks SKIP (no test set yet — correct, the test set is written in Phase 4); rule-5 files unchanged; both identical-ids checks PASS (64 tokens each); all six scripts re-run from a clean `git archive HEAD` with `HF_HUB_OFFLINE=1` and each `deterministic` section is identical to the committed one. Log: `/tmp/review-1b/check-phase1.log`.

**Deliberately broken input.** I copied `results/1/` to `/tmp/review-1b/broken/` and changed one value under `deterministic`: `tokens.json`, `vocab_size.smollm2-360m` 49,152 → 49,153. Then:

```
FF_P1_RESULTS=/tmp/review-1b/broken python autopilot/check.py --phase 1
→ FAIL  tokens.json: deterministic part identical on a re-run:
        deterministic.vocab_size.smollm2-360m: 49153 vs 49152
  FAIL: 1 failing check(s)   (exit 1)
```

The FAIL names the exact key I broke; every other check still passes (the global table check regenerates from the untouched `results/1`, the greedy identical-ids checks read unbroken copies). The check is real and specific. The break was in a copy outside the repo.

## 2. Headline numbers, recomputed with my own script

`/tmp/review-1b/recompute.py` imports no repo code and reads only `results/1/*.json`, the model file's size, and (for scipy's chi-square) scipy. 213 checks, all matching. Full output: `recompute-out.txt`. Summary of what it covers:

- **1.1:** for each of the six texts, both token counts, the S/Q ratio, chars/token for both tokenizers, partial-character pieces recounted from the raw `pieces` arrays (not the stored counters), round-trip exactness (stored flag plus, for texts with no partial pieces, re-joining the pieces' UTF-8 bytes into the text's bytes), vocabulary sizes (49,152 / 49,152 / 248,077), 135M ids == 360M ids, and Hindi's 38 chars / 98 UTF-8 bytes.
- **1.2:** for both devices, median/min/max seconds of the 5 interleaved repeats for all three methods, ms/token, nocache/cached and generate/cached ratios, prompt = 40 tokens, n_generated = 64, clock = `CLOCK_MONOTONIC_RAW`, and the full step table recomputed the way the report states it (per-step median over the 5 repeats, then median over the ranges 1, 2–11, 55–64): CPU 144.6/156.3/0.93, 160.8/80.6/1.99, 320.4/81.9/3.91 ms; CUDA 29.9/25.3/1.18, 24.3/22.1/1.10, 33.7/22.1/1.53 — all as published. Also the CPU step at 103 positions = 0.339 s, quoted in P-009's `learned`.
- **1.3:** all group bytes for both SmolLM2 models, totals, both embedding shares, groups-sum-to-total, params = bytes/2 (bf16), tied + same-storage flags, and the 65.2 % MLP share.
- **1.4:** the model file's size by `stat` (1,557,662,528), all GGUF totals (tensor bytes 1,546,700,032; params 772,845,888; 335 tensors; 10,962,496 bytes outside tensors), `tensor_data_offset + tensor bytes == file bytes` exactly, every tensor's `n_bytes == n_elements × dtype size` (335/335), token_embd shape/bytes/share, no `output.weight`, MTP bytes and 2.65 % share, GGUF token list 248,320 = HF tokenizer 248,077 + 243, block classification and per-block bytes, attention/recurrent/MTP totals (which sum with token_embd and output_norm to the exact tensor-byte total), BF16 195/1,544,683,520 and F32 140/2,016,512, and the 17.6 % recurrent-vs-attention size difference.
- **1.5:** the toy softmax probabilities recomputed in plain Python from the definition, nucleus minimality (top-2 mass 0.777 < 0.9 ≤ top-3 mass 0.9017), renormalised nucleus probabilities = the table's "probability" column, expected counts, observed counts, 0 draws outside, the chi-square statistic (0.0856, by hand and scipy) and p = 0.958113, the model part's parameters (64 tokens, T=0.7, top-p 0.9, seeds 0–4), all five first-differing positions recomputed from the raw ids (4, 8, 15, 2, 5), 5 distinct samples, and the cross-check that `sample.json`'s greedy ids and text equal `greedy-cpu.json`'s cached ids and text.
- **Cross-file:** CPU cached ids == GPU cached ids; same `prompt_ids` across greedy-cpu, greedy-cuda and sample.
- **Independent of the repo's own tools:** I also re-derived both SmolLM2 parameter totals from the HF `config.json` shapes with my own formula (360M: 361,821,120; 135M: 134,515,008 — exact), without using `toy/phase1_predict.py`.

`check.py --tables` (inside the run above) re-ran all 8 generated-table commands byte-identically, so no table cell is typed by hand.

## 3. Independent replication against v2

Compared value by value (`/tmp/review-1b/v2-replication.txt`): file bytes, tensor bytes, tensor count, vocabulary (248,320), width (1,024), token_embd type/bytes/share, share-without-MTP (0.33773747…), no separate output tensor, attention block list and per-block bytes (all 6 at 36,710,400), recurrent block list (18) and per-block bytes (all at 43,164,288), MTP block list and bytes (40,916,992), all 25 per-block byte counts, output_norm (4,096), block kinds, and — from v2's params.json — the 360M's parameter total (summed from v2's groups: 361,821,120), total bytes, vocabulary, hidden size, tie and shared-storage flags, dtype, all five group params/bytes, and the embedding fraction (0.13041228770725158).

**82 of 83 exact.** The single difference: v2 labels blk.24's kind `mtp(attn)` and the current run labels it `mtp`. Both mean the same thing (the MTP block is also a full-attention layer; the current run's `block_kind` checks `nextn.*` first, and the answer text says explicitly that blk.24 "is a full-attention layer too", NOTEBOOK.md:480). This is a naming convention, not a measured-number mismatch — explained, so not BLOCKING under the brief's rule. v2's params.json covers only the 360M; the 135M has no v2 counterpart, and I re-derived it from config.json instead (§2). The current `gguf.json` has a different schema with more detail, so this is a genuine replication, not a copy.

**The bootstrap v2-listing claim is confirmed.** `git show v2 --stat` prints the tip commit message ("phase 1"), 17 changed paths and insertion counts, and nothing else; `git ls-tree -r --name-only` prints names only. No file contents, no measured numbers.

## 4. Gate 1 answers (the corrected text, commit 1110df1)

| # | grade | reason and evidence |
|---|---|---|
| 1 | **CORRECT** | Logit = raw pre-softmax score, one per vocabulary entry; LM head = hidden vector × one-row-per-entry matrix; cost grows with V because every row is read and multiplied for every token. 248,320 × 1,024 × 2 = 508,559,360 bytes, verified (NOTEBOOK.md:469–472; `results/1/gguf.json`; my recompute). |
| 2 | **CORRECT** | Token t is part of the input that produces t+1; all three loops are strictly sequential (64 passes for 64 tokens, verified); checking is what parallelises — 41–50 positions at 1.10× and 94–103 at 1.53× a one-position cached step, both recomputed from `step_seconds` (NOTEBOOK.md:474–477). |
| 3 | **CORRECT** | KV cache = per-layer, per-position K/V; Gated DeltaNet = fixed 128×128 per-head state plus a kernel-4 conv state; undo = truncation vs checkpoint/re-run. The 12,288 bytes/token arithmetic checks out from config.json (2 KV heads × head_dim 256 × 6 layers × 2 matrices × 2 bytes) and from the GGUF's `blk.3.attn_k` shape [1024, 512]; the MTP-block cache correction (r1 N3) is in and is right — blk.24 has `attn_q/k/v/output` tensors (NOTEBOOK.md:479–483; `results/1/gguf.json` blocks). |
| 4 | **CORRECT** | 508,559,360 / 1,546,700,032 = 32.88 %, and 33.77 % excluding the MTP block (both recomputed; the second also matches v2's `token_embd_fraction_without_mtp` exactly); "about one third — yes"; the assumption that plain decode skips blk.24 is stated as unchecked (NOTEBOOK.md:485–488). |
| 5 | **CORRECT** | MTP block 40,916,992 bytes; no head matrix in blk.24 (tensor list verified: `nextn.shared_head_norm`, `nextn.eh_proj`, no output-sized tensor); `mtp_use_dedicated_embeddings: False` verified in config.json; 92.6 % of a drafted token's bytes and 36.5 % of a plain step's bytes recomputed (0.92553, 0.36490); ~4.8 ms at 105.7 GB/s (= 68.32 tok/s × 1,546,700,032 B, Phase 0's screened figure) recomputed; the v1 figures (~5.3 ms untrimmed, ~4.0 ms saved by an 8k-row trim, 1.15× → 1.41× / 1.60×) match lines 9 and 59 of `ref/flashference/results/2026-10-26-bf16-b4-trim-draft.md` verbatim (NOTEBOOK.md:490–495). |

The answers are the executor's own words, built from the results. Nothing in them is copied from v2 (I read the only two v2 files allowed and the labels that would give it away — `mtp(attn)`, v2's flat schema — appear nowhere in the NOTEBOOK).

## 5. Losslessness and equality claims

- **1.2 "three methods give identical ids"** (the brief's named claim): true in the raw data for both devices (`ids.nocache == ids.cached == ids.generate`, 64 tokens each, `all_three_equal` true); the script asserts it on every repeat and the run completed 5 repeats per method; check.py re-verifies it; the clean-checkout re-run reproduces it.
- **CPU ids == GPU ids** (NOTEBOOK.md:342): verified from the raw files.
- **Token round-trips exact** for all 12 text × tokenizer combinations: verified (flags, ids/pieces/tokens lengths, and byte re-joining for the partial-free texts).
- **135M ids == 360M ids** for every text: verified from the raw id lists.
- **1.5 cross-check** (greedy ids == 1.2's CPU cached ids; greedy text equal): verified. The toy check is non-circular (draws go through `sample()`; the reference is computed separately in plain Python), the nucleus is minimal, and 0 of 100,000 draws fall outside it.
- No speculative config exists in Phase 1, so §7.4's divergence classifier doesn't apply yet; every equality claim the phase makes is listed above and holds.

## 6. Predictions: ranges, results, honesty

- All six prediction lines were committed (a2b4e25, 11:12:16Z) with `got __`, evidence paths `results/1/…`, and filled once at 042bebe; `check.py --predictions` passes (12 in order, P-006 void from Phase 0).
- Results vs ranges: P-009 (a–d), P-010 (a–d), P-011, P-012 (a–d), P-013 (a–b) all inside/exact, recomputed by me. P-008's Hindi ratio (2.44× vs 3–10×) is the one miss, and it is explained in `learned` with a corrected model (multi-byte Devanagari pieces in SmolLM2's vocabulary), which the raw pieces confirm.
- Blindness disclosures check out: P-012(a) restates `model_size`/`model_n_params` from `results/0/llama-bench-1.json` (I grepped the file: 1,546,700,032 / 772,845,888 — exactly as disclosed), the f32 set was fitted to the seen total (disclosed in the line itself), and P-011's cross-check arithmetic (weight files 32,672 and 30,536 bytes larger than the counted bytes) is right. The genuinely blind parts of P-012 — per-block byte split, attention-vs-recurrent classification, MTP bytes, non-tensor bytes — all matched exactly and replicate against v2.
- Rule 11 nit (already recorded as a deviation in STATE.md after r1): P-011/P-012 state exact values rather than ranges; future exact predictions should say "exactly X".

## 7. No Phase 1 code ran before the prediction commit

- **Commit order:** a2b4e25 (predictions, 11:12:16Z) → a6c843c (scripts, 11:52:51Z) → a7f3d6e (raw results, 11:59:06Z) → 042bebe (reports + answers, 12:04:12Z). Correct order; the tree was clean at each preflight (`dirty: false`).
- **/tmp mtimes:** earliest Phase 1 dev output anywhere is `/tmp/ff-p1-dev/tokens.json` at 11:21:11Z — 8m55s after the prediction commit. `/tmp/ff-p1-dev` spans 11:21:11–11:52:11, `/tmp/ff-p1-broken` 11:33:04, `/tmp/ff-p1-dev0` 11:38:02, `/tmp/ff-p1-short` 11:45:48. Nothing earlier.
- **Job record:** `results/_jobs/phase1-runs/` — cmd `toy/phase1_run_all.sh` with `lock=1`, start 2026-11-01T11:53:20Z, end 11:58:32Z, exit 0; `results/1/preflight.jsonl` has three code-0 readings (11:53:35, 11:56:54, 11:58:32) with `gpu_lock: held-by-caller`, AC power, idle GPU, and the kernel ticks 10833/10000/10000 µs exactly as NOTEBOOK.md:281 quotes them.
- **Session log:** I scanned the executor's session log for every execution of a `phase1_*` script. The only one before 11:12:16Z is `python toy/phase1_predict.py` at 11:11:18Z — the prediction-derivation script, which reads config.json and loads no weights, and whose run is disclosed in a2b4e25's own commit message ("Evidence: … python toy/phase1_predict.py"). No measurement script ran before the commit. The first measurement (a dev run) is at 11:21:11Z, and the official job at 11:53:20Z.
- **Conclusion:** the executor's claim is supported by all four records. One wording nit: NOTEBOOK.md:242 says "Nothing from Phase 1 has been run", which is imprecise — `toy/phase1_predict.py` is Phase 1 code and ran 58 s before the commit (to produce the predictions, which is the right order). "No Phase 1 measurement has been run" would be exact. See N4 below.

## 8. Code review (child-written, executor-reviewed)

I read all seven `toy/phase1_*.py` files and check.py's Phase 1 additions line by line.

- `phase1_greedy.py`: three genuinely different loops; warm-up then A B C interleaved repeats; `CLOCK_MONOTONIC_RAW` timers with CUDA sync at the boundaries and `int(nxt)` forcing completion inside the timed region; a per-repeat `assert` on the ids. No bug.
- `phase1_params.py`: groups every `named_parameters()` tensor (with an `other == 0` assertion that would catch an ungrouped tensor), proves the tied head by `data_ptr` equality, and records whether `named_parameters()` lists `lm_head` at all (it doesn't, with duplicates removed — matching the note). No bug.
- `phase1_gguf.py`: classifies blocks `nextn.*` → mtp, then `ssm_*` → recurrent, then `attn_q/k/v/qkv` → attention — the right order (recurrent blocks carry `attn_qkv` and ssm tensors; blk.24 carries nextn and attention tensors); byte/element accounting verified for all 335 tensors; `bytes_not_in_tensors` equals `tensor_data_offset` and the two sum to the file size exactly. No bug.
- `phase1_tokens.py`, `phase1_sample.py`: no bug. In particular, the executor's fix to the child's toy check is real and necessary: the child's version drew from the sampler's own nucleus, which would have made "never outside the nucleus" true by construction; the committed version computes the reference separately in plain Python and draws through `sample()`. I verified the fix end-to-end (nucleus matches the reference, 0 outside, p = 0.958113).
- `phase1_report.py`: deterministic formatting; the step table computes per-step medians across repeats, then medians over step ranges, exactly as its comment claims (and as I reproduced). No bug.
- `phase1_predict.py`: its Qwen and SmolLM2 arithmetic reproduces the measured counts exactly (I re-derived the SmolLM2 side independently).
- `phase1_run_all.sh` and check.py's Phase 1 section: preflight before the CPU and GPU timings and at the end; clean-checkout re-run under `flock` with `HF_HUB_OFFLINE=1`; `first_difference` reports the exact JSON path (proven by my broken-copy test). No bug.

## 9. The corrections made after review r1 ("re-check of review r1", NOTEBOOK.md:520–536)

- **Step-table medians (1.2, Gate 1 answer 2, P-010 `learned`):** the corrected text compares medians over ranges of steps (1.10× for 41–50 positions, 1.53× for 94–103) instead of single noisy steps. I recomputed all of it; the correction is right, and the old single-step claim ("no slower") was indeed unsupported. P-010's `got`/`error` were untouched, as the entry says — only `learned` changed, which the log's rules allow.
- **1.5 "all five samples gave the same explanation":** corrected to "only seeds 1–3 name Rayleigh scattering … seed 0 adds a false statement". Verified from the raw texts: seeds 1–3 name Rayleigh, seed 0 states "Blue light has the smallest amount of energy" (false — blue photons carry more energy than red), seed 4 attributes it to scattering without naming Rayleigh. The correction is right.
- **The executor's critique of r1's recomputation table:** confirmed. r1's `/tmp/review-1/recompute.py` prints `step_seconds['nocache'][0][k]` — the *first repeat's* step times — labelled as step results (CUDA step 1: 69.7 ms), while the published value is the median over the 5 repeats (29.9 ms, which my median-based recompute matches). The published tables are right; r1's "recomputed" step column was not what its script's label implied.
- The other post-r1 fixes (N3 MTP cache in Q3, N4 retags, N5 "What this shows" + sources, N6 "true at this commit", N7 tick wording, N1 deviation recorded in STATE.md) are all in place and correct; I verified each against the data.

## 10. BLOCKING issues

None.

## 11. NON-BLOCKING issues

**N1. A leftover inconsistency in 1.2: "~10 ms" vs "~8 ms".** NOTEBOOK.md:344 ("Open questions: how the ~10 ms of GPU overhead per step splits…") contradicts the corrected estimate in the same entry, NOTEBOOK.md:338 ("The other ~8 ms is per-step overhead [est: 22.1 − 13.7]"). The ~10 ms reads like a leftover from the pre-correction single-step text (a ~24 ms step minus ~14 ms). One number should change; the tagged ~8 ms is the one the data support.

**N2. A few untagged restatements in prose (rule 10's letter, not substance).** NOTEBOOK.md:414 (1.3–1.4 "What this shows": "248,320-row", "508.6 MB", "2.65 %"), NOTEBOOK.md:492 (answer 5: "248,320 rows, 508.6 MB" — the bullet's [meas] tag covers the tensor names, not these tail numbers), NOTEBOOK.md:335 ("all 32 layers" — a config fact), and the restatements in the "What I did not test"/"Open questions" bullets (:275, :277, :344, :350). Every one duplicates a tagged, verified number from the same entry. (r1's N5 listed some of these; the list is not yet exhausted.)

**N3. The predictions entry (NOTEBOOK.md:233–244) has no "What I did not test".** The four report entries all have one. Whether a prediction-only entry counts as a "report" under rule 15 is a judgment call; if it does, it needs the section (e.g.: not tested — whether the config-derived counts survive a different GGUF conversion; whether the timing ranges hold on battery).

**N4. "Nothing from Phase 1 has been run" (NOTEBOOK.md:242) is imprecise.** `toy/phase1_predict.py`, committed in a2b4e25 itself, ran at 11:11:18Z, 58 s before the commit — properly (predictions must be derived before they're committed), and disclosed in the commit message, but the sentence, read strictly, is false for that one script. The five measurement scripts had never run, which is what matters. Reword to "no Phase 1 measurement has been run".

**N5. (Confirming r1 N1, already remediated)** P-011/P-012 are exact-value predictions; the deviation is now recorded in STATE.md with a sensible reading (an exact count is a range of width zero). Future exact predictions should still say "exactly X" on their face.

**N6. (Confirming r1 N2, disclosed)** P-012(a) wasn't blind and the f32 tensor set was fitted to the seen total. Disclosed in the prediction line itself; the blind parts (per-block split, block kinds, MTP bytes) matched exactly and replicate against v2, which is the stronger check.

**N7. v2's blk.24 kind label (`mtp(attn)`) differs from the current one (`mtp`).** Same facts, different label (v2's records that the MTP block is attention; the current text says so in prose). If a future reviewer diffs the two files' `kind` fields, this difference will reappear; a one-line note in the NOTEBOOK or the report script would save them the investigation.

## 12. What I did not test

- No new timings, per the brief. The medians, ranges, ratios and step tables are recomputed from the raw `measured.seconds`/`step_seconds`; check.py's re-runs used `--repeats 0` (warm-up only, no timing output).
- I did not re-tokenize the six texts myself; I verified the recorded ids, pieces, counts and flags, and relied on check.py's clean-checkout re-run (which does re-tokenize) for reproduction.
- I did not run llama.cpp or check which tensors a plain decode step actually reads; Q4's assumption (plain decode skips blk.24) remains the answer's own honest caveat.
- I did not re-verify v1's B4 numbers beyond the cited lines 9 and 59 and README line 37 (all verified verbatim), nor v1's other results.
- The 135M has no v2 counterpart; I checked it against config arithmetic and its own params.json instead.
- I did not examine the opencode child's session log (only its committed code and the executor's account); the child made no commits.
- I did not open, list or print anything under `prompts/test/` (it doesn't exist yet, and check.py confirms the test set isn't written).
- I did not re-run the Phase 0 checks or re-grade Phase 0's entries; the re-check entry's own caveat (NOTEBOOK.md:536) that Phase 0 prose wasn't re-read stands.

## 13. Where this review and the first review differ

My grades and verdict were fixed before I read `reviews/phase-1-r1.md` (proof: `/tmp/review-1b/my-grades.txt`, written before the read). Comparison:

- **Grades and verdict: identical.** All five answers CORRECT, no BLOCKING issue, verdict PASS. On the substance we agree everywhere.
- **I confirm the executor's finding that r1's recomputation had a wrong table.** r1's script printed first-repeat step times as step results (CUDA step 1: 69.7 ms vs the 29.9 ms median), and its report's step row doesn't reliably correspond to its own script's output. My median-based recompute matches the published tables exactly, so the published numbers were never in doubt.
- **New findings r1 didn't have** (it reviewed the pre-correction text): the ~8/~10 ms inconsistency (N1); the additional untagged restatements and the `mtp(attn)` label difference (N2, N7); the missing "What I did not test" in the predictions entry (N3); the "Nothing from Phase 1 has been run" wording (N4).
- **Where I went beyond r1:** re-derived the SmolLM2 parameter totals from config.json independently; verified every tensor's n_bytes = n_elements × dtype size and offset+tensors = file size; verified the nucleus's minimality and the chi-square by hand and with scipy; verified v1's README line 37 (r1 explicitly did not); verified the preflight tick readings and the 69.7 ms first-repeat value; verified r1's scratch evidence (`/tmp/review-1/broken/tokens.json` does hold 248,078, `recompute.py` runs), which the executor's re-check had vouched for.
- **Where r1 went beyond me:** it checked for paraphrase from v2's NOTEBOOK (and disclosed its own out-of-bounds read of v2:NOTEBOOK.md's first 90 lines — a deviation I did not repeat; my v2 reading stayed within the two files and `--stat`).
- r1's N1–N7 were all valid; every one is fixed or recorded, and I checked each fix against the data.

## 14. Reviewer compliance

- The only file I created or changed is `reviews/phase-1-r1-second.md`. Scratch work is entirely under `/tmp/review-1b/`.
- No git write commands; read-only git only (log, show, diff, status, ls-tree).
- Every command that imports torch or transformers, or runs a llama.cpp binary, ran under `flock /tmp/flashference-gpu.lock` — in practice, only check.py's internal re-runs, which take the lock themselves (the brief says to run check.py bare for exactly that reason); my own scripts import neither torch nor transformers. No two GPU uses overlapped.
- The deliberate break was in a copy under `/tmp/review-1b/broken/`; no repo file was touched.
- Nothing under `prompts/test/` was opened, listed or printed. No installs, no downloads, no sudo, no writes under `/mnt/c`.
- v2 reads: only the two permitted results files and `git show v2 --stat`.
- Final `git status --short` (below the verdict in my reply, and run at the end): only `?? reviews/phase-1-r1-second.md`.

verdict: PASS
