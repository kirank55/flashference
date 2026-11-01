# plan.md: the flashference framework

Written 2026-11-01 for the same machine as branch `op`: RTX 3050 A Laptop GPU (4 GB, compute capability 8.9), WSL2 Ubuntu 24.04, llama.cpp b11884 (63412d873).

---

## 0. Read this first

### The goal

Branch `op` has this goal: lossless speed-ups for LLM decoding that are measured honestly, are explained, and leave a study trail the owner can learn from. This branch has the same goal, with one more requirement: **the result has to work on other models.**

`op` makes one model fast (Qwen3.5-0.8B). The framework makes a *procedure* that works on any model that llama.cpp runs. Each step of the procedure is a script:

```
ff inspect  <model.gguf>                → model card (what the model is; which drafters and hooks apply)
ff baseline <model> <prompts>           → plain-decoding control, noise floor, numerics threshold
ff rank     <model> <prompts>           → draft-vocabulary ranking built from this model's own drafts
ff tune     <model> <prompts/dev>       → the fastest lossless config, frozen
ff verify   <model> <frozen config>     → losslessness proof on dev, then once on test
ff report   <model>                     → report + claims, generated from raw data
ff run      <model.gguf>                → all of the above, end to end
```

The result for a new model is a frozen config, a proof that it is lossless, and a report. All three come from one command, with no code written for that model.

### The hard constraint: no per-model code

| # | Rule | How it's checked |
|---|---|---|
| C1 | **No edits under `llama.cpp/src/models/`**, and no edits to a model-specific branch anywhere else (`switch (arch)` cases, per-model tensor loading). | `ff check` fails if `git diff stock..<branch>` touches `src/models/` or adds an `LLM_ARCH_` reference. |
| C2 | **The target model's computation is never changed.** Patches change only the *drafter* (what gets guessed), how guesses are scheduled, or host-side bookkeeping. The target still computes every logit it computes in stock llama.cpp. | For every patch, the same plain-decoding run with the patch built in, the feature both on and off, must hash-equal stock's output token for token on all dev prompts. That holds for every model in the matrix. |
| C3 | **Patches live only in model-agnostic code:** `common/`, `tools/server/`, `src/llama-graph.*`, `src/llama-context.*`, the generic parts of `src/llama-model.*` (tensor creation, not per-arch loading), and `ggml/` (optional track F10 only). | Same `ff check`, with an allow-list of paths. |
| C4 | **Every patch works unchanged on at least 2 architectures** from the model matrix (§F1), and on at least 2 drafter kinds where that applies. | A coverage script lists, for each architecture, whether each hook fires. A test asserts it fires on the matrix models. |
| C5 | **Model knowledge lives in data, never in code.** Vocabulary size, head tensor, tying, chat template, drafter availability and rankings all go in a generated model card (`models/<id>/card.yaml`) and in per-model data files. | No model name or vocabulary size appears in `ff/`, `bench/` or a patch (grep check). |

**What this drops from `op`'s plan.md:**
- **8.3 and 8.4, the certified target head.** They change how the target computes its logits: `result_output` in each model's graph. That breaks C1 and C2.
- **8.8, the self-distilled MTP head.** It needs each model's MTP forward pass re-implemented in PyTorch, then trained. That is per-model code and per-model training.
- **8.9, lossless weight compression.** It needs a new ggml weight type and a fused CUDA kernel. That's a fork of ggml, not a framework feature. It's out of scope; §9 lists it.
- **v1's patch 1 as written.** It edits `src/models/qwen35.cpp` (`graph_mtp`), so it breaks C1. Phase F4 rebuilds the same idea as a generic hook.

### Why the trim can be generic (checked in b11884, 2026-11-01)

- v1's trim replaces the draft head's matrix multiply inside `llama_model_qwen35::graph_mtp` with a K-row matrix, then pads the logits to the full vocabulary with `-inf`. That is a model-file edit.
- Every MTP graph is built with graph type `LLM_GRAPH_TYPE_DECODER_MTP`. The context maps `LLAMA_CONTEXT_TYPE_MTP` to it (`src/llama-context.cpp:115`).
- By a grep, 15 of the 17 `graph_mtp` implementations in `src/models/` compute the draft head through the shared `llm_graph_context::build_lora_mm` (`src/llama-graph.cpp:1632`). The exceptions are `deepseek4.cpp` and `bailingmoe3.cpp`. F1 confirms this by reading the code.
- So the substitution can live inside `build_lora_mm`: "if this is a draft graph and `w` is the head tensor, multiply by the K-row tensor instead, then pad and mask". It happens in one generic place, and no model file changes.
- A separate draft model (`--spec-type draft-simple`) runs in its own context, so the same hook covers its output head. That gives two drafter kinds through one hook.
- The target's graphs never take this branch, so C2 holds by construction. F4 still tests it.

### Three levels of "lossless" (same words as op)

| Level | Meaning |
|---|---|
| **L1: distribution-lossless** | With sampling, the output distribution equals the target's. |
| **L2: verify-lossless** | With greedy decoding, every output token is the target's argmax as computed in the verification batch. It may differ from plain decoding only through floating-point order. |
| **L3: bit-exact** | Byte-identical to plain greedy decoding of the same target, always. |

Every config the framework emits states its level and how that was checked. The default target is L2. L3 is the optional track F10.

### Rules

`op`'s 8 rules all apply, unchanged:
1. predict before you measure;
2. no code you can't explain;
3. one change at a time;
4. a same-session control;
5. raw data → generated tables;
6. a dev/test split;
7. losslessness is checked;
8. write down what you did not test.

These are added:

9. **Every number belongs to a triple:** (model card hash, GPU, llama.cpp commit). A number without its triple is not a result.
10. **Never slower than plain.** `ff tune` always includes plain decoding as a candidate. A config is chosen only if it beats plain by more than 2 × that model's noise floor on that domain. Otherwise the answer for that domain is "plain". The framework's worst case is 1.00×, not a slow-down.
11. **A held-out model.** In F1, pick one model that is never used during development, like a test prompt set. Its sha256 is committed and the file is not downloaded until F12. The final portability claim rests on it.
12. **Two independent reviewers from different model families at every gate**, as on op (STATE.md `second_reviewer`). The second doesn't read the first's report. A FAIL from either makes the round a FAIL.

### Relationship to branch `op`

- `op` remains the learning track on one model. This branch doesn't wait for it, and it never edits `op`'s files.
- **Reused from op**, copied with their commit id noted in NOTEBOOK.md:
  - the Phase 0 environment: CUDA 12.8 in `~/opt`, the b11884 `stock` build, and the WSL2 clock decisions in op's STATE.md;
  - `autopilot/` (the guard, the GPU lock, job.sh, check.py's prediction-order check);
  - op's Phase 4 harness spec.
- **Shared resources.** The GPU lock (`/tmp/flashference-gpu.lock`), `models/` (a symlink) and the `llama.cpp/` checkout. This branch's llama.cpp branches are named `fw-*`, so they never collide with op's `my-trim` or `exp-8.x`.
- **Worktree.** Work happens in a separate git worktree, `~/code-wsl/flashference-framework`, so op's checkout is never switched.
- **Execution.** Same operating model as op's `autoplan.md`: an executor and fresh-context reviewers, predictions committed before their runs, STATE.md as the progress board, BLOCKED.md for blocks. Bootstrap (F0) writes this branch's `autoplan.md`, adapted from op's. Where the two disagree, this plan's C1–C5 and rules 9–12 win.

---

## Layout

```
flashference/  (branch framework)
├── plan.md  autoplan.md  STATE.md  NOTEBOOK.md  BLOCKED.md  README.md
├── autopilot/                 # from op: guard.sh, gpu-run.sh, job.sh, preflight.sh, check.py
├── ff/                        # the framework (Python): one module per command
│   ├── inspect.py  baseline.py  rank.py  tune.py  verify.py  report.py  run.py
│   ├── costmodel.py           # predicted speed-up from coverage, head bytes and measured timings
│   └── check.py               # C1–C5 checks on llama.cpp branches and on ff/ itself
├── bench/                     # harness: bench.py, report.py, divergence.py, tests
├── patches/                   # git format-patch stock..fw-<name>, each applies to clean b11884
│   ├── fw-subvocab/           # F4: generic draft sub-vocabulary head
│   ├── fw-subvocab-quant/     # F6: quantized sub-vocabulary head
│   ├── fw-arbitrate/          # F7: drafter arbitration
│   └── fw-…/                  # F9, F10 if they pass
├── models/<id>/               # per-model DATA only (the GGUF itself is git-ignored)
│   ├── card.yaml              # generated by ff inspect
│   ├── rank.txt               # generated by ff rank
│   └── frozen.json            # generated by ff tune; committed before the test run
├── prompts/dev/  prompts/test/  prompts/test.sha256  prompts/README.md
├── results/<model-id>/<phase>/  # raw JSONL and generated reports
├── results/matrix.md          # generated: every model × domain × config
├── results/claims.md          # generated claims ledger
├── reviews/
└── docs/  study-guide.md  hooks.md  porting.md
```

**The portable kit.** To use the framework on another model, or in another repo, you need only:
- `ff/` and `bench/`;
- `patches/`;
- `prompts/`;
- the model card that `ff inspect` generates.

Nothing in the kit names a model.

---

## Phase F0: Bootstrap (2 units)

1. Create the worktree. Copy `autopilot/` and the environment notes from op at a named commit. Link `models/` and `llama.cpp/`.
2. Write this branch's `autoplan.md`, adapted from op's. Changes: the phase list, C1–C5 as hard rules, and `fw-*` branch names.
3. Write `ff/check.py` with the C1, C3 and C5 checks first, before any patch exists. Test each check on a deliberately bad diff kept in `/tmp`.
4. Commit STATE.md with `phase: F1`.

**Gate F0:** `ff check` fails on a diff that touches `src/models/qwen35.cpp` and passes on an empty diff. op's checkout is untouched (`git -C <op worktree> status` is clean, and HEAD is unchanged).

---

## Phase F1: Model census and model cards (4 units)

**Goal:** know which models the framework can run on this GPU, and what each one offers.

1. **`ff inspect`.** It reads a GGUF with the `gguf` library and writes `card.yaml`:
   - architecture, `n_vocab`, `d_model`, `n_layer`;
   - head tensor name, dtype and bytes, and whether it's tied to the embedding;
   - the head's share of total weight bytes;
   - whether `nextn`/MTP tensors exist;
   - tokenizer hash: the vocabulary list plus merges, so draft-model compatibility is checked by data, not by name;
   - chat template;
   - weights sha256.

   Check it on Qwen3.5-0.8B against op's Phase 1.4 numbers (`git show op:results/1/gguf.json`).
2. **Hook coverage, statically.** A script reads `src/models/*.cpp` at 63412d873. For each architecture it records whether the draft head goes through `build_lora_mm` with the head tensor. Its output goes to `docs/hooks.md`. Read the two exceptions, deepseek4 and bailingmoe3, by hand and record why they differ.
3. **The model matrix.** Requirements:
   - at least 3 development models, from at least 2 architecture families;
   - each fits in 4 GB with a 4096 context, including the KV cache or recurrent state;
   - at least one with an in-model MTP head;
   - at least one target + small draft pair whose tokenizer hashes are equal;
   - at least one with an untied head;
   - vocabulary sizes that differ by at least 2×.

   Candidates to check, none verified yet:
   - Qwen3.5-0.8B bf16 with MTP: the v1 anchor; this one is known;
   - a Qwen3 target at Q8_0 with Qwen3-0.6B as the draft;
   - a Llama-3.2 or Gemma-family target with a small same-tokenizer draft.

   Record each candidate's card, then pick. Quantized targets are fine. Losslessness is then relative to that quantized target, and the card says so.
4. **The held-out model (rule 11).** Pick it now with the same criteria, but from a third family if one fits. Commit its repo id, filename and sha256 to `models/heldout.yaml`. Don't download it.

**Prediction (commit before F3):** for each matrix model, the ceiling of the head-trim gain, from bytes only. The draft head's share of a draft step's bytes, times the draft steps per cycle, gives the largest saving trimming can buy. The formula is in `ff/costmodel.py`. Predict which model gains most and least from F4.

**Gate F1:**
1. Why can a trimmed draft head never change the output, while trimming the target's head could? Tie the answer to C2.
2. From the cards: which matrix model has the largest head share, and what does that predict for F4?
3. Which drafter kinds does each model support, and how was tokenizer compatibility proven?
4. Why must the held-out model stay unseen until F12?

---

## Phase F2: Multi-model harness (6 units)

Build to op's Phase 4 spec: `bench.py`, `report.py`, `divergence.py`, tests, `--final`, the test ledger and the prediction gate. If op has finished its Phase 4, port that code and note the commit. Additions:

- Every raw line carries the triple (rule 9), the card hash and the frozen-config hash.
- **Chat formatting comes from the card**, applied by llama-server's template, never by a hand-written string.
- **Prompts.** `prompts/dev/` has 10 prompts in each of 4 domains: prose, code, copy-heavy, and multilingual (es, de, zh, hi). They're plain text with no model-specific formatting. As on op, a fresh agent writes `prompts/test/`, and the executor never opens it.
- **`ff baseline`** runs, for each model:
  - an A/A session, which gives that model's noise floor, with the definition written down;
  - a numerics probe, which gives that model's `tau_numerics`: at least 20 first divergences between speculative and plain output, classified by `divergence.py`, as op's Phase 6 does. Each value goes into the card.
- Plain tok/s for a model with K-row heads and two drafters is a different number from Qwen3.5's. Only ratios against the same session's control are compared, never absolute speeds across models.

**Gate F2:** a noise floor and a `tau_numerics` exist for every matrix model, each with its evidence file. `report.py` regenerates `results/matrix.md` with the plain column filled. `pytest bench/` passes.

---

## Phase F3: Stock tuning, with zero patches (5 units)

**Goal:** before writing any C++, find what stock b11884 already gives on each model. That is the framework's floor, and it ports to any model for free.

1. **`ff tune`, version 1.** For each model and dev domain, search over stock options only:
   - plain, and plain with `-bs`;
   - every applicable `--spec-type`: `draft-mtp`, `draft-simple`, and `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`;
   - `draft-eagle3`, `draft-dflash` and `draft-dspark` too, if a published draft head for that target fits in memory (F1 records which exist);
   - two-drafter chains;
   - `--spec-draft-n-max` from 1 to 8, and the n-gram sizes.

   Use successive halving: a short run on 2 prompts, then keep the top third. Don't sweep the full grid at full length.
2. Every candidate's outputs go through `divergence.py` against plain. A candidate with an unexplained mismatch is dropped, and the drop is logged.
3. `frozen.json` per model is committed. Then `ff verify --final` runs once on test.

**Prediction:** for each model and domain, which stock config wins and its ratio. Commit it before the sweep.

**Gate F3:**
1. The best stock config per model and domain, from `report.py`. Are there domains where plain wins?
2. Does the same config win on every model? If not, explain the difference from the cards: head share, drafter availability, vocabulary.
3. How far was each prediction off, and why?

---

## Phase F4: Generic sub-vocabulary draft head ★ (8 units)

**Goal:** v1's biggest idea, the trimmed draft head, rebuilt so that it works on any model.

1. **Design note first** (NOTEBOOK.md, half a page):
   - where the K-row tensor is created: the generic loader path, gathering rows of the head tensor named in the card;
   - how a graph knows it's a draft graph: `gtype == LLM_GRAPH_TYPE_DECODER_MTP`, or a context flag on the draft model's context;
   - how `build_lora_mm` recognises the head tensor: by pointer identity with the model's output head;
   - why padding to `n_vocab` with `-inf` keeps every downstream consumer unchanged;
   - why the target can't take this branch.
2. **Interface.** Generic flags: `--spec-draft-vocab-file <rank.txt> --spec-draft-vocab-k <K>`. Environment variables mirror them for the bench.
3. **Implement in 4 steps**, each a commit on `fw-subvocab` with its own test, as on op:
   1. the tensor gather;
   2. the hook in `build_lora_mm`;
   3. the pad and mask;
   4. the flags.

   A reviewer reads each step with the brief "find anything that could change the target model's output, or anything model-specific".
4. **Tests:**
   - **C2, target untouched:** plain decoding with the patch built in and the feature on hash-equals stock, on every matrix model.
   - **Draft-only change:** at K = n_vocab (no trim), speculative output hash-equals the unpatched drafter's output on dev, so the hook alone changes nothing. At K < n_vocab, outputs equal plain within `tau_numerics`, classified by `divergence.py`.
   - **C4, coverage:** the hook fires, logged once per context, on the MTP model and on the target + draft pair.
   - **Anchor:** on Qwen3.5-0.8B with v1's ranking file, the trimmed MTP ratios match op's Phase 6 reproduction table, or v1's, within about 10 %. If not, investigate in op's order: prompts, then patch, then acceptance.
5. **Clean-room check, as on op.** A fresh agent gets only the design note and `docs/hooks.md`. It implements the hook on `fw-subvocab-cr`, and its outputs must hash-equal `fw-subvocab`'s.

**Prediction:** from F1's ceilings and the F3 baselines, the trim gain per model at K = 8k, 32k and full.

**Gate F4:**
1. Show that no line of the patch is model-specific: the C1, C3 and C5 checks, plus your own reading.
2. Explain the coverage table. Which architectures does the hook miss, and why?
3. Why does the gain differ between models? Answer with the cards' byte counts, not by guessing.

---

## Phase F5: Per-model vocabulary ranking (6 units)

**Goal:** a ranking file built automatically for any tokenizer, with K chosen by a cost model instead of by hand.

1. **The coverage tool** (op's 8.2 idea). Log the *untrimmed* drafter's argmax at every draft step, over the dev prompts and a domain-mixed corpus (WikiText-2, standard-library code, and the copy-heavy pattern). coverage(K) is the fraction of those argmaxes inside the top K of a ranking.
2. **`ff rank`.** It builds candidate rankings per model:
   1. corpus token frequency (v1's method);
   2. the model's own drafts;
   3. a mix of the two.

   It writes `models/<id>/rank.txt`. It never uses test prompts.
3. **Validate coverage → acceptance on each model.** Use 4–5 configs, and commit the predicted rank correlation first. If coverage doesn't predict acceptance on some model, stop and find out why before using the tool there.
4. **`ff/costmodel.py`** predicts the speed-up from coverage(K), the K-row head's bytes and measured T_verify and T_draft. `ff tune` uses it to propose K. It then measures only the predicted best K and its neighbours.

**Gate F5:** on every matrix model, the cost model's best K is within one step of the measured best K, or the miss is explained. The ranking built from the model's own drafts is compared with corpus frequency on each model, and the report says which is better where.

---

## Phase F6: Quantized sub-vocabulary head (5 units)

op's 8.1, kept generic: after the gather, quantize the K rows with `ggml_quantize_chunk` to `--spec-draft-vocab-type bf16|q8_0|q4_0`.

- **Losslessness argument:** the drafter's numbers never reach the output. C2 is tested again.
- **The sweep:** K ∈ {8k, 16k, 32k, 64k, n_vocab} × type ∈ {bf16, q8_0, q4_0} × domain, on each model. Successive halving, with the cost model's proposal measured first.
- **The prediction to commit:** the largest K whose q4_0 head costs no more than the 32k bf16 head, on each model, and whether multilingual gains most. A bigger K covers more of the non-English tokens.

**Stop if:** no type beats bf16 at the best K by more than 2 × the noise floor on any model or domain. "Stopped" is a full result.

---

## Phase F7: Drafter arbitration (5 units)

op's 8.5, in `common/speculative.cpp` only.
- Keep a per-sequence EMA of each chained drafter's acceptance. Skip a drafter while its EMA is below τ, and retry it every M cycles.
- τ and M are tuned per model on dev by `ff tune`. They are stored in `frozen.json`, never in code.

**Done when:** on each matrix model, one config reaches at least 0.98× of the best single-drafter config on prose and at least 0.95× of the best chain on copy-heavy, on test.

**Optional, in this phase's timebox:** v1's context-adaptive rows (patch 2, E4) rebuilt on the F4 hook. These are rows added to the K-row head from the prompt's own tokens. Only if the rebuild passes C1–C5.

---

## Phase F8: Per-model profile (5 units)

op's Phase 7, done once per matrix model: a per-token budget of the target's GEMVs, head, drafter, sampling and idle gaps, plus graph re-captures. Use nsys if the WSL2 trace has kernel rows. Otherwise use the synchronizing eval-callback method, tagged [partial].

**Gate F8 decisions** come from a script, per model:
- idle gaps plus re-captures at 5 % or more of a cycle → F9 is in for that model;
- otherwise F9 is out.

Record it in STATE.md.

---

## Phase F9: Fewer host round-trips (conditional, 6 units)

op's 8.6, limited to `common/speculative.cpp`, `tools/server/` and `src/llama-context.*`. Examples are keeping the verify batch shape fixed so CUDA graphs are replayed, not re-captured, and removing a host sync between draft steps. Fix only the largest gap, re-profile, then re-measure.

**Level:** L2. Output hashes must equal the pre-fix reference exactly, because no maths changes.

---

## Phase F10: Bit-exact speculation (optional, 12 units)

op's 8.7, restricted by C3: changes are allowed only in generic `ggml/` ops and backends, such as a batch-invariant matmul path, never in a model file.
1. Build a minimal reproducer on one model.
2. Find the first op that differs between batch sizes 1 and 3.
3. Make that op batch-invariant.

**Stop if** the first op that differs is model-specific code (a custom op used by one architecture). Write that up as the finding.

**Done when:** 100 % byte-identical output on test for at least 2 matrix models, with the speed cost documented.

**Note on C2:** this track does change how the target computes its logits, but only to make speculative output equal plain output. So the C2 test becomes "with the feature on, speculative output = plain output, byte for byte".

---

## Phase F11: The framework package (6 units)

1. **`ff run <model.gguf>`** chains inspect → baseline → rank → tune → verify → report, and is resumable from STATE-like checkpoints in `results/<model-id>/`. A model whose tokenizer has no compatible draft and no MTP head still gets the n-gram drafters and arbitration. When nothing beats plain, it reports "plain".
2. **`docs/porting.md`:**
   - how to add a model: nothing, if `ff inspect` succeeds;
   - what to do when the coverage script says the hook misses an architecture: the honest answer is "that model gets stock and n-gram drafters only";
   - how to rebase the patch stack to a newer llama.cpp.
3. **Rebase test.** Apply the patch stack to one newer llama.cpp release. Run the C2 test and one model's anchor ratio. Record what broke. This measures how portable the patches are across llama.cpp versions, not only across models.
4. `results/claims.md`, generated: claim → model triple → evidence file → level → test-confirmed.

**Gate F11:** `ff run` on one matrix model, from a fresh clone (`/tmp/ff-repro-<date>`), reproduces that model's frozen ratios within 2 × its noise floor. It does this against its own same-session control.

---

## Phase F12: Portability proof and write-up ★ (5 units)

1. **The held-out model.** Download it and check its sha256 against `models/heldout.yaml`. Run `ff run` once, with no code changes allowed: any change to `ff/` or `patches/` after the download voids the result. It passes if:
   - the losslessness checks pass;
   - the chosen config is at least the stock-tuned ratio;
   - every step ran without a model-specific edit.

   A fail is a published finding, not something to fix and re-run.
2. **Independent check.** A fresh agent gets only `README.md` and runs `ff run` on one matrix model. It passes if the ratio is within 2 × the noise floor of the published one.
3. **`README.md`:** what the framework does, the generated `results/matrix.md`, which architectures the hooks cover, and the limits.
4. **`docs/study-guide.md`,** as on op: 3 key facts per phase with evidence links, gate questions as a self-quiz, and "try it yourself" commands.
5. **Final review** by two reviewers from different model families. Then set `status: done-all`.

---

## Budget

| Phase | Units |
|---|---|
| F0 | 2 |
| F1 | 4 |
| F2 | 6 |
| F3 | 5 |
| F4 ★ | 8 |
| F5 | 6 |
| F6 | 5 |
| F7 | 5 |
| F8 | 5 |
| F9 (conditional) | 6 |
| F10 (optional) | 12 |
| F11 | 6 |
| F12 ★ | 5 |
| **Total** | **~63 required, ~75 with F9 and F10, plus review rounds** |

On op, Phases 0–1 ran at about 1.6× their timeboxes once review rounds were counted. Plan for about 120 units in total. A hard block comes at 160.

GPU time scales with the number of models: expect about 3–4× op's per-phase GPU hours for F2–F8. Successive halving in `ff tune` exists to keep this bounded.

---

## Done when

- Every matrix model and the held-out model have a frozen config, an L2 proof (or L3, if F10 passed), and a generated report.
- Every config is at least 1.00× plain on every domain (rule 10), and beats plain by more than 2 × the noise floor wherever it isn't plain.
- `ff check` passes on every `fw-*` branch: no model code, target untouched, no model names in code.
- The held-out run and the independent check are recorded, pass or fail.

## What this plan does not test (fill in at the end)

- GPUs other than the RTX 3050 Laptop, unless one is borrowed or rented for a ratio-only check.
- Models over 4 GB, and multi-GPU or CPU-offload setups.
- Batch sizes above 1 (one stream, as on op).
- Sampling (L1) beyond one prose check per model.
- Architectures the coverage script marks as missed by the hook.
- Out of scope by design: certified target heads (op 8.3/8.4), trained draft heads (op 8.8), weight compression (op 8.9).
