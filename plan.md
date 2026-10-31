# plan.md: a learning-first route to lossless LLM speed-ups

Written 2026-10-29 for your machine: RTX 3050 A Laptop GPU (4 GB, compute capability 8.9 / Ada), WSL2 Ubuntu 24.04, 7 GB RAM visible to WSL.

The goal is to reach the experiment ideas we discussed (quantized drafter, better draft vocabulary, certified target head, drafter arbitration, bit-exact speculation, and so on), and to understand every line of code and every number along the way, so you never again publish results you can't explain.

---

## 0. Read this first

### The mistake this plan prevents

flashference has real, careful-looking numbers, but you can't explain them. That means you can't tell which results are solid, which are artefacts, or what to try next. The fix isn't "stop using AI". The fix is **"never move forward past something you can't explain."** Every phase below ends in a **gate**: questions you answer in your own words, plus a prediction you make *before* measuring. You don't start the next phase until you pass the gate.

### The 8 rules (print these)

1. **Predict before you measure.** Write down the number you expect and why. Afterwards, record how far off you were. Your prediction error is the honest measure of your understanding.
2. **No code you can't explain line by line.** AI may write code (see Appendix A), but before you commit it, write 3–5 sentences in your notebook saying what it does and why it can't change the model's output.
3. **One change at a time.** Every experiment changes one thing against a control.
4. **Compare against a control from the same session.** Never compare with last week's number. Your GPU's speed drifts by about 5 % between sessions.
5. **Keep raw data, generate tables.** No number goes into a report by hand. Scripts turn raw JSONL into tables.
6. **Keep a dev set and a test set.** Tune only on dev prompts. Report on test prompts you never looked at while tuning. (flashference tuned K on the same 6 prompts it reported. That's a weakness.)
7. **Losslessness is checked, not assumed.** Every config is compared, output hash against output hash, with a reference.
8. **Write down what you did NOT test.** Every report ends with that list.

### Three levels of "lossless" (use these words precisely)

| Level | Meaning | Who has it |
|---|---|---|
| **L1: distribution-lossless** | With sampling (temperature > 0), the output *distribution* equals the target model's. | Speculative sampling with rejection sampling |
| **L2: verify-lossless** | With greedy decoding, every output token is the target's own argmax *as computed in the verification batch*. Output may still differ from plain decoding through floating-point order. | flashference today (6/18 prose outputs byte-identical to plain) |
| **L3: bit-exact** | Byte-identical to plain greedy decoding of the same target, always. | Nobody yet; that's experiment 8.7 |

Every experiment must state which level it keeps and how you checked it.

### Who this is for, and timing

This plan assumes you can run terminal commands and read simple Python. It assumes **no** C++, CUDA, or ML maths. At about 8–10 hours a week:

| Phase | What you learn | What you build | Gate | Time |
|---|---|---|---|---|
| 0 Workbench | WSL2, CUDA toolkit, builds | Working llama.cpp + Python env | Everything runs | 1 wk |
| 1 One token | How an LLM produces a token; where the bytes are | Hand-written generation loop | 5 questions | 2 wk |
| 2 Speed = bytes | GPU memory bandwidth, GEMV, batching, floating point | Micro-benchmarks + plots | A prediction of plain tok/s | 1–2 wk |
| 3 Speculation | Speculative decoding, n-gram drafting, trimmed draft head | Your own toy speculative decoder | Re-explain flashference | 2–3 wk |
| 4 Measuring | Noise, drift, controls, statistics | Your own benchmark harness | Noise floor known | 1–2 wk |
| 5 llama.cpp | C++ reading, ggml, the speculation code path | Logging hooks, call diagram | Annotated patch 1 | 2–3 wk |
| 6 Rebuild | flashference's trick, written by you | Your own patch + reproduction in WSL | Reproduction table | 2 wk |
| 7 Profile | Where the ~15 ms per token go | Per-token time budget | Budget within 5 % | 1 wk |
| 8 Experiments | One idea at a time | See section 8 | Per experiment | open |
| 9 Write-up | Honest reporting | Report + flashference audit | Someone else reproduces it | 1 wk |

You can skip a phase if you can pass its gate cold. Be honest about that.

### How the earlier ideas map to experiments

| Idea (earlier answer) | Experiment | Can start after |
|---|---|---|
| 1. Quantize the drafter, bigger K | 8.1 | Phase 6 |
| 2. 32k + dynamic rows; better ranking | 8.2 | Phase 6 |
| 3. Profile first | Phase 7 | Phase 6 |
| 4. Offline mixed-precision certified head | 8.3 | Phases 3 + 5 |
| 5. Certified head in llama.cpp | 8.4 | 8.3 is positive |
| 6. Drafter arbitration | 8.5 | Phase 6 |
| 7. Fewer host round-trips, graph reuse | 8.6 | Phase 7 shows it's worth it |
| 8. Bit-exact speculation | 8.7 | Phase 7 |
| 9. Self-distilled MTP head | 8.8 | 8.1 and 8.2 |
| 10. Lossless weight compression | 8.9 | Phase 7 shows GEMVs are bandwidth-bound |
| 11. Portability (bigger model, other GPU) | 8.10 | Phase 6 |

### Your lab layout

```
~/code-wsl/flashference/    # this repo, branch v2 ("the lab"); v1 is tag v1.0
  plan.md                   # copy of this file
  NOTEBOOK.md               # dated entries: predictions, results, questions (template in Appendix B)
  .gitignore                # models/, llama.cpp/, ref/, .venv/
  models/                   # GGUF files (not committed)
  llama.cpp/                # your checkout; one git branch per experiment
  ref/flashference/         # git worktree of tag v1.0; read-only reference, never build on it
  toy/                      # Phase 1-3 Python
  bench/                    # your harness (Phase 4)
  prompts/dev/  prompts/test/
  results/                  # raw JSONL + generated reports (committed)
  analysis/                 # offline studies (8.3, 8.9)
```

Keep everything on the Linux filesystem (`~/...`), not under `/mnt/c/...`. Windows-mounted files are far slower to read in WSL.

---

## Phase 0: Workbench (about 1 week)

**Goal:** everything builds and runs in WSL. You don't need to understand it yet. You're laying out tools.

### 0.1 Give WSL more memory

Building CUDA code with 7 GB can run out of memory. On **Windows**, create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=12GB
```

Then in PowerShell run `wsl --shutdown` and reopen your terminal. Check with `free -g`.

### 0.2 Install the CUDA toolkit inside WSL (compiler only, NOT a driver)

> ⚠️ Never install an NVIDIA *driver* inside WSL. The Windows driver (592.82) already provides the GPU to Linux. Install only the toolkit, from NVIDIA's **wsl-ubuntu** repository.

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
apt-cache search cuda-toolkit-12 | sort      # pick a 12.x version from this list
sudo apt-get install -y cuda-toolkit-12-8    # example; use one the list shows
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
nvcc --version
sudo apt-get install -y build-essential ninja-build ccache gdb
```

### 0.3 Python environment (uv is already installed)

```bash
cd ~/code-wsl/flashference      # already a git repo, branch v2
printf "models/\nllama.cpp/\nref/\n.venv/\n__pycache__/\n" > .gitignore
uv venv .venv && source .venv/bin/activate
uv pip install torch transformers accelerate datasets numpy matplotlib gguf huggingface_hub
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 0.4 Model and reference repo

```bash
cd ~/code-wsl/flashference
hf download bartowski/Qwen_Qwen3.5-0.8B-GGUF Qwen_Qwen3.5-0.8B-bf16.gguf --local-dir models
#   (older hub versions: huggingface-cli download ... same arguments)
sha256sum models/Qwen_Qwen3.5-0.8B-bf16.gguf
#   flashference measured AD1549EEDC613064971DCBBBFAB6C9B7990984D1C9AB38F792C6F2EC1207BBC2
#   if yours differs, write that in NOTEBOOK.md; upstream files can be re-uploaded
git worktree add --detach ref/flashference v1.0    # v1, read-only
```

### 0.5 Build stock llama.cpp at the same release flashference used

```bash
cd ~/code-wsl/flashference
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b11884
git rev-parse --short HEAD          # expect 63412d873
git switch -c stock
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4 --target llama-server llama-cli llama-bench llama-tokenize
```

`-DCMAKE_CUDA_ARCHITECTURES=89` matches your GPU (`nvidia-smi --query-gpu=compute_cap --format=csv` says 8.9), which makes the build faster. `RelWithDebInfo` keeps debug symbols for Phase 5. `-j 4` keeps memory use down. Raise it if the build is stable.

### 0.6 Smoke test

```bash
./build/bin/llama-bench -m ../models/Qwen_Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa on -p 0 -n 128
```

Write the tok/s in your notebook. On Windows, flashference measured about 71 tok/s with llama-bench. **WSL2 may differ.** That's fine: from now on, WSL is your platform and you compare only against WSL controls.

### Gate 0
- [ ] `nvidia-smi`, `nvcc --version` and `torch.cuda.is_available()` all work in WSL.
- [ ] llama-bench runs the bf16 model on the GPU. You've written the number down, along with the model's sha256.
- [ ] `~/code-wsl/flashference` is a git repo with your first notebook entry committed.

---

## Phase 1: How an LLM produces one token (about 2 weeks)

**Goal:** understand what happens between "prompt in" and "next token out", and where the model's bytes live, because speed is mostly about bytes.

### Concepts to learn (watch or read first; see Appendix E)
- Tokens and vocabulary; embeddings; transformer layers; final norm; **LM head**; **logits**; argmax and sampling.
- The **KV cache** (attention layers remember past keys and values).
- **Recurrent layers.** Qwen3.5 is a *hybrid*: some layers are attention, others are **Gated DeltaNet** (a linear-attention/recurrent layer that keeps a fixed-size state instead of a growing cache).
- **Tied embeddings:** the input embedding table and the output LM head are the same matrix.
- **MTP (multi-token prediction) head:** a small extra block that guesses the token after next from the model's hidden state.

Suggested order: Karpathy's "Let's build GPT" video (watch it, code along), *The Illustrated Transformer*, then the Gated DeltaNet paper's introduction only.

### Exercises (in `~/code-wsl/flashference/toy/`, write them yourself; ask AI only for API questions)

Use two small models that share a tokenizer: **`HuggingFaceTB/SmolLM2-360M-Instruct`** (the "target") and **`HuggingFaceTB/SmolLM2-135M-Instruct`** (the future "draft"). They're small enough to experiment with freely.

**1.1 Token round-trip.** Tokenize a sentence, print the ids, decode each id separately. Try code, emoji and Hindi. Notice how unevenly different text splits into tokens.

**1.2 Generation by hand.**
1. Write `greedy(model, ids, n)` that runs the model, takes `logits[0, -1].argmax()`, appends the token and repeats, **without a KV cache** (re-run the whole sequence each step).
2. Write a second version that passes `past_key_values` so only the new token runs each step.
3. Check that both versions produce exactly the same tokens as `model.generate(do_sample=False, max_new_tokens=n)`.
4. Time all three. Explain in your notebook why the cached version is faster.

**1.3 Where are the parameters?**
- Loop over `model.named_parameters()` and sum the bytes by group: embedding, attention, MLP, norms, lm_head.
- Check `model.config.tie_word_embeddings`.
- What fraction of all bytes is the embedding/head?

**1.4 Look inside the real model (Qwen3.5-0.8B GGUF):**

```python
from gguf import GGUFReader
r = GGUFReader("models/Qwen_Qwen3.5-0.8B-bf16.gguf")
for k in r.fields: print(k)                      # metadata keys
for t in r.tensors: print(t.name, list(t.shape), t.tensor_type.name, t.n_bytes)
```

Group tensors by name prefix (`token_embd`, `blk.0` … `blk.23`, `blk.24` = the MTP/"nextn" block, `output_norm`). Answer these in your notebook:
- How many bytes in total? How many in `token_embd`, which is also the output head?
- What's the vocabulary size and the embedding width? (Expect about 248k and 1024.)
- Which blocks are attention and which are recurrent? (Look at the tensor names inside each block.)
- How big is the MTP block?

**1.5 Sampling by hand.** Implement temperature and top-p sampling from a logits vector. Generate 5 samples at T = 0.7 and compare them with greedy output.

### Gate 1 (answer in your own words in NOTEBOOK.md, *then* ask an AI to grade you; see Appendix A)
1. What is a logit? What does the LM head compute, and why does its cost grow with vocabulary size?
2. Why can't the model produce token t+1 before token t? (That's the bottleneck speculative decoding attacks.)
3. What does a KV cache store? What does a recurrent layer store instead? Why is "undo the last 3 tokens" easy for a KV cache and hard for a recurrent state?
4. From your 1.4 output: what fraction of Qwen3.5-0.8B's bytes is the head? (flashference claims about one third. Do you agree?)
5. Why did tied embeddings plus a 248k vocabulary make flashference's MTP draft expensive?

---

## Phase 2: Why speed = bytes moved (about 1–2 weeks)

**Goal:** be able to *predict* decode speed from first principles. This is the most important mental model in the whole plan.

### Concepts
- **Memory bandwidth** (GB/s) versus **compute** (FLOP/s). **Arithmetic intensity** and the **roofline** model.
- **GEMV** (matrix × vector, one token) versus **GEMM** (matrix × matrix, several tokens).
- **Kernel launch overhead**, and **CUDA graphs** (record many launches once, replay them cheaply).
- **bf16 / fp16 / fp32** formats; **floating-point addition is not associative**.

Read Horace He's *Making Deep Learning Go Brrrr From First Principles* and kipply's *Transformer Inference Arithmetic*.

### Exercises (PyTorch on your GPU; time with `torch.cuda.Event(enable_timing=True)`, always warm up first, and average many repeats)

**2.1 Measure your memory bandwidth.**
- Copy a 1 GiB tensor to another one (`dst.copy_(src)`). Bandwidth = 2 × bytes / time, because a copy reads and writes.
- `nvidia-smi` reports a max memory clock of 8001 MHz but not the bus width. Look up your exact GPU's bus width, compute the theoretical peak, and compare it with your measurement.

**2.2 GEMV size sweep.**
- For `rows` from 1,000 to 248,320, time `W @ h` with `W = randn(rows, 1024, dtype=bfloat16, device="cuda")` and `h = randn(1024)`.
- Plot time against bytes of W. Fit `t = a + b·bytes`: `a` is the fixed overhead and `1/b` is the effective bandwidth.
- Predict how long the 485 MiB Qwen3.5 head GEMV takes. Compare that with "about 1/3 of a 14.7 ms token ≈ 4.9 ms".

**2.3 Batch sweep, the key to speculative decoding.**
- Time `W @ H` for the full 248,320 × 1024 matrix with H having 1, 2, 3, 4, 8 and 16 columns.
- Plot it. Explain why checking 3 tokens costs almost the same as checking 1, and where that stops being true.

**2.4 Launch overhead.**
- Time 1000 tiny ops (e.g. `x.add_(1)` on a small tensor) in a Python loop.
- Then capture the same ops in a `torch.cuda.CUDAGraph` and replay. Compute the overhead per launch in microseconds.
- A 0.8B model runs hundreds of kernels per token. Estimate the overhead per token.

**2.5 Floating point.**
- In numpy, sum the same 4096 random numbers in three different orders, in float16 and float32. Print the differences.
- Build two "logits" 1e-3 apart and show that a different summation order can flip which one is the argmax.
- That's why flashference's speculative output isn't always byte-identical to plain decoding.

### Gate 2
1. **Prediction (write it now and keep it):** from Qwen3.5-0.8B's bytes per token (Phase 1.4), your measured bandwidth (2.1) and your overhead estimate (2.4), predict plain bf16 decode tok/s. You'll check this in Phase 4.
2. Why does a 4-bit model decode faster than bf16, but not 4× faster?
3. Why is verifying several tokens at once nearly free on this GPU?
4. Why can two mathematically identical computations disagree on the argmax?

---

## Phase 3: Speculative decoding from scratch (about 2–3 weeks)

**Goal:** implement flashference's three core ideas yourself in about 300 lines of Python you fully understand: speculative decoding, the n-gram drafter, and the trimmed draft head. After this phase the flashference README should read as obvious.

Read first: Leviathan et al. 2023 (*Fast Inference from Transformers via Speculative Decoding*), sections 1–3, and the README of *Prompt Lookup Decoding* (apoorvumang on GitHub).

### Exercises (`~/code-wsl/flashference/toy/spec.py`)

**3.1 Baseline.** Reuse `greedy()` from 1.2 with the 360M target. Use **fp32 on the CPU** for the correctness tests, so numerics can't confuse you yet.

**3.2 Greedy speculative decoding.** Each cycle:
1. The draft model proposes γ tokens greedily.
2. The target runs **once** on `[last_token, d1, …, dγ]`.
3. Compare the target's argmax at each position with the draft. Accept the longest matching prefix.
4. Append the target's own token at the first mismatch. If everything matched, append the **bonus token**.
5. Roll back both models' caches past the rejected positions (`DynamicCache.crop(n)`). Or skip caches in v1 and re-run everything; it's slower, but simpler to get right.

Requirements:
- **Test:** on 20 prompts, the output must equal `greedy(target)` **exactly** (fp32, CPU). Also compare with transformers' built-in `generate(assistant_model=draft)`.
- **Instrument:** drafted tokens, accepted tokens, cycles, tokens per cycle, and acceptance by position.
- Now run in **bf16 on the GPU** and count the mismatches against plain bf16 greedy. You have just reproduced flashference's "not byte-identical" caveat. Explain it using 2.5.

**3.3 The cost model.**
- Measure `T_target(1)`, `T_target(γ+1)` and `T_draft`.
- Predict speedup = `tokens_per_cycle × T_target(1) / (T_target(γ+1) + γ·T_draft)`, then measure it.
- Derive `E[tokens/cycle] = (1 − α^(γ+1)) / (1 − α)` for a per-token acceptance α and check it against your data.
- Worked example from flashference (B3 report): plain 14.7 ms per token, one cycle 21.9 ms, 1.62 tokens per cycle → 14.7 / (21.9/1.62) ≈ 1.09×. Redo that arithmetic yourself.

**3.4 The n-gram (prompt-lookup) drafter.**
- Find the most recent earlier occurrence of the last n tokens in the context and propose the m tokens that followed it.
- Compare acceptance and tokens per cycle on a "fix the typos in this paragraph: …" prompt against a "write a short story" prompt.
- Chain the drafters: n-gram first, draft model if the n-gram finds nothing. Measure on both prompts. Explain why chaining helps copy-heavy text and can hurt prose.

**3.5 The trimmed draft head, flashference's main trick.**
- Build a token-frequency ranking from a corpus. Prose: WikiText-2 (`datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")`). Code: the Python standard library on your disk (`/usr/lib/python3.12/**/*.py`).
- Keep only the top-K rows of the draft's lm_head. The draft picks its argmax among those K and maps the index back to a token id.
- Sweep K over {1k, 4k, 8k, 16k, 32k, full 49k} on prose and code prompts. Plot acceptance against K.
- **Check that the output still equals the target's greedy output.** It must, because the target is untouched.

**3.6 (Optional) Speculative *sampling*.**
- Implement the rule "accept with probability min(1, p/q), otherwise resample from max(0, p − q), normalized".
- On a toy 10-token distribution, draw 10,000 samples and show with a histogram that the output matches the target's distribution. That's L1 losslessness.

### Gate 3
1. In 5 sentences: why does greedy speculative decoding return the target's own output? What's the one way it can still differ?
2. Why is the bonus token free?
3. Why does trimming the *draft* head keep the output unchanged, while trimming the *target* head would not?
4. Compute without code first, then check with a script: α = 0.5, γ = 2, T_draft = 1.3 ms, T_verify(3) = 15.5 ms, T_plain = 14.7 ms. What's the speedup?
   <details><summary>answer</summary>E[tokens/cycle] = (1 − 0.5³)/(1 − 0.5) = 1.75; cycle = 15.5 + 2 × 1.3 = 18.1 ms; 18.1/1.75 = 10.34 ms per token; 14.7/10.34 ≈ 1.42×.</details>
5. Why did n-gram + MTP make flashference *slower* on prose (0.83× of MTP alone)?
6. **The big one:** rewrite flashference's README "How it works" section in your own words in NOTEBOOK.md. For every number in its headline tables, say which experiment produced it.

---

## Phase 4: Measuring honestly, with your own harness (about 1–2 weeks)

**Goal:** a small benchmark harness you wrote and understand, plus a known **noise floor**. flashference's harness exists, but you'll write your own and use theirs only as a reference to compare against.

### Concepts
Warm-up; drift; same-session controls; bracketing (control, A, B, …, control); interleaving (A B A B); mean and standard deviation; uncertainty of a ratio; **A/A tests**; dev/test splits; provenance (record exactly which binary, flags and model produced each number).

### Write your prompt sets first (`~/code-wsl/flashference/prompts/`)
Four domains: **prose**, **code** (Python, JS, SQL, …), **copy-heavy** (edit or rewrite a provided text), **multilingual** (es, de, zh, hi). For each domain write **10 dev and 10 test prompts** yourself. Commit the test set and don't look at its results until a final report.

Use the chat format flashference used, with thinking turned off:
`<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`

### Spec for `bench/bench.py` (you write it; AI may review it)
- **Input:** a JSON config list. Each entry has: name, server binary path, args, env vars, prompt set, `n_predict` (256), passes (3), warm-up (1).
- **Before each config, check the GPU is idle:** `nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits`. Wait or abort if utilization is above 5 % or memory used is above about 500 MiB.
- **Start a fresh `llama-server`** for each config: `-m … -ngl 99 -fa on -np 1 -kvu -c 4096 --port …` plus the config's own args and env. Poll `/health`.
- **For each prompt:** POST `/completion` with `{"prompt": …, "n_predict": 256, "temperature": 0, "cache_prompt": false}`.
- **Record:** `timings` (`predicted_n`, `predicted_ms`, `prompt_ms`, `draft_n`, `draft_n_accepted`; check the exact names in your server's response), the sha256 of `content`, nvidia-smi readings before and after, a timestamp, the llama.cpp git commit, the full command line and env.
- **Shut down the server.** Append one JSON line per request to `results/<session>.jsonl`.

### Spec for `bench/report.py`
- tok/s per pass = Σ tokens / Σ predicted_ms over the prompt set. Report the mean ± SD over passes.
- Ratio against a named control **from the same session**: `r = a/b`, `σ_r = r·√((σ_a/a)² + (σ_b/b)²)`.
- Acceptance = Σ draft_n_accepted / Σ draft_n.
- Output equality = the fraction of (prompt × pass) hashes equal to a named reference config.
- It writes a markdown table. You never edit numbers by hand.

### Exercises
- **4.1 A/A test.** Run the plain config 3 times in one session as 3 "different" configs. The spread between them is your **noise floor**. Any effect smaller than about 2× that is noise.
- **4.2 Check your Phase 2 prediction.** Measure plain bf16 tok/s. How far off was your prediction? Write down why.
- **4.3 Your first real effect.** flashference found `-bs` (backend sampling) gives about +4 %. Predict, then measure plain against plain + `-bs`, bracketed by controls.

### Gate 4
1. What's your noise floor, as a percentage?
2. Why a same-session control? Why bracket?
3. Show a report table generated only by `report.py` from raw JSONL.
4. Why must the test prompts stay untouched until the end?

---

## Phase 5: llama.cpp, build, run and read (about 2–3 weeks)

**Goal:** you can find, read and modify the code path of one speculative cycle.

### Concepts
- **C++ reading basics:** structs, references and pointers, `std::vector`, and lambdas, roughly to the level of learncpp.com chapters 1–12 and 16–17. You need to *read* C++, not master it.
- **ggml:** tensors (`ne` = shape, `nb` = strides, `type`), *lazy* graphs (you build a graph of ops, then compute it), backends and buffers, the scheduler. Read Hugging Face's blog post "Introduction to ggml".
- **llama.cpp layers:** model loading → context → batch → `llama_decode` → graph built per architecture → ggml ops → CUDA kernels. Server slots → speculative drafting → verification → accept/rollback.

### Guided tour (always grep first, then read)
These names come from flashference's patches, so they exist in b11884:

```bash
cd ~/code-wsl/flashference/llama.cpp
grep -rn "common_speculative" tools/server | head -30        # where the server calls the drafter
grep -n  "struct common_speculative_impl" common/speculative.cpp   # each drafter type; find draft_mtp and the n-gram ones
grep -n  "graph_mtp\|result_output" src/models/qwen35.cpp   # target head and MTP head graphs
grep -n  "load_tensors" src/llama-model.cpp | head           # where weights are placed on the GPU
grep -rn "seq_rm" tools/server common | head                 # rollback of rejected drafts
grep -n  "ggml_cuda_mul_mat\b" ggml/src/ggml-cuda/ggml-cuda.cu   # how the matmul kernel is chosen by batch size
```

### Exercises
- **5.1 Log one cycle.** Add `LLAMA_LOG_INFO` lines that print (a) the drafted token ids in the MTP drafter and (b) how many were accepted per cycle in the server. Rebuild, send one request, read the log. Then put the logging behind an env var so it costs nothing when off.
- **5.2 Follow the calls.** Run `gdb --args ./build/bin/llama-server …`, set `break llama_decode`, run, send a request, and print the call stack with `bt`. Write the chain from the HTTP request down to `llama_decode`.
- **5.3 Draw one speculative cycle** (on paper or as a mermaid diagram): which graphs run on the GPU (target verify, MTP draft × n), where the host waits for the GPU (syncs), and where tokens are accepted and state rolled back.
- **5.4 Find the kernel choice.** In `ggml_cuda_mul_mat`, find how the code picks a kernel depending on the number of tokens in the batch. Write which kernel runs for 1 token and which for 3. (This is the root of the L2-versus-L3 difference and matters for 8.7.)

### Gate 5
1. Where does verification happen, and where is the number of accepted tokens decided? Give file and function.
2. What happens to the recurrent (DeltaNet) state when drafts are rejected? Point to the code.
3. Annotate **every hunk** of `ref/flashference/patches/b4-trim-draft-head.patch` in your notebook. In particular: why does it `ggml_pad` the K logits back to the full vocabulary and add a `-inf` mask? What would break without that?

---

## Phase 6: Rebuild flashference yourself and reproduce it in WSL (about 2 weeks)

**Goal:** write the trimmed-draft-head patch yourself, then reproduce the headline ratios on your platform with your harness.

### 6.1 Rewrite patch 1 yourself
1. `git switch stock && git switch -c my-trim`.
2. Write a half-page **design** in NOTEBOOK.md first: which tensor is gathered, when, where it lives, how the draft's ids map back, and why the target is unaffected.
3. Implement in 4 steps, testing after each one (AI as reviewer only):
   1. Read `LLAMA_MTP_TRIM_FILE` / `LLAMA_MTP_TRIM_K`, load the ids and log K.
   2. Copy the K rows of the head into a host buffer, upload them to a new GPU tensor, and log the size in MiB.
   3. Use that tensor in `graph_mtp` (`src/models/qwen35.cpp`).
   4. Map drafted ids back to token ids in `common/speculative.cpp`.
4. **Test:** on your dev prompts, trimmed output hashes must equal **untrimmed MTP** output hashes, and the speed must go up.
5. `git diff stock my-trim` against flashference's patch: explain every difference.

### 6.2 Make your own ranking file
Tokenize your prose corpus with `llama-tokenize` (read `--help` for how to print ids only). Count, sort, and add the never-seen ids at the end. Then:
- Compare your ranking's top 100 with flashference's `results/2026-10-26-bf16-b4-token-rank.txt`.
- **Check** that the tokenizer you used gives exactly the GGUF's ids on 20 test strings. A silent tokenizer mismatch would ruin the ranking.

### 6.3 Reproduce (dev prompts, your harness, one session, bracketed controls)

| Config | flashference ratio (Windows) | Your ratio (WSL) |
|---|---|---|
| plain | 1.00 | |
| plain `-bs` | about 1.04 | |
| MTP untrimmed n-max 2 `-bs` | prose 1.15, code 1.41 | |
| trim 8k n-max 2 `-bs` | prose about 1.6, code 1.48 | |
| trim 32k n-max 2 `-bs` | prose 1.60, code 1.88 | |
| ngram-simple only (N4, M8) | copy-heavy 1.62 | |
| ngram-simple + MTP trim 8k, n-max 8 | copy-heavy 2.97, prose 1.37 | |

Absolute tok/s will differ (WSL2 versus Windows). **If a ratio differs by more than about 10 %, investigate before going on.** Possible causes: WSL overhead, different prompts, a bug in your patch.

### 6.4 An ablation flashference never ran
Its copy-heavy headline used patch 2's `LLAMA_MTP_ADAPT=1`, which was "not separately ablated". Build a second branch with flashference's two patches (`ff-both`) and measure copy-heavy with and without `LLAMA_MTP_ADAPT`. That's your first new finding.

### Gate 6
- [ ] You could rewrite patch 1 from a blank b11884 checkout in an afternoon, without AI writing the code.
- [ ] The reproduction table exists, generated by `report.py`, and you can explain every row.
- [ ] You know whether the ADAPT setting matters for copy-heavy text.

---

## Phase 7: Profile: where do the ~15 ms per token go? (about 1 week)

**Goal:** a **per-token time budget**. It decides which experiments in Phase 8 are worth doing.

### Setup
Nsight Systems (`nsys`) may come with the CUDA toolkit (`ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems*`). If not, download it for Linux from developer.nvidia.com/nsight-systems. If CUDA tracing comes back empty under WSL2, use the Windows version of Nsight Systems with a native Windows build instead.

```bash
nsys profile -o prof/plain --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
     --delay 15 --duration 10 ./build/bin/llama-server -m ../models/... -ngl 99 -fa on -np 1 -kvu -c 4096
#   during the capture window, have bench.py send a few requests
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum prof/plain.nsys-rep
```

`--cuda-graph-trace=node` matters. Without it, CUDA graphs hide the individual kernels.

### Tasks
1. **Plain-decode budget.** In the timeline, find one token's kernels: the per-layer GEMVs, the DeltaNet ops, attention, the norms, and the **head GEMV** (the big one at the end), then sampling. Sum by group.
2. **Effective bandwidth per GEMV** = weight bytes / kernel time, compared with your Phase 2.1 bandwidth. Is the head GEMV near the hardware limit?
3. **Idle gaps.** How much time per token is the GPU doing nothing? Do the same for one speculative cycle (best code config): MTP draft 1 → MTP draft 2 → verify, plus the gaps between them.
4. **Graph reuse.** Count `cudaStreamBeginCapture` (a new graph) against `cudaGraphLaunch` (a replay) in `cuda_api_sum`, for plain, MTP, and ngram + MTP.
5. **Predict first, then check:** how much does the `ggml_pad` + mask in patch 1 cost per draft token?

### Deliverable: the budget table (in NOTEBOOK.md)

| Component | ms per token (plain) | ms per cycle (spec) | % | Achieved GB/s |
|---|---|---|---|---|
| body GEMVs | | | | |
| head GEMV | | | | |
| DeltaNet / attention / other | | | | |
| MTP draft (× n) | | | | |
| sampling | | | | |
| idle gaps | | | | |
| **total (must match measured ms/token within 5 %)** | | | | |

### Gate 7: decisions
- If GEMVs already run at **80 % or more** of measured bandwidth → bytes are the limit. 8.3/8.4 (fewer head bytes) and 8.9 (compression) are promising.
- If GEMVs are well below the limit → something else limits speed (launch overhead, small kernels). 8.9 won't help; 8.6 might.
- If idle gaps plus graph re-captures are **5 % or more** of a cycle → 8.6 is worth doing. Otherwise skip it.

---

## Phase 8: The experiments

Each experiment uses this template in its own report (`results/8.x-name.md`, see Appendix B):
**Question · Learn first · Prediction (written before running) · Steps · Measurement · Done when · Stop if · Losslessness level and how checked · What I did not test.**

Run everything on **dev** prompts. Confirm the final number once on **test** prompts.

---

### 8.1 Quantize the drafter, then spend the savings on a bigger K

**Question.** The drafter's numbers never reach the output (the target verifies everything). Can a Q8_0/Q4_0 drafter be faster at the same acceptance, or allow a larger K and higher acceptance for code and multilingual text?

**Learn first.**
- Quantization formats: Q8_0 = blocks of 32 int8 values plus one fp16 scale = 8.5 bits per weight. Q4_0 = 4.5 bits.
- Why a lossy drafter can't break L2 losslessness.
- Read `ggml_quantize_chunk` and `ggml_bf16_to_fp32_row` in `ggml/include/ggml.h`.

**Prediction.** From the Phase 7 budget: draft head bytes at K = 32k bf16 = 32768 × 1024 × 2 = 64 MiB. What does that cost per draft token at your bandwidth, and what at Q8 and Q4? What's the *largest* K whose Q4 head costs no more than the 32k bf16 head?

**Steps.**
1. On branch `exp-8.1`, from `my-trim`: after gathering the K rows, convert them to f32, quantize with `ggml_quantize_chunk` to an env-chosen type (`LLAMA_MTP_TRIM_TYPE=bf16|q8_0|q4_0`), and create the GPU tensor with that type.
2. Separately, quantize only the MTP block (`blk.24.*`) in a new GGUF. Look for `--tensor-type` in `llama-quantize --help`. Never touch `token_embd`, because it's also the target's head.
   **Losslessness check:** with `gguf`, hash every non-`blk.24` tensor in both files. All must be identical.
3. Sweep K ∈ {8k, 16k, 32k, 64k, 128k} × type ∈ {bf16, q8_0, q4_0} × domain. Record tok/s and acceptance.

**Done when.** You have a table of the best config per domain, confirmed on test prompts.
**Stop if.** No config beats the 32k bf16 baseline by more than 2 × your noise floor.
**Level:** L2. Outputs must equal the reference (same K, bf16) up to the batch-numerics noise you measured in Phase 6.

---

### 8.2 Better draft vocabularies

**Question.** (a) Does the dynamic-rows trick (patch 2's E4) on top of a 32k head beat 125 tok/s on code? (b) Does a ranking built from the *model's own drafts* beat a WikiText frequency ranking?

**Learn first.** Read flashference patch 2 the way you read patch 1 in Gate 5, annotating every hunk. Also read the FR-Spec paper (frequency-ranked draft vocabulary).

**Steps.**
1. **Coverage tool, which is the key idea.** With your 5.1 logging, record the **untrimmed** MTP draft's argmax for every draft step over many dev prompts across all domains. For each candidate ranking, coverage(K) = the fraction of those argmax tokens inside the top K. Coverage is what acceptance depends on, and it's computed offline in seconds.
2. Build the rankings: (i) WikiText (flashference's), (ii) model drafts on mixed domains, (iii) a mix. Plot coverage(K) for each.
3. **Validate the tool:** does coverage predict measured acceptance? Plot one against the other for 4–5 configs. If it doesn't, find out why before trusting it.
4. Measure the best ranking at K = 8k/16k/32k, and the 32k head + dynamic rows on code (needs patch 2's logic on your branch; port it yourself).

**Stop if.** Coverage doesn't predict acceptance and you can't explain why. Fix the tool first.
**Level:** L2.

---

### 8.3 Offline study: a mixed-precision *certified* target head (no llama.cpp changes)

**Question.** The target head reads 485 MiB every cycle. Can we read far fewer bytes and still get the **exact** bf16 argmax, proved by a bound rather than hoped for?

**Learn first.**
- The Cauchy–Schwarz inequality: |h·e − h·q| ≤ ‖h‖·‖e − q‖.
- Maximum inner product search (MIPS); k-means clustering.
- Why a fixed "safety margin" is unsafe: flashference measured that 8.5 % of top-1/top-2 gaps are below 0.2.
- Read flashference's E5 section in `results/2026-10-27-bf16-E1-E5.md` and **then** write your own analysis. Don't start from their script.

**Data collection.**
1. Run `llama-server --embeddings --pooling none` and send prompt + generated text. You get one final hidden state h per position. (You need to confirm what these vectors are; see the sanity check.)
2. Load the head matrix E (248,320 × 1024) from the GGUF with `gguf`.
3. **Sanity check (gate):** argmax(E·h) must equal the generated token in more than 99 % of positions. If not, you have the wrong vectors. Stop and investigate.
4. Collect at least 5,000 positions across all 4 domains.

**The bound to test.**
- Split the vocabulary into **hot** rows (the top 16k/32k by your 8.2 ranking), scored exactly in bf16, and **cold** rows, scored only by an *upper bound*:
  - (i) the norm bound ‖h‖·‖e_i‖, which needs no row bytes at all;
  - (ii) Q4/Q8 rows: h·q_i + ‖h‖·ε_i, where ε_i = ‖e_i − q_i‖;
  - (iii) clusters: k-means the cold rows into 1–8k clusters with centroid c_j and radius r_j; bound = h·c_j + ‖h‖·r_j. Expand (score exactly) only the clusters whose bound beats the best hot logit.
- A position is **certified** if the best exact hot logit ≥ every cold upper bound.

**Prediction.** Bytes for hot 32k bf16 + cold Q4 ≈ 64 + 119 = 183 MiB (0.38× of 485). What certification rate do you expect, and why?

**Measure.** For each variant: certified %, mean bytes read (counting the expanded clusters and the fallback cost when not certified), and how both change with the hot-set size.

**Done when.** You have a table of variant → certified % → expected bytes. If some variant certifies at least 99 % with at most 0.5× the bytes, go to 8.4.
**Stop if.** Nothing beats flashference's measured Q8 + top-16 regather (100 % certified at 0.53× the bytes). Then use that design in 8.4 instead.
**Level:** this is an *analysis*. The question it answers is "can L2 be kept with fewer bytes?"

---

### 8.4 Certified head inside llama.cpp (only after 8.3 is positive)

**Question.** Does the 8.3 design give a real end-to-end speed-up while keeping exactly the same argmax?

**Learn first.** How the target's output graph is built (`result_output` in qwen35.cpp); how `-bs` greedy sampling consumes logits; ggml ops (`mul_mat`, `get_rows`, `argmax`, `top_k`, `sqr`, `sum_rows`, `sqrt`).

**A design trick worth reusing:** patch 1 already pads K logits to the full vocabulary with `-inf`. Do the same with the exact hot logits, so the existing greedy sampler runs unchanged. Then output one extra 0/1 "certified" flag per position, which the host reads with the token ids it already reads each cycle, so no extra GPU sync is needed.

**Steps, staged so correctness comes first.**
1. **Shadow mode:** compute *both* the certified token and the full-head argmax, assert they're equal, and log the flag. No speed-up is expected. Run all dev prompts, then 10× more. Count the failures.
2. **Cheap path only:** remove the full head. When a position isn't certified, log it and stop generating (no fallback yet). Measure speed on prompts where the flag never fails.
3. **Real fallback:** when a position isn't certified, accept the tokens before it, then compute that one position with the full head (a separate small graph, or a forced full-head decode).
4. Measure end-to-end against your best config, all domains, then confirm on the test prompts.

**Done when.** The certified head is faster than the bf16 head by more than 2 × your noise floor, and shadow mode had **zero** mismatches.
**Stop if.** Shadow mode ever shows a mismatch you can't explain. A lossless claim with one unexplained mismatch is not lossless.
**Level:** L2, and the argmax is *proved* equal to the full head's argmax in the same batch.

---

### 8.5 Drafter arbitration: one config for every workload

**Question.** flashference's chain uses "first non-empty draft wins", which costs 0.83× on prose. Can per-request gating keep the copy-heavy gain *and* lose nothing on prose?

**Learn first.** Find the chain logic in `common/speculative.cpp` (how a `--spec-type a,b` list is tried in order). Read about exponential moving averages (EMA).

**Steps.** Keep a per-sequence EMA of the n-gram drafter's acceptance. Skip the n-gram drafter while the EMA is below τ, and re-try it every M cycles. Optionally also require a minimum match length. Sweep τ and M on dev.

**Done when.** One config reaches at least 0.98× of MTP-only on prose and at least 0.95× of ngram + MTP on copy-heavy (test prompts).
**Level:** L2 (only the draft changes). Outputs must equal the reference within the measured numerics noise.

---

### 8.6 Fewer host round-trips and stable graph shapes (only if Gate 7 says so)

**Question.** Can removing GPU idle gaps and CUDA-graph re-captures shorten each cycle?

**Steps.**
1. From Phase 7, list each gap and its cause: the host waiting for a draft token, a re-capture caused by a new batch shape, and so on.
2. Fix the largest one only. For example, run both MTP draft steps in one graph, feeding the first step's argmax into the second without going through the host, or keep the verify batch size fixed.
3. Re-profile, then re-measure.

**Level:** L2. Output hashes should equal the reference exactly, since you're not changing any maths.

---

### 8.7 Bit-exact speculative decoding (L3)

**Question.** Can speculative output be **byte-identical** to plain greedy decoding, every time?

**Learn first.** Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025), on batch invariance. Your 2.5 and 5.4 notes.

**Steps.**
1. **Minimal reproducer:** a prompt where plain and speculative outputs differ. Find the first differing token position.
2. **Find the op:** dump intermediate tensors for that position computed with batch size 1 versus 3. Build `llama-eval-callback`, or write a small `cb_eval` callback. Find the **first** op whose output differs. Suspects: the matmul kernel choice (5.4), the chunked versus single-step DeltaNet path, flash attention with different query counts.
3. **Make that op batch-invariant:** compute each token the same way regardless of how many tokens share the batch (e.g. use the single-token kernel for every column). Repeat until no op differs.
4. Measure (a) byte-equality with plain on all prompts in all domains (target: 100 %) and (b) the speed cost.

**Done when.** 100 % byte-identical on test prompts, with the speed cost documented. Even a slower bit-exact mode is a publishable result.
**Level:** L3.

---

### 8.8 Self-distilled MTP head (research-grade; after 8.1 and 8.2)

**Question.** Can fine-tuning the MTP block on the model's own outputs (with the trimmed K-row head built in) raise acceptance well above 48 % on prose?

**Learn first.** PyTorch training loops; KL divergence; the DeepSeek-V3 technical report's MTP section; EAGLE (Li et al.) for the idea of training a draft head on the target's features.

**Gates, in order.**
1. **Re-implement the MTP forward pass in PyTorch** from the GGUF weights. It must reproduce llama.cpp's draft logits on the same inputs (max absolute difference about 1e-2 in bf16). Don't train anything before this passes.
2. Collect training data: target hidden states plus the target's next tokens on many *dev-domain* prompts (not test).
3. Train only the MTP block (it's small and fits in 4 GB). Evaluate the **offline** acceptance proxy from 8.2.
4. Write the weights back into a GGUF (with the `gguf` writer). Hash every non-MTP tensor to prove the target is unchanged.
5. Measure end to end.

**Level:** L2. The target is unchanged, which the tensor hashes prove.

---

### 8.9 Lossless weight compression (only if Gate 7 shows GEMVs are bandwidth-bound)

**Question.** bf16 weights can be compressed *without any loss*, e.g. DFloat11-style Huffman coding of the exponent bits gives about 70 % of the size. Would reading fewer bytes beat the decompression cost at batch 1?

**Steps.**
1. **One evening, offline:** compute the entropy of the 8 exponent bits over all of the model's bf16 weights. That gives the best possible compression ratio. If it's above about 85 %, stop.
2. **Standalone CUDA prototype,** outside llama.cpp: a fused decompress + GEMV kernel against a plain bf16 GEMV, on the head matrix alone. Only if it wins clearly, consider integrating it.

**Level:** L3-capable. The decompressed weights are bit-identical. (Output is identical only if the kernel's reduction order also matches.)

---

### 8.10 Portability

**Questions.**
- Do the ratios hold on a bigger Qwen3.5 model? Check in its GGUF that it has `nextn` tensors. With 4 GB of VRAM, use a Q8_0 target; losslessness is then relative to the Q8 target.
- Does the head trim matter less as the head becomes a smaller share of the model? Predict this from Phase 1.4-style byte counts first.
- On another GPU (borrow one, or rent an hour): same harness, same prompts, ratios only.

---

## Phase 9: Writing it up, and auditing flashference

1. **One report per experiment** (Appendix B template), linked to its raw JSONL and the llama.cpp commit.
2. **Claims ledger:** a table of every claim → evidence file → level (L1/L2/L3) → test-set confirmed yes/no.
3. **Reproduction script:** from a clean clone, one command rebuilds and re-runs the headline numbers. Test it on a fresh directory.
4. **Independent check:** give only your README to someone else (or an AI in a fresh session with no history) and see whether they can reproduce one number.
5. **Audit flashference.** For each headline claim in its README, mark *reproduced by me in WSL* / *not reproduced* / *not tested*, and update its README honestly. That closes the loop on the original mistake.

---

## Appendix A: Working with AI without vibecoding

**Phases 1–4: AI is a tutor, not an author.** Useful prompts:
- "Explain {concept} to someone who knows basic Python but no C++ or CUDA. Then ask me 3 questions to check my understanding. Don't give the answers until I reply."
- "Here's my answer to this question: {answer}. Grade it. Tell me what's wrong or missing, but don't rewrite it."
- "I'm getting {error}. Explain what it means and where I should look. Don't fix the code."

**Phases 5–8: AI may write code, under these rules.**
1. **You write the spec first** (inputs, outputs, what must not change). AI writes code only to that spec.
2. **Small diffs:** one function, or 100 lines at most, at a time.
3. **You write the test first.** Example: "trimmed output hash == untrimmed output hash".
4. **Explain-back before commit:** close the chat and write 3–5 sentences on what the diff does and why it's lossless. If you can't, don't commit; ask until you can.
5. **Review prompt:** "Review this diff for bugs and for anything that could change the target model's output. List the problems; don't fix them."

**Red flags: stop if you see any of these.**
- A number in a report you didn't produce with your own harness.
- An estimate (`[est]`) quietly becoming a fact in a README.
- A diff you scrolled past instead of reading.
- "It works", with no test that would have failed if it didn't.
- Several changes measured together as one experiment.

Optional: if your Claude Code version has the built-in **Learning** output style (look in `/config`), it leaves small pieces of code for *you* to write. That suits Phases 3–6.

---

## Appendix B: Templates

### NOTEBOOK.md entry
```markdown
## 2026-11-12 — Phase 2.2 GEMV sweep
**Plan:** time W@h for rows 1k..248k.
**Prediction:** head GEMV ≈ __ ms because 485 MiB / __ GB/s + __ µs overhead.
**Result:** __ ms (results/2026-11-12-gemv.jsonl, plot analysis/gemv.png).
**Prediction error:** __ %. **Why:** ...
**What I now understand:** ...
**Open questions:** ...
```

### Prediction log (one line each, at the top of NOTEBOOK.md)
`P-007 | 2026-11-20 | 8.1 | code ratio 1.95× ± 0.05 because draft head bytes halve | got 1.91× | error −2 % | learned: ...`

### Experiment report (`results/8.x-name.md`)
```markdown
# 8.x <name>
Question:
Prediction (written <date>, before running):
Setup: llama.cpp commit, branch, flags, env, model sha256, prompt set (dev/test), passes
Results: (table generated by report.py; link to raw JSONL)
Losslessness: level L_, check: output hash equality vs <reference> = __/__
Noise floor this session: __ %
Conclusion (only what the data shows):
What I did not test:
Next step / decision:
```

---

## Appendix C: Formula sheet

| Quantity | Formula |
|---|---|
| Bytes per plain token | ≈ all weight bytes read once (tied head counted once) + KV/state traffic (small) |
| Achieved bandwidth | bytes per token / seconds per token |
| Time for a memory-bound op | a + bytes / bandwidth (a = fixed overhead) |
| Expected tokens per cycle | (1 − α^(γ+1)) / (1 − α), with α = per-token acceptance and γ = draft length |
| llama.cpp "acceptance" | draft_n_accepted / draft_n (not the same as α; know the difference) |
| Cycle time | T_verify(γ+1) + γ·T_draft (+ gaps) |
| Speedup | (tokens per cycle × T_plain) / cycle time |
| Ratio uncertainty | r = a/b, σ_r = r·√((σ_a/a)² + (σ_b/b)²) |
| Q8_0 bits per weight | 8 + 16/32 = 8.5 |
| Q4_0 bits per weight | 4 + 16/32 = 4.5 |
| Certified-head bound | h·e_i ≤ h·q_i + ‖h‖·‖e_i − q_i‖ (Cauchy–Schwarz) |
| Cluster bound | h·e_i ≤ h·c_j + ‖h‖·r_j for all i in cluster j |

---

## Appendix D: Glossary

- **Token / vocabulary:** the pieces text is split into; Qwen3.5 has about 248k.
- **Logit:** the score for each vocabulary token; the argmax is the greedy choice.
- **LM head:** the matrix (vocab × width) that turns the final hidden state into logits.
- **Tied embeddings:** the LM head is the same matrix as the input embedding table.
- **KV cache:** stored attention keys and values for past tokens.
- **Gated DeltaNet:** a recurrent/linear-attention layer with a fixed-size state; Qwen3.5 mixes these with attention layers.
- **MTP / nextn:** the model's built-in head that predicts the token after next; used as a drafter.
- **Drafter / target:** the drafter guesses tokens; the target model checks (verifies) them.
- **Acceptance:** how many drafted tokens the target agreed with.
- **Tokens per cycle:** output tokens per target forward pass.
- **Bonus token:** the target's own next token, which comes free with every verification.
- **n-gram / prompt-lookup drafter:** guesses by copying what followed the same recent tokens earlier in the context.
- **FR-Spec / trimmed draft head:** the drafter only considers the K most frequent tokens.
- **GEMV / GEMM:** matrix × vector (one token) / matrix × matrix (several tokens).
- **Roofline:** speed is limited by either bandwidth or compute, whichever runs out first.
- **Kernel:** one function that runs on the GPU.
- **CUDA graph:** a recorded sequence of kernels, replayed cheaply.
- **Backend sampling (`-bs`):** argmax on the GPU instead of copying 248k logits to the CPU.
- **ggml / GGUF:** llama.cpp's tensor library / its model file format.
- **Batch invariance:** each token's result doesn't depend on how many tokens share the batch. Needed for L3.
- **A/A test:** measuring the same thing twice to find the noise floor.
- **Drift:** machine speed changing over time (temperature, clocks, background load).

---

## Appendix E: Reading list (in order of use)

**Phase 1**
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube).
- Jay Alammar, *The Illustrated Transformer*.
- Yang, Kautz, Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule* (2024). Introduction only.

**Phase 2**
- Horace He, *Making Deep Learning Go Brrrr From First Principles*.
- kipply, *Transformer Inference Arithmetic*.

**Phase 3**
- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding* (2023).
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).
- *Prompt Lookup Decoding* (apoorvumang, GitHub).
- Xia et al., *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding* (2024). Skim it.

**Phase 5**
- learncpp.com, selected chapters.
- Hugging Face blog, *Introduction to ggml*.
- llama.cpp's `docs/` folder.

**Phase 7**
- NVIDIA Nsight Systems user guide: CUDA trace and CUDA graph trace sections.

**Phase 8**
- *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling* (2025), and llama.cpp issue #25187 (for 8.1, 8.2).
- Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025) (for 8.7).
- DeepSeek-V3 technical report, MTP section; Li et al., *EAGLE* and *EAGLE-3* (for 8.8).
- Zhang et al., *70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float* (DFloat11, 2025) (for 8.9).

---

## Appendix F: WSL2 pitfalls

- **Driver:** never install a Linux NVIDIA driver in WSL. The Windows driver provides the GPU.
- **Speed:** keep models and code under `~` (ext4), not `/mnt/c`.
- **WSL ≠ Windows:** WSL2 speed may differ from native Windows. Never mix the two in one comparison.
- **Shared GPU:** the GPU is shared with Windows apps (browsers, games, Hyper-V VMs). flashference lost a session to a Hyper-V worker using the GPU. Your idle check catches that, so keep it on.
- **Power:** laptops need AC power, the same power plan every session, and the same position, since temperature affects clocks. Record GPU temperature and clocks in every result.
- **Memory:** if a build is killed for no visible reason, you ran out of memory. Lower `-j` or raise `memory=` in `.wslconfig`.
