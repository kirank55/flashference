# toy/: Phase 1 exercises

Run every command from the repo root (`~/code-wsl/flashference`) with the venv's Python.
Each script prints its results and writes raw numbers to `results/phase1/`.
The SmolLM2-360M-Instruct weights (~724 MB) download from the Hugging Face Hub on first use.
After that, set `HF_HUB_OFFLINE=1`: on 2026-10-30 a run hung for over 9 minutes on a Hub network check.

| Exercise | Command | Writes |
|---|---|---|
| 1.1 Token round-trip | `.venv/bin/python toy/phase1_tokens.py` | `results/phase1/tokens.json` |
| 1.2 Greedy, short prompt | `.venv/bin/python toy/phase1_greedy.py` | `results/phase1/greedy_cuda_bf16.json` |
| 1.2 Greedy, ~1.2k-token prompt | `.venv/bin/python toy/phase1_greedy.py --long` | `results/phase1/greedy_cuda_bf16_long.json` |
| 1.2 step 4: one forward vs length | `.venv/bin/python toy/phase1_forward.py` | `results/phase1/forward_cuda_bf16.json` |
| 1.3 Parameter bytes | `.venv/bin/python toy/phase1_params.py` | `results/phase1/params.json` |
| 1.4 Qwen3.5-0.8B GGUF | `.venv/bin/python toy/phase1_gguf.py` | `results/phase1/gguf.json` |
| 1.5 Sampling by hand | `.venv/bin/python toy/phase1_sample.py` | `results/phase1/sample.json` |

Useful flags:

- `phase1_greedy.py --n 32 --reps 9 --device cuda|cpu --dtype bf16|fp16|fp32 --long`.
  Timed runs are interleaved (nocache, cached, generate, nocache, ...) so drift hits all three alike.
  The script exits with an error if the three methods produce different token ids.
  Without a GPU, use `--device cpu --dtype fp32`.
- `phase1_gguf.py --keys --tensors` prints every metadata key and every tensor (name, shape, type, bytes).
- `phase1_sample.py --seed 1234 --temperature 0.7 --top-p 0.9 --samples 5`. The same seed gives the same samples.

`phase1_sample.py` and `phase1_forward.py` import the loader from `phase1_greedy.py`.
The paths `models/…` and `results/phase1/…` are relative, so run the scripts from the repo root.
