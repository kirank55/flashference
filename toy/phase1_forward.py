"""Phase 1.2 step 4 helper: what does ONE forward pass cost, by sequence length?

No-cache greedy runs a full forward over the whole sequence every step; the cached loop runs
a 1-token forward. If a full forward over L tokens costs about the same as over 1 token, the
cache cannot save anything visible. This script measures where that stops being true.

Run from the repo root:  .venv/bin/python toy/phase1_forward.py [--reps 30]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from phase1_greedy import DTYPES, FILLER, MODEL, PROMPT, load, prompt_ids

LENGTHS = [1, 40, 72, 150, 300, 600, 1240]


def median_min_max(fn, reps, device):
    fn()                                   # warmup
    xs = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        xs.append(time.perf_counter() - t)
    xs.sort()
    return {"median_ms": xs[len(xs) // 2] * 1e3, "min_ms": xs[0] * 1e3, "max_ms": xs[-1] * 1e3}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    args = ap.parse_args()

    tok, model = load(args.device, args.dtype)
    ids = prompt_ids(tok, args.device, FILLER + PROMPT)        # 1240 tokens; take prefixes
    print(f"model={MODEL} device={args.device} dtype={args.dtype} reps={args.reps}\n")

    full = {}
    for L in LENGTHS:
        x = ids[:, :L]
        full[L] = median_min_max(lambda: model(x, use_cache=False), args.reps, args.device)
        print(f"full forward, {L:5d} tokens: median {full[L]['median_ms']:6.1f} ms  "
              f"(min {full[L]['min_ms']:6.1f}, max {full[L]['max_ms']:6.1f})")

    # One cached decode step (1 new token) after a prompt of length L; crop(-1) undoes it.
    step = {}
    for L in [40, 1200]:
        past = model(ids[:, :L], use_cache=True).past_key_values
        nxt = ids[:, L:L + 1]

        def one_step():
            model(nxt, past_key_values=past, use_cache=True)
            past.crop(-1)
        step[L] = median_min_max(one_step, args.reps, args.device)
        print(f"cached 1-token step after {L:5d}: median {step[L]['median_ms']:6.1f} ms  "
              f"(min {step[L]['min_ms']:6.1f}, max {step[L]['max_ms']:6.1f})")

    out_path = Path(f"results/phase1/forward_{args.device}_{args.dtype}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": MODEL, "device": args.device, "dtype": args.dtype, "reps": args.reps,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch": torch.__version__, "prompt": FILLER + PROMPT,
        "full_forward": full, "cached_step": step,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
