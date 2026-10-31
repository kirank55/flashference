"""Phase 1.2: generation by hand.

greedy_nocache: re-run the WHOLE sequence every step (no KV cache).
greedy_cached:  pass past_key_values so only the newest token runs each step.
Both must match model.generate(do_sample=False) token for token.

Run from the repo root:  .venv/bin/python toy/phase1_greedy.py [--n 32] [--device cuda] [--dtype bf16] [--long]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT = "Explain in two sentences why the sky is blue."
# --long prepends this filler so the no-cache path has a long sequence to re-run.
FILLER = ("Sunlight is made of many colours. Air molecules are much smaller than the "
          "wavelength of visible light. ") * 60
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(device, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPES[dtype]).to(device).eval()
    return tok, model


def prompt_ids(tok, device, prompt=PROMPT):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def greedy_nocache(model, ids, n):
    for _ in range(n):
        logits = model(ids, use_cache=False).logits          # [1, len(ids), vocab]
        nxt = logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids


@torch.no_grad()
def greedy_cached(model, ids, n):
    out = model(ids, use_cache=True)                        # prefill: whole prompt once
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().view(1, 1)
    ids = torch.cat([ids, nxt], dim=1)
    for _ in range(n - 1):
        out = model(nxt, past_key_values=past, use_cache=True)  # only the new token runs
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


@torch.no_grad()
def hf_generate(model, ids, n):
    return model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                          max_new_tokens=n, temperature=None, top_p=None, top_k=None)


def timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    ap.add_argument("--long", action="store_true", help="prepend ~1.2k tokens of filler")
    ap.add_argument("--reps", type=int, default=9, help="timed runs per method (after 1 warmup), interleaved")
    args = ap.parse_args()

    tok, model = load(args.device, args.dtype)
    prompt = FILLER + PROMPT if args.long else PROMPT
    ids = prompt_ids(tok, args.device, prompt)
    plen = ids.shape[1]
    print(f"model={MODEL} device={args.device} dtype={args.dtype} prompt_tokens={plen} n={args.n}")

    fns = {
        "nocache": lambda: greedy_nocache(model, ids, args.n),
        "cached": lambda: greedy_cached(model, ids, args.n),
        "generate": lambda: hf_generate(model, ids, args.n),
    }
    for fn in fns.values():   # warmup: one run of each, not timed
        fn()

    # Interleave the methods (A B C A B C ...) so slow drift on this GPU hits all three alike.
    new, runs = {}, {name: [] for name in fns}
    for _ in range(args.reps):
        for name, fn in fns.items():
            out, t = timed(fn, args.device)
            runs[name].append(t)
            new[name] = out[0, plen:].tolist()
    secs = {name: sorted(r)[len(r) // 2] for name, r in runs.items()}   # median

    # generate() stops early at EOS; the hand loops always run n steps.
    k = len(new["generate"])
    match_cached = new["cached"][:k] == new["generate"]
    match_nocache = new["nocache"][:k] == new["generate"]
    match_hand = new["cached"] == new["nocache"]

    print(f"\ncontinuation: {tok.decode(new['generate'])!r}\n")
    for name in fns:
        print(f"{name:9s} median {secs[name] * 1000:8.1f} ms  ({secs[name] * 1000 / args.n:5.1f} ms/token)  "
              f"min {min(runs[name]) * 1000:8.1f}  max {max(runs[name]) * 1000:8.1f}")
        print(f"          runs_ms={[round(t * 1000, 1) for t in runs[name]]}")
        print(f"          ids={new[name]}")
    print(f"\nmatch nocache==generate: {match_nocache}  cached==generate: {match_cached}  "
          f"nocache==cached: {match_hand}  (compared {k} tokens)")

    tag = "_long" if args.long else ""
    out_path = Path(f"results/phase1/greedy_{args.device}_{args.dtype}{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": MODEL, "device": args.device, "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch": torch.__version__, "long": args.long, "prompt": prompt, "prompt_tokens": plen,
        "n": args.n, "reps": args.reps, "new_ids": new, "seconds_median": secs, "seconds_runs": runs,
        "match": {"nocache_eq_generate": match_nocache, "cached_eq_generate": match_cached,
                  "nocache_eq_cached": match_hand, "tokens_compared": k},
    }, indent=2))
    print(f"wrote {out_path}")

    assert match_nocache and match_cached and match_hand, "token ids differ between methods"


if __name__ == "__main__":
    main()
