"""Phase 1.5: temperature + top-p sampling by hand (no transformers sampler).

Run from the repo root:  .venv/bin/python toy/phase1_sample.py [--seed 1234] [--temperature 0.7] [--top-p 0.9]
"""
import argparse
import json
from pathlib import Path

import torch

from phase1_greedy import DTYPES, MODEL, PROMPT, greedy_cached, load, prompt_ids


def sample_from_logits(logits, temperature, top_p, gen):
    """logits: [vocab]. Returns one token id (0-dim tensor)."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_p, sorted_ids = probs.sort(descending=True)
    # Keep the smallest prefix whose mass reaches top_p. Comparing the mass *before*
    # each token means the most likely token is always kept.
    keep = (sorted_p.cumsum(-1) - sorted_p) < top_p
    sorted_p = sorted_p * keep
    sorted_p = sorted_p / sorted_p.sum()
    choice = torch.multinomial(sorted_p, 1, generator=gen)
    return sorted_ids[choice]


@torch.no_grad()
def sample_cached(model, ids, n, temperature, top_p, gen):
    out = model(ids, use_cache=True)
    for _ in range(n):
        nxt = sample_from_logits(out.logits[0, -1], temperature, top_p, gen).view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
        out = model(nxt, past_key_values=out.past_key_values, use_cache=True)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    args = ap.parse_args()

    tok, model = load(args.device, args.dtype)
    ids = prompt_ids(tok, args.device)
    plen = ids.shape[1]
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"model={MODEL} device={args.device} dtype={args.dtype} seed={args.seed} "
          f"T={args.temperature} top_p={args.top_p} n={args.n}")
    print(f"prompt: {PROMPT!r}\n")

    greedy = greedy_cached(model, ids, args.n)[0, plen:].tolist()
    print(f"greedy   : {tok.decode(greedy)!r}")
    samples = []
    for i in range(args.samples):
        s = sample_cached(model, ids, args.n, args.temperature, args.top_p, gen)[0, plen:].tolist()
        same = sum(a == b for a, b in zip(s, greedy))
        prefix = next((j for j, (a, b) in enumerate(zip(s, greedy)) if a != b), len(s))
        samples.append({"ids": s, "text": tok.decode(s), "same_position_as_greedy": same,
                        "common_prefix_with_greedy": prefix})
        print(f"sample {i}: {tok.decode(s)!r}")
        print(f"          common prefix with greedy: {prefix} tokens; same token at {same}/{len(s)} positions")

    out_path = Path("results/phase1/sample.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": MODEL, "device": args.device, "dtype": args.dtype, "seed": args.seed,
        "temperature": args.temperature, "top_p": args.top_p, "n": args.n, "prompt": PROMPT,
        "greedy": {"ids": greedy, "text": tok.decode(greedy)}, "samples": samples,
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
