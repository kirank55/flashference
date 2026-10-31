"""Phase 1.5: sampling by hand (plan.md 1.5).

sample() turns one logits vector into a token: divide by the temperature T, softmax, keep the
smallest set of most-likely tokens whose probability adds up to top_p ("nucleus"), renormalise,
draw one. T = 0 means plain argmax (greedy).

1. Toy check: on a 10-token logits vector, 100,000 draws must match the intended distribution.
2. Five samples at T = 0.7 (top_p 0.9, seeds 0-4) from the SmolLM2-360M target, against its greedy
   output. CPU, fp32, KV cache as in phase1_greedy.py.

  python toy/phase1_sample.py [--n 48] [--out results/1]

Writes <out>/sample.json. With fixed seeds on the CPU, all of it is deterministic.
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import torch
from scipy.stats import chisquare
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TARGET = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT = "Write the opening sentence of a story about a lighthouse keeper."


def nucleus_probs(logits, temperature, top_p):
    """The distribution sample() draws from, over the full vocabulary."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    p_sorted, idx = probs.sort(descending=True)
    keep = (p_sorted.cumsum(0) - p_sorted) < top_p     # mass *before* a token < top_p: keeps at least one
    p_sorted = p_sorted * keep
    out = torch.zeros_like(probs)
    out[idx] = p_sorted / p_sorted.sum()
    return out


def sample(logits, temperature, top_p, gen):
    if temperature == 0:
        return int(logits.argmax())
    return int(torch.multinomial(nucleus_probs(logits, temperature, top_p), 1, generator=gen))


def generate(model, ids, n, eos, temperature, top_p, seed):
    gen = torch.Generator().manual_seed(seed)
    out, past, x = [], None, ids
    for _ in range(n):
        o = model(x, past_key_values=past, use_cache=True)
        past = o.past_key_values
        t = sample(o.logits[0, -1], temperature, top_p, gen)
        out.append(t)
        if t == eos:
            break
        x = torch.tensor([[t]])
    return out


def toy_check(draws=100_000):
    logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -5.0])
    gen = torch.Generator().manual_seed(1234)
    rows = []
    for T, top_p in [(1.0, 1.0), (0.7, 1.0), (0.7, 0.9), (1.5, 0.8)]:
        want = nucleus_probs(logits, T, top_p)
        counts = torch.bincount(torch.tensor([sample(logits, T, top_p, gen) for _ in range(draws)]), minlength=10)
        kept = want > 0
        stat, p = chisquare(counts[kept].double().numpy(), (want[kept] * draws).numpy())
        rows.append({"temperature": T, "top_p": top_p, "expected_probs": [round(float(v), 6) for v in want],
                     "counts": counts.tolist(), "tokens_kept": int(kept.sum()),
                     "draws_outside_nucleus": int(counts[~kept].sum()),
                     "max_abs_freq_error": float((counts.double() / draws - want).abs().max()),
                     "chi2_p_value": float(p)})
    return {"logits": logits.tolist(), "draws": draws, "cases": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    torch.manual_seed(0)
    det = {"toy": toy_check()}

    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).eval()
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)])
    eos = tok.eos_token_id
    with torch.inference_mode():
        greedy = generate(model, ids, a.n, eos, 0.0, 1.0, 0)
        samples = [generate(model, ids, a.n, eos, 0.7, 0.9, s) for s in range(5)]

    def first_diff(x, y):
        return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None if len(x) == len(y) else min(len(x), len(y)))
    det["model"] = {
        "model": TARGET, "dtype": "float32", "device": "cpu", "prompt": PROMPT, "n": a.n,
        "temperature": 0.7, "top_p": 0.9, "seeds": list(range(5)),
        "greedy_ids": greedy, "greedy_text": tok.decode(greedy),
        "samples": [{"seed": s, "ids": x, "text": tok.decode(x), "first_position_differing_from_greedy": first_diff(x, greedy)}
                    for s, x in enumerate(samples)],
        "distinct_samples": len({tuple(x) for x in samples}),
    }
    for c in det["toy"]["cases"]:
        print(f"toy T={c['temperature']} top_p={c['top_p']}: kept {c['tokens_kept']}/10, outside nucleus "
              f"{c['draws_outside_nucleus']}, max |freq - p| {c['max_abs_freq_error']:.4f}, chi2 p {c['chi2_p_value']:.3f}")
    print(f"\ngreedy: {det['model']['greedy_text']!r}")
    for s in det["model"]["samples"]:
        print(f"seed {s['seed']} (differs from greedy at {s['first_position_differing_from_greedy']}): {s['text']!r}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_sample.py", "started_utc": started, "deterministic": det}
    (out / "sample.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
