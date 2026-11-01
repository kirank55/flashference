"""Phase 1.5: sampling by hand (plan.md 1.5).

Greedy decoding always takes the most likely token. Sampling instead draws from a distribution:

  1. divide the logits by the temperature T (small T -> sharper, large T -> flatter);
  2. softmax to turn them into probabilities;
  3. keep the smallest set of most-likely tokens whose probability adds up to top_p ("nucleus");
  4. renormalise inside the nucleus and draw one token with a seeded torch.Generator.

The toy check draws 100,000 tokens through sample() itself from a fixed 6-token logits vector,
and compares the counts with probabilities worked out separately, in plain Python, from the
definition above (a chi-square goodness-of-fit test). Tokens outside the nucleus must never come up. The model part runs
SmolLM2-360M in fp32 on the CPU: greedy 64 tokens, then five 64-token samples at T = 0.7,
top_p = 0.9, seeds 0-4, each with its own generator and the same KV-cached loop as 1.2.

  python toy/phase1_sample.py [--n 64] [--out results/1]

Writes <out>/sample.json. With fixed seeds on the CPU, all of it is deterministic.
"""
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path

import torch
from scipy.stats import chisquare
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TARGET = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT = "Explain in one paragraph why the sky is blue."


def nucleus(logits, temperature, top_p):
    """Return (token ids, renormalised probabilities) of the smallest top_p nucleus."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    p_sorted, idx = probs.sort(descending=True)
    csum = p_sorted.cumsum(0)
    # first index whose cumulative mass reaches top_p; +1 turns it into a prefix length
    k = int(torch.searchsorted(csum, torch.tensor(top_p, dtype=csum.dtype))) + 1
    k = min(k, p_sorted.numel())
    ids = idx[:k]
    p = p_sorted[:k] / p_sorted[:k].sum()
    return ids, p


def sample(logits, temperature, top_p, gen):
    """One draw. T = 0 is plain argmax (greedy)."""
    if temperature == 0:
        return int(logits.argmax())
    ids, p = nucleus(logits, temperature, top_p)
    return int(ids[torch.multinomial(p, 1, generator=gen)])


def toy_check(draws=100_000):
    """Draw through sample() itself and compare with a reference worked out separately, in plain
    Python, from the definition: softmax(logits / T), then the smallest most-likely set whose
    probability reaches top_p, renormalised. A bug anywhere in sample() shows up as a bad fit or
    as draws outside the nucleus."""
    logits = [2.0, 1.5, 1.0, 0.5, 0.0, -1.0]
    temperature, top_p, seed = 0.7, 0.9, 1234
    w = [math.exp(x / temperature) for x in logits]
    p = [x / sum(w) for x in w]
    keep, mass = [], 0.0
    for i in sorted(range(len(p)), key=lambda i: -p[i]):
        keep.append(i)
        mass += p[i]
        if mass >= top_p:
            break
    ref = {i: p[i] / mass for i in keep}
    gen = torch.Generator().manual_seed(seed)
    t = torch.tensor(logits)
    observed = [0] * len(logits)
    for _ in range(draws):
        observed[sample(t, temperature, top_p, gen)] += 1
    outside = sum(c for i, c in enumerate(observed) if i not in ref)
    stat = pval = None
    if outside == 0:                                   # chisquare needs equal totals
        stat, pval = chisquare([observed[i] for i in keep], [draws * ref[i] for i in keep])
    ids, q = nucleus(t, temperature, top_p)            # the sampler's own nucleus, for comparison
    return {
        "logits": logits, "temperature": temperature, "top_p": top_p, "draws": draws, "seed": seed,
        "softmax_probs": p,
        "nucleus_ids": keep, "nucleus_probs": [ref[i] for i in keep],
        "sampler_nucleus_matches_reference": [int(i) for i in ids] == keep
        and all(abs(float(a) - ref[i]) < 1e-12 for a, i in zip(q, keep)),
        "expected_counts": [draws * ref.get(i, 0.0) for i in range(len(logits))],
        "observed_counts": observed,
        "chi_square_statistic": None if stat is None else float(stat),
        "chi_square_p_value": None if pval is None else float(pval),
        "draws_outside_nucleus": outside,
    }


def cached_ids(model, ids, n, eos, pick):
    """Greedy or sampled generation with a KV cache; pick(logits) returns the next token id."""
    out, past, x = [], None, ids
    for _ in range(n):
        o = model(x, past_key_values=past, use_cache=True)
        past = o.past_key_values
        t = pick(o.logits[0, -1])
        out.append(t)
        if t == eos:
            break
        x = torch.tensor([[t]], device=ids.device)
    return out


def first_diff(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()
    torch.manual_seed(0)

    det = {"toy": toy_check()}

    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).eval()
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)])
    eos = tok.eos_token_id
    with torch.inference_mode():
        greedy = cached_ids(model, ids, a.n, eos, lambda logits: int(logits.argmax()))
        samples = []
        for s in range(5):
            gen = torch.Generator().manual_seed(s)
            samples.append(cached_ids(model, ids, a.n, eos,
                                      lambda logits, g=gen: sample(logits, 0.7, 0.9, g)))

    det["model"] = {
        "model": TARGET, "dtype": "float32", "device": "cpu", "prompt": PROMPT, "n": a.n,
        "prompt_ids": ids[0].tolist(),
        "temperature": 0.7, "top_p": 0.9, "seeds": list(range(5)),
        "greedy_ids": greedy, "greedy_text": tok.decode(greedy),
        "samples": [{"seed": s, "ids": x, "text": tok.decode(x),
                     "differs_from_greedy": x != greedy,
                     "first_position_differing_from_greedy": first_diff(x, greedy)}
                    for s, x in enumerate(samples)],
        "distinct_samples": len({tuple(x) for x in samples}),
    }

    toy = det["toy"]
    print(f"toy T={toy['temperature']} top_p={toy['top_p']}: nucleus {toy['nucleus_ids']} "
          f"({len(toy['nucleus_ids'])} of {len(toy['logits'])}), outside {toy['draws_outside_nucleus']}, "
          f"chi2 p {toy['chi_square_p_value']}, sampler nucleus matches reference: {toy['sampler_nucleus_matches_reference']}")
    for i, p in zip(toy["nucleus_ids"], toy["nucleus_probs"]):
        print(f"  id {i}: p {p:.6f}, observed {toy['observed_counts'][i]}")
    print(f"\ngreedy: {det['model']['greedy_text']!r}")
    for s in det["model"]["samples"]:
        print(f"seed {s['seed']} (differs from greedy at {s['first_position_differing_from_greedy']}): {s['text']!r}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_sample.py", "started_utc": started, "deterministic": det}
    (out / "sample.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
