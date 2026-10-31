"""Phase 1.2: greedy generation by hand, three ways (plan.md 1.2).

  A  greedy_nocache  re-runs the model on the whole sequence at every step
  B  greedy_cached   passes past_key_values, so each step runs only the newest token
  C  generate        model.generate(do_sample=False), transformers' own loop

All three must return the same token ids. Timed repeats interleave A B C A B C ... so that any
drift (temperature, clocks, background load) hits all three alike. fp32 throughout, so the
three can't disagree through rounding.

  python toy/phase1_greedy.py --device cpu|cuda [--n 64] [--repeats 5] [--out results/1]

Writes <out>/greedy-<device>.json. "deterministic" holds the ids; "measured" holds the timings.
"""
import argparse, json, statistics, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TARGET = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT = "Explain in one paragraph why the sky is blue."


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def greedy_nocache(model, ids, n, eos, steps):
    """Each step feeds the whole sequence so far; the model recomputes every position."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        logits = model(ids).logits                    # [1, len, vocab]: a row for every position
        nxt = logits[0, -1].argmax()                  # only the last row predicts the next token
        out.append(int(nxt))                          # int() waits for the GPU
        steps.append(time.perf_counter() - t0)
        if out[-1] == eos:
            break
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return out


def greedy_cached(model, ids, n, eos, steps):
    """The first step runs the prompt and keeps each layer's keys and values; later steps run one token."""
    out, past, x = [], None, ids
    for _ in range(n):
        t0 = time.perf_counter()
        o = model(x, past_key_values=past, use_cache=True)
        past = o.past_key_values                      # grows by one position per step
        nxt = o.logits[0, -1].argmax()
        out.append(int(nxt))
        steps.append(time.perf_counter() - t0)
        if out[-1] == eos:
            break
        x = nxt.view(1, 1)
    return out


def generate(model, ids, n, eos, steps):
    o = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=n,
                       eos_token_id=eos, pad_token_id=eos)
    return [int(t) for t in o[0, ids.shape[1]:]]


METHODS = {"nocache": greedy_nocache, "cached": greedy_cached, "generate": generate}


def gpu_reading():
    q = "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,driver_version"
    p = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return dict(zip(q.split(","), [v.strip() for v in p.stdout.strip().split(",")])) if p.returncode == 0 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).to(a.device).eval()
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=a.device)
    eos = tok.eos_token_id

    before = gpu_reading()
    results, times, steps = {}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
    with torch.inference_mode():
        for m, fn in METHODS.items():                 # warm-up: first calls pay one-off setup costs
            results[m] = fn(model, ids, a.n, eos, [])
        for _ in range(a.repeats):
            for m, fn in METHODS.items():             # A B C, A B C, ...
                st = []
                sync(a.device)
                t0 = time.perf_counter()
                got = fn(model, ids, a.n, eos, st)
                sync(a.device)
                times[m].append(time.perf_counter() - t0)
                steps[m].append(st)
                assert got == results[m], f"{m} gave different ids on a repeat"
    after = gpu_reading()

    n_new = len(results["cached"])
    det = {
        "model": TARGET, "dtype": "float32", "device": a.device, "prompt": PROMPT, "n": a.n,
        "prompt_ids": ids[0].tolist(), "ids": results,
        "all_three_equal": results["nocache"] == results["cached"] == results["generate"],
        "n_generated": n_new, "text": tok.decode(results["cached"]),
    }
    meas = {
        "repeats": a.repeats, "order": "interleaved A B C per repeat, after one warm-up call each",
        "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
        "gpu_before": before, "gpu_after": after,
        "seconds": times,
        "median_s": {m: statistics.median(v) for m, v in times.items()},
        "min_s": {m: min(v) for m, v in times.items()}, "max_s": {m: max(v) for m, v in times.items()},
        "median_ms_per_token": {m: 1000 * statistics.median(v) / n_new for m, v in times.items()},
        "step_seconds": {m: v for m, v in steps.items() if m != "generate"},
    }
    print(f"{a.device}: prompt {ids.shape[1]} tokens, generated {n_new}; all three equal: {det['all_three_equal']}")
    for m in METHODS:
        print(f"  {m:9s} median {meas['median_s'][m]:.3f} s  (range {meas['min_s'][m]:.3f}..{meas['max_s'][m]:.3f}),"
              f" {meas['median_ms_per_token'][m]:.1f} ms/token")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_greedy.py", "started_utc": started, "deterministic": det, "measured": meas}
    (out / f"greedy-{a.device}.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
