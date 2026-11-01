"""Phase 1.2: greedy generation by hand, three ways (plan.md 1.2).

Greedy decoding picks the most likely next token (argmax of the logits) and repeats. The model
is a transformer: at every step it turns the sequence so far into a probability for every
vocabulary token. Three implementations:

  A  greedy_nocache  re-runs the model on the whole sequence at every step
  B  greedy_cached   passes past_key_values, so each step runs only the newest token
  C  generate        model.generate(do_sample=False), transformers' own loop

The KV cache is the important idea: attention layers store the keys and values they computed
for earlier positions, so a new token only needs its own query against that cache. A must
recompute every earlier position; B does not. All three must return the same ids. Timed repeats
interleave A B C A B C ... so that drift (temperature, clocks, background load) hits all three
alike. fp32 throughout keeps rounding small, but the three add things up in different orders
(a full sequence at once, or one position at a time), so identical ids are a finding, not a given.

Durations use CLOCK_MONOTONIC_RAW: this machine's CLOCK_MONOTONIC runs about 5 % fast and its
wall clock is stepped back, so neither perf_counter() nor time() is a trustworthy stopwatch.

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


def now():
    """Seconds from CLOCK_MONOTONIC_RAW: a monotonic clock the NTP daemon cannot step."""
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def sync(device):
    """On the GPU, kernels run asynchronously; wait for them before stopping the clock."""
    if device == "cuda":
        torch.cuda.synchronize()


def greedy_nocache(model, ids, n, eos, steps):
    """Each step feeds the whole sequence so far; the model recomputes every position."""
    out = []
    for _ in range(n):
        t0 = now()
        logits = model(ids, use_cache=False).logits   # [1, len, vocab]: a row for every position
        nxt = logits[0, -1].argmax()                  # only the last row predicts the next token
        out.append(int(nxt))                          # int() waits for the GPU
        steps.append(now() - t0)
        if out[-1] == eos:
            break
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return out


def greedy_cached(model, ids, n, eos, steps):
    """The first step runs the prompt and keeps each layer's keys and values; later steps run one token."""
    out, past, x = [], None, ids
    for _ in range(n):
        t0 = now()
        o = model(x, past_key_values=past, use_cache=True)
        past = o.past_key_values                      # grows by one position per step
        nxt = o.logits[0, -1].argmax()
        out.append(int(nxt))
        steps.append(now() - t0)
        if out[-1] == eos:
            break
        x = nxt.view(1, 1)
    return out


def generate(model, ids, n, eos, steps):
    """transformers' own greedy loop. It uses a KV cache internally; no per-step times."""
    o = model.generate(ids, do_sample=False, max_new_tokens=n)
    return [int(t) for t in o[0, ids.shape[1]:]]


METHODS = {"nocache": greedy_nocache, "cached": greedy_cached, "generate": generate}


def gpu_reading():
    q = "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,driver_version"
    p = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return dict(zip(q.split(","), [v.strip() for v in p.stdout.strip().split(",")])) if p.returncode == 0 else {}


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).to(a.device).eval()
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=a.device)
    eos = tok.eos_token_id

    before = gpu_reading() if a.device == "cuda" else {}
    results, times, steps = {}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
    with torch.inference_mode():
        for m, fn in METHODS.items():                 # warm-up: first calls pay one-off setup costs
            results[m] = fn(model, ids, a.n, eos, [])
        for _ in range(a.repeats):
            for m, fn in METHODS.items():             # A B C, A B C, ...
                st = []
                sync(a.device)
                t0 = now()
                got = fn(model, ids, a.n, eos, st)
                sync(a.device)
                times[m].append(now() - t0)
                steps[m].append(st)
                assert got == results[m], f"{m} gave different ids on a repeat"
    after = gpu_reading() if a.device == "cuda" else {}

    n_new = len(results["cached"])
    det = {
        "model": TARGET, "dtype": "float32", "device": a.device, "prompt": PROMPT, "n": a.n,
        "prompt_ids": ids[0].tolist(), "ids": results,
        "all_three_equal": results["nocache"] == results["cached"] == results["generate"],
        "n_generated": n_new, "text": tok.decode(results["cached"]),
    }
    med = {m: (statistics.median(v) if v else None) for m, v in times.items()}
    meas = {
        "repeats": a.repeats, "order": "one warm-up call per method, then repeats interleaved A B C",
        "clock": "CLOCK_MONOTONIC_RAW",
        "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
        "seconds": times,
        "median_s": med,
        "min_s": {m: (min(v) if v else None) for m, v in times.items()},
        "max_s": {m: (max(v) if v else None) for m, v in times.items()},
        "median_ms_per_token": {m: (1000 * med[m] / n_new if med[m] is not None and n_new else None)
                                for m in METHODS},
        "step_seconds": {m: steps[m] for m in ("nocache", "cached")},
    }
    if a.device == "cuda":
        meas["gpu_name"] = torch.cuda.get_device_name(0)
        meas["gpu_before"] = before
        meas["gpu_after"] = after

    print(f"{a.device}: prompt {ids.shape[1]} tokens, generated {n_new}; all three equal: {det['all_three_equal']}")
    for m in METHODS:
        if med[m] is None:
            print(f"  {m:9s} warm-up only (no timed repeats)")
        else:
            print(f"  {m:9s} median {med[m]:.3f} s  (range {meas['min_s'][m]:.3f}..{meas['max_s'][m]:.3f}),"
                  f" {meas['median_ms_per_token'][m]:.1f} ms/token")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_greedy.py", "started_utc": started, "deterministic": det, "measured": meas}
    (out / f"greedy-{a.device}.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
