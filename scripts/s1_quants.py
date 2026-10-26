#!/usr/bin/env python3
r"""S1 (Qwen3.5-0.8B only): speed (llama-bench) + quality (llama-perplexity PPL, KLD vs bf16) per quant.
Usage: python scripts\s1_quants.py bench|ppl|report
Order: bench pass 1 (ascending size) -> ppl/KLD -> bench pass 2 (reverse order, drift/control check).
"""
import glob, json, os, re, subprocess, sys, time, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LC = os.path.join(ROOT, "llama-cpp")
RES = os.path.join(ROOT, "results")
TEXT = os.path.join(ROOT, "data", "wiki.test.raw")
LOGITS = os.path.join(ROOT, "logs", "s1-bf16-logits.bin")
OUT = os.path.join(RES, "2026-10-26-s1-quants-0.8b.raw.json")
CHUNKS, CTX = 20, 512

def models():
    fs = glob.glob(os.path.join(ROOT, "models", "Qwen_Qwen3.5-0.8B-*.gguf"))
    return sorted(fs, key=os.path.getsize)

def qname(p): return os.path.basename(p)[len("Qwen_Qwen3.5-0.8B-"):-5]

def idle():
    for _ in range(120):
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        u, m = [float(x) for x in o.split(",")]
        if u < 5 and m < 300: return {"util": u, "mem_mib": m}
        time.sleep(2)
    raise RuntimeError("GPU not idle")

def bench(m):
    r = subprocess.run([os.path.join(LC, "llama-bench.exe"), "-m", m, "-ngl", "99", "-fa", "1", "-p", "512", "-n", "256", "-r", "5", "-o", "json"], capture_output=True, text=True)
    j = json.loads(r.stdout[r.stdout.index("["):])
    d = {}
    for e in j:
        d["pp" if e["n_prompt"] else "tg"] = {"mean": e["avg_ts"], "sd": e["stddev_ts"], "samples": e.get("samples_ts")}
        d["size"] = e["model_size"]; d["params"] = e["model_n_params"]; d["type"] = e["model_type"]
        d["build"] = e.get("build_commit")
    return d

def run_bench(pass_name, order):
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data.setdefault(pass_name, {})
    for m in order:
        idle()
        data[pass_name][qname(m)] = bench(m)
        d = data[pass_name][qname(m)]
        print(pass_name, qname(m), "tg %.1f +/- %.1f  pp %.0f" % (d["tg"]["mean"], d["tg"]["sd"], d["pp"]["mean"]), flush=True)
        json.dump(data, open(OUT, "w"), indent=1)

def ppl_run(m, extra):
    r = subprocess.run([os.path.join(LC, "llama-perplexity.exe"), "-m", m, "-f", TEXT, "-c", str(CTX), "--chunks", str(CHUNKS), "-ngl", "99", "-fa", "on"] + extra, capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr

def num(rx, s):
    mm = re.search(rx, s, re.M)
    return [float(x) for x in mm.groups()] if mm else None

def run_ppl():
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data.setdefault("quality", {})
    ms = models()
    bf = [m for m in ms if qname(m) == "bf16"][0]
    idle()
    if not os.path.exists(LOGITS):
        log = ppl_run(bf, ["--kl-divergence-base", LOGITS])
        open(os.path.join(ROOT, "logs", "s1-ppl-bf16.log"), "w").write(log)
    else:
        log = open(os.path.join(ROOT, "logs", "s1-ppl-bf16.log")).read()
    data["quality"]["bf16"] = {"ppl": num(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", log), "kld": [0.0, 0.0], "same_top": [100.0, 0.0]}
    for m in ms:
        q = qname(m)
        if q == "bf16": continue
        idle()
        log = ppl_run(m, ["--kl-divergence-base", LOGITS, "--kl-divergence"])
        open(os.path.join(ROOT, "logs", "s1-ppl-%s.log" % q), "w").write(log)
        data["quality"][q] = {"ppl": num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)\D+?([\d.]+)\s*$", log),
                              "ppl_bf16_ref": num(r"Mean PPL\(base\)\s*:\s*([\d.]+)\D+?([\d.]+)\s*$", log),
                              "kld": num(r"Mean\s+KLD:\s*([\d.]+)\D+?([\d.]+)\s*$", log),
                              "same_top": num(r"Same top p:\s*([\d.]+)\D+?([\d.]+)\s*%", log),
                              "rms_dp": num(r"^RMS \S+\s*:\s*([\d.]+)\D+?([\d.]+)\s*%", log)}
        print(q, data["quality"][q], flush=True)
        json.dump(data, open(OUT, "w"), indent=1)


def reparse():
    """Re-read logs\\s1-ppl-*.log (written by `ppl`) into the raw json (the runs themselves are not repeated)."""
    data = json.load(open(OUT))
    data["quality"] = {}
    for m in models():
        q = qname(m)
        log = open(os.path.join(ROOT, "logs", "s1-ppl-%s.log" % q), encoding="latin-1").read()
        if q == "bf16":
            data["quality"][q] = {"ppl": num(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", log), "kld": [0.0, 0.0], "same_top": [100.0, 0.0]}
            continue
        data["quality"][q] = {"ppl": num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)\D+?([\d.]+)\s*$", log),
                              "ppl_bf16_ref": num(r"Mean PPL\(base\)\s*:\s*([\d.]+)\D+?([\d.]+)\s*$", log),
                              "kld": num(r"Mean\s+KLD:\s*([\d.]+)\D+?([\d.]+)\s*$", log),
                              "same_top": num(r"Same top p:\s*([\d.]+)\D+?([\d.]+)\s*%", log),
                              "rms_dp": num(r"^RMS \S+\s*:\s*([\d.]+)\D+?([\d.]+)\s*%", log)}
        print(q, data["quality"][q])
    json.dump(data, open(OUT, "w"), indent=1)
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bench": run_bench("pass1", models())
    elif mode == "bench2": run_bench("pass2", models()[::-1])
    elif mode == "ppl": run_ppl()
    elif mode == "reparse": reparse()



