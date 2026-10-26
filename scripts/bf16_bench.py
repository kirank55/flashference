#!/usr/bin/env python3
r"""bf16 free-text experiments (B1/B2/B3) on Qwen3.5-0.8B via llama-server / llama-bench (stdlib only).
Per server config: fresh server, GPU-idle check, one discarded warm-up pass over the 6 free-text prompts, 5 timed passes,
greedy (temperature 0, seed 1), cache_prompt=false. Per pass tok/s = sum(n)/sum(predicted_ms) over the 6 prompts.
llama-bench configs: -p 0 -n N -r 5 (bench does its own discarded warm-up); mean +/- SD of the 5 reps.
Usage: python scripts\bf16_bench.py --session B1 [--only name ...] [--list] [--tag SUFFIX] [--reps 5]
Resumable: configs already present in the session's raw json are skipped. Does not touch any system setting.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LC = os.path.join(ROOT, "llama-cpp")
SERVER, BENCH = os.path.join(LC, "llama-server.exe"), os.path.join(LC, "llama-bench.exe")
PORT = 8093
BASE = "http://127.0.0.1:%d" % PORT
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
M = lambda f: os.path.join(ROOT, "models", f)
BF16 = "Qwen_Qwen3.5-0.8B-bf16.gguf"
F16 = "Qwen3.5-0.8B-f16-from-bf16.gguf"
Q4MTP = "Qwen3.5-0.8B-MTP-Q4_K_M.gguf"
Q4 = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
RAW = {"B1": "2026-10-26-bf16-B1-overhead.raw.json", "B2": "2026-10-26-bf16-B2-backend-sampling.raw.json",
       "B3": "2026-10-26-bf16-B3-mtp.raw.json"}
PROMPTS = [p for p in json.load(open(os.path.join(ROOT, "prompts", "prompts.json")))["free"]]
BASE_ARGS = ["-ngl", "99", "-fa", "on", "-np", "1", "-kvu", "-c", "4096"]
ENVS = ["GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_GRAPH_OPT", "GGML_CUDA_DISABLE_FUSION", "LLAMA_GRAPH_REUSE_DISABLE", "GGML_CUDA_GRAPH_DEBUG"]

def chatml(u): return "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def http(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r: return json.loads(r.read())

def loopinfo(text):
    w = text.split()
    g4 = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    dup4 = 1 - len(set(g4)) / len(g4) if g4 else 0.0
    g6 = {}
    for i in range(len(w) - 5):
        k = tuple(w[i:i + 6]); g6[k] = g6.get(k, 0) + 1
    max6 = max(g6.values()) if g6 else 0
    return {"dup4": round(dup4, 3), "max6": max6, "loop": bool(dup4 >= 0.30 or max6 >= 3)}

def complete(p, n_predict, extra=None):
    t0 = time.perf_counter()
    body = {"prompt": chatml(p["user"]), "n_predict": n_predict, "temperature": 0.0, "seed": 1, "cache_prompt": False, "stream": False}
    body.update(extra or {})
    r = http("/completion", body)
    wall = time.perf_counter() - t0
    tm = r["timings"]
    return {"id": p["id"], "text": r["content"], "n": tm["predicted_n"], "ms": tm["predicted_ms"], "pp_ms": tm.get("prompt_ms"),
            "draft_n": tm.get("draft_n", 0), "draft_acc": tm.get("draft_n_accepted", 0), "wall": wall, "stop": r.get("stop_type")}

def gpu_check():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    u, m = [float(x) for x in o.split(",")]
    ps = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Process | ? {$_.Name -match 'llama|ggml'} | % {$_.Name}"], capture_output=True, text=True).stdout.split()
    return {"util": u, "mem_mib": m, "stray": ps}

def require_idle():
    for _ in range(8):
        s = gpu_check()
        if s["util"] <= 2 and s["mem_mib"] < 300 and not s["stray"]: return s
        time.sleep(2)
    raise SystemExit("GPU NOT IDLE: %s -- stopping" % s)

def env_for(env):
    e = {k: v for k, v in os.environ.items() if k not in ENVS}
    e.update(env or {}); return e

def start_server(name, model, args, env, tag):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log = open(os.path.join(ROOT, "logs", "bf16-%s%s.log" % (name, tag)), "w")
    cmd = [SERVER, "-m", M(model)] + args + ["--no-webui", "--host", "127.0.0.1", "--port", str(PORT)]
    p = subprocess.Popen(cmd, stdout=log, stderr=log, env=env_for(env))
    for _ in range(180):
        time.sleep(1)
        if p.poll() is not None: raise RuntimeError("server exited early; see logs\\bf16-%s%s.log" % (name, tag))
        try:
            if OPENER.open(BASE + "/health", timeout=2).status == 200: return p, " ".join(cmd[1:])
        except Exception: pass
    p.kill(); raise RuntimeError("server start timeout")

def stop(p):
    p.terminate()
    try: p.wait(20)
    except Exception: p.kill()

def agg(run, key="n", ms="ms"):
    return sum(r[key] for r in run) / (sum(r[ms] for r in run) / 1000)

def run_server(name, c, reps, tag):
    idle = require_idle()
    n_pred = c.get("n_predict", 256)
    proc, cmd = start_server(name, c["model"], c["args"], c.get("env"), tag)
    rec = {"name": name, "kind": "server", "desc": c["desc"], "model": c["model"], "env": c.get("env", {}), "cmd": cmd, "n_predict": n_pred,
           "idle_before": idle, "runs": [], "t_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        for p in PROMPTS: complete(p, n_pred, c.get("req"))  # discarded warm-up pass
        for rep in range(reps):
            run = []
            for p in PROMPTS:
                r = complete(p, n_pred, c.get("req"))
                r["hash"] = hashlib.md5(r["text"].encode()).hexdigest()
                r.update(loopinfo(r["text"]))
                if rep > 0: r.pop("text")
                run.append(r)
            rec["runs"].append(run)
            extra = " accept %.1f%%" % (100 * sum(r["draft_acc"] for r in run) / max(1, sum(r["draft_n"] for r in run))) if any(r["draft_n"] for r in run) else ""
            print(name + tag, "rep", rep + 1, "%.2f tok/s" % agg(run), "n=%d" % sum(r["n"] for r in run) + extra, flush=True)
    finally:
        stop(proc)
    return rec

def run_bench(name, c, reps, tag):
    idle = require_idle()
    cmd = [BENCH, "-m", M(c["model"])] + c["args"] + ["-r", str(reps), "-o", "json"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env_for(c.get("env")))
    txt = r.stdout
    j = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
    rec = {"name": name, "kind": "bench", "desc": c["desc"], "model": c["model"], "env": c.get("env", {}), "cmd": " ".join(cmd[1:]), "idle_before": idle,
           "tests": [{"n_prompt": x["n_prompt"], "n_gen": x["n_gen"], "n_depth": x.get("n_depth"), "avg_ts": x["avg_ts"], "stddev_ts": x["stddev_ts"],
                      "samples_ts": x["samples_ts"], "n_threads": x.get("n_threads"), "flash_attn": x.get("flash_attn"), "n_ubatch": x.get("n_ubatch")} for x in j],
           "wall_s": round(time.time() - t0, 1), "t_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    t = rec["tests"][-1]
    print(name + tag, "bench tg %.2f +/- %.2f" % (t["avg_ts"], t["stddev_ts"]), flush=True)
    return rec

def configs(session):
    C = {}
    def S(name, desc, model=BF16, args=None, env=None, **kw):
        C[name] = dict(kind="server", desc=desc, model=model, args=BASE_ARGS if args is None else args, env=env, **kw)
    def B(name, desc, model=BF16, args=None, env=None):
        C[name] = dict(kind="bench", desc=desc, model=model, args=args or ["-ngl", "99", "-fa", "on", "-p", "0", "-n", "256"], env=env)
    if session == "B1":
        S("control", "server bf16: -ngl 99 -fa on -np 1 -kvu -c 4096, -t/-b/-ub default (CONTROL)")
        B("bench-default", "llama-bench bf16 -ngl 99 -fa on -p 0 -n 256 (-t/-b/-ub default)")
        B("bench-prev-style", "llama-bench as in earlier baseline: -ngl 99 -fa 1 -p 512 -n 128", args=["-ngl", "99", "-fa", "on", "-p", "512", "-n", "128"])
        B("bench-prev-nowarmup", "llama-bench as in the earlier baseline AND --no-warmup (cold first rep counted)", args=["-ngl", "99", "-fa", "on", "-p", "512", "-n", "128", "--no-warmup"])
        S("server-np-auto", "server without -np/-kvu (default auto slots, non-unified KV)", args=["-ngl", "99", "-fa", "on", "-c", "4096"])
        S("server-np1-nokvu", "server -np 1 without -kvu (S2-session flags)", args=["-ngl", "99", "-fa", "on", "-np", "1", "-c", "4096"])
        S("env-disable-graphs", "GGML_CUDA_DISABLE_GRAPHS=1", env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
        B("bench-disable-graphs", "bench with GGML_CUDA_DISABLE_GRAPHS=1", env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
        S("env-graph-opt", "GGML_CUDA_GRAPH_OPT=1", env={"GGML_CUDA_GRAPH_OPT": "1"})
        S("env-disable-fusion", "GGML_CUDA_DISABLE_FUSION=1", env={"GGML_CUDA_DISABLE_FUSION": "1"})
        S("env-graph-reuse-disable", "LLAMA_GRAPH_REUSE_DISABLE=1", env={"LLAMA_GRAPH_REUSE_DISABLE": "1"})
        S("fa-off", "-fa off", args=["-ngl", "99", "-fa", "off", "-np", "1", "-kvu", "-c", "4096"])
        S("ctx2048", "-c 2048", args=["-ngl", "99", "-fa", "on", "-np", "1", "-kvu", "-c", "2048"])
        S("ctx8192", "-c 8192", args=["-ngl", "99", "-fa", "on", "-np", "1", "-kvu", "-c", "8192"])
        S("gen512", "n_predict 512 (generation length 256 -> 512)", n_predict=512)
        S("f16-weights", "f16 weights (converted from the bf16 GGUF)", model=F16)
        B("bench-f16", "bench f16 weights", model=F16)
        S("threads8", "-t 8 (explicit; bench default is 8)", args=BASE_ARGS + ["-t", "8"])
        B("bench-t4", "bench -t 4", args=["-ngl", "99", "-fa", "on", "-p", "0", "-n", "256", "-t", "4"])
    elif session == "B2":
        S("control", "server bf16 plain, CPU sampling (CONTROL)")
        S("backend-sampling", "-bs / --backend-sampling", args=BASE_ARGS + ["-bs"])
        S("bs-np-auto", "-bs without -np 1 -kvu (expected worse per PR #17004)", args=["-ngl", "99", "-fa", "on", "-c", "4096", "-bs"])
        S("np-auto", "no -bs, no -np 1/-kvu (reference for bs-np-auto)", args=["-ngl", "99", "-fa", "on", "-c", "4096"])
        S("req-nonsampling-topk1", "request-level top_k=1 (no -bs)", req={"top_k": 1})
        S("bs-topk1", "-bs + request top_k=1 (GPU top-k path)", args=BASE_ARGS + ["-bs"], req={"top_k": 1})
        S("bs-nograph", "-bs + GGML_CUDA_DISABLE_GRAPHS=1", args=BASE_ARGS + ["-bs"], env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
    elif session == "B3":
        mt = lambda k, p=None: ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)] + (["--spec-draft-p-min", str(p)] if p is not None else [])
        S("control", "server bf16 plain (CONTROL)")
        for k in (1, 2, 3):
            for p in (None, 0.5, 0.75):
                S("mtp-n%d-p%s" % (k, "def" if p is None else p), "bf16 target + native MTP (draft-mtp) n-max=%d p-min=%s" % (k, "default(0)" if p is None else p), args=BASE_ARGS + mt(k, p))
        S("mtp-n1-bs", "B5 combo: bf16 + draft-mtp n-max=1 + -bs (backend sampling)", args=BASE_ARGS + mt(1) + ["-bs"])
        S("q4mtp-control", "DIFFERENT TARGET: unsloth MTP Q4_K_M plain", model=Q4MTP)
        for k in (1, 2, 3):
            S("q4mtp-n%d-pdef" % k, "DIFFERENT TARGET: unsloth MTP Q4_K_M + draft-mtp n-max=%d" % k, model=Q4MTP, args=BASE_ARGS + mt(k))
        for k in (2,):
            S("q4draft-n%d" % k, "bf16 target + Q4_K_M(unsloth MTP file) as -md draft-simple n-max=%d p-min=0.6" % k, args=BASE_ARGS + ["--spec-type", "draft-simple", "-md", M(Q4), "--spec-draft-n-max", str(k), "--spec-draft-p-min", "0.6", "-ngld", "99"])
    for n, c in C.items(): c["name"] = n
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--only", nargs="*"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    C = configs(a.session)
    if a.list: print("\n".join(C)); return
    out = os.path.join(ROOT, "results", RAW[a.session])
    data = json.load(open(out)) if os.path.exists(out) else {"session": a.session, "configs": {}}
    for name in (a.only or list(C)):
        key = name + a.tag
        if key in data["configs"] and "error" not in data["configs"][key]: print("skip", key); continue
        c = C[name]
        try:
            data["configs"][key] = (run_server if c["kind"] == "server" else run_bench)(name, c, a.reps, a.tag)
        except SystemExit: raise
        except Exception as e:
            data["configs"][key] = {"name": name, "error": str(e)}; print("ERROR", key, e, flush=True)
        json.dump(data, open(out, "w")); time.sleep(3)
    print("DONE", flush=True)

if __name__ == "__main__": main()