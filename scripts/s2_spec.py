#!/usr/bin/env python3
r"""S2a/S2b: speculative decoding on Qwen3.5-0.8B via llama-server (stdlib only).
Per config: fresh server, one discarded warm-up pass over all prompts, 5 timed passes; greedy, seed 1, cache_prompt=false.
Usage: python scripts\s2_spec.py [--only name1 name2 ...] [--list] [--reps 5]
Results: results/2026-10-26-s2-spec-0.8b.raw.json (resumable; configs already stored are skipped).
"""
import argparse, hashlib, json, os, subprocess, sys, time, statistics as st, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "llama-cpp", "llama-server.exe")
PORT = 8091
BASE = "http://127.0.0.1:%d" % PORT
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OUT = os.path.join(ROOT, "results", "2026-10-26-s2-spec-0.8b.raw.json")
Q4 = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
MTP = "Qwen3.5-0.8B-MTP-Q4_K_M.gguf"
PROMPTS = json.load(open(os.path.join(ROOT, "prompts", "prompts.json")))

def chatml(user):
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def http(path, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())

def complete(p):
    t0 = time.perf_counter()
    r = http("/completion", {"prompt": chatml(p["user"]), "n_predict": p["n_predict"], "temperature": 0.0, "seed": 1,
                             "cache_prompt": False, "stream": False})
    wall = time.perf_counter() - t0
    tm = r["timings"]
    return {"id": p["id"], "text": r["content"], "n": tm["predicted_n"], "ms": tm["predicted_ms"], "pp_ms": tm.get("prompt_ms"),
            "prompt_n": tm.get("prompt_n"), "draft_n": tm.get("draft_n", 0), "draft_acc": tm.get("draft_n_accepted", 0), "wall": wall}

def all_prompts():
    return [dict(p, type=t) for t in ("copy", "free") for p in PROMPTS[t]]

def idle():
    for _ in range(120):
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        u, m = [float(x) for x in o.split(",")]
        if u < 5 and m < 300: return {"util": u, "mem_mib": m}
        time.sleep(2)
    raise RuntimeError("GPU not idle")

def start_server(name, model, extra):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log = open(os.path.join(ROOT, "logs", "s2-%s.log" % name), "w")
    cmd = [SERVER, "-m", os.path.join(ROOT, "models", model), "-ngl", "99", "-fa", "on", "-np", "1", "-c", "4096",
           "--no-webui", "--host", "127.0.0.1", "--port", str(PORT)] + extra
    p = subprocess.Popen(cmd, stdout=log, stderr=log)
    for _ in range(120):
        time.sleep(1)
        if p.poll() is not None: raise RuntimeError("server exited early; see logs\\s2-%s.log" % name)
        try:
            if OPENER.open(BASE + "/health", timeout=2).status == 200: return p, " ".join(cmd[1:])
        except Exception: pass
    p.kill(); raise RuntimeError("server start timeout")

def configs():
    C = {}
    def add(name, model, extra, mode, params): C[name] = {"model": model, "extra": extra, "mode": mode, "params": params}
    add("control-q4km", Q4, [], "plain", "-")
    for mode in ("ngram-simple", "ngram-map-k"):
        key = {"ngram-simple": "simple", "ngram-map-k": "map-k"}[mode]
        for n in (2, 3, 4):
            for m in (4, 8, 16):
                add("%s-n%d-m%d" % (mode, n, m), Q4, ["--spec-type", mode, "--spec-ngram-%s-size-n" % key, str(n), "--spec-ngram-%s-size-m" % key, str(m)], mode, "n=%d m=%d" % (n, m))
    for n, m in ((2, 8), (3, 16)):
        add("ngram-map-k4v-n%d-m%d" % (n, m), Q4, ["--spec-type", "ngram-map-k4v", "--spec-ngram-map-k4v-size-n", str(n), "--spec-ngram-map-k4v-size-m", str(m)], "ngram-map-k4v", "n=%d m=%d" % (n, m))
    for nm in (3, 5, 8):
        add("ngram-mod-match%d" % nm, Q4, ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", str(nm), "--spec-ngram-mod-n-min", "4", "--spec-ngram-mod-n-max", "16"], "ngram-mod", "match=%d min=4 max=16" % nm)
    add("ngram-mod-default", Q4, ["--spec-type", "ngram-mod"], "ngram-mod", "defaults (match 24, min 48, max 64)")
    add("control-mtpfile", MTP, [], "plain", "- (MTP file, no spec)")
    for k in (1, 2, 3, 4, 6):
        add("mtp-nmax%d" % k, MTP, ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)], "draft-mtp", "n-max=%d" % k)
    for k in (1, 2, 3):
        add("bart-mtp-nmax%d" % k, Q4, ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)], "draft-mtp (bartowski file)", "n-max=%d" % k)
    for k in (2, 3):
        add("combo-mtp%d-ngramsimple-n3-m16" % k, MTP, ["--spec-type", "draft-mtp,ngram-simple", "--spec-draft-n-max", str(k), "--spec-ngram-simple-size-n", "3", "--spec-ngram-simple-size-m", "16"], "draft-mtp + ngram-simple", "mtp n-max=%d + ngram n=3 m=16" % k)
    return C
def run_config(name, c, reps, cold=False):
    idle_state = idle()
    proc, cmd = start_server(name, c["model"], c["extra"])
    rec = {"name": name, "mode": c["mode"], "params": c["params"], "model": c["model"], "cmd": cmd, "idle_before": idle_state, "runs": []}
    try:
        ps = all_prompts()
        if not cold:
            for p in ps: complete(p)  # warm-up pass, discarded (skipped with --cold: first pass over each prompt is recorded)
        for rep in range(reps):
            run = []
            for p in ps:
                r = complete(p); r["type"] = p["type"]
                if rep > 0: r["hash"] = hashlib.md5(r.pop("text").encode()).hexdigest()
                else: r["hash"] = hashlib.md5(r["text"].encode()).hexdigest()
                run.append(r)
            rec["runs"].append(run)
            print(name, "rep", rep + 1, "copy %.1f free %.1f" % tuple(setps(run, t) for t in ("copy", "free")), flush=True)
    finally:
        proc.terminate()
        try: proc.wait(15)
        except Exception: proc.kill()
    return rec

def setps(run, t):
    rs = [r for r in run if r["type"] == t]
    return sum(r["n"] for r in rs) / (sum(r["ms"] for r in rs) / 1000)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*"); ap.add_argument("--list", action="store_true"); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="no warm-up pass; use with --reps 1 and --tag=-cold to see learning effects of ngram-mod")
    ap.add_argument("--tag", default="", help="suffix for stored record name, e.g. --tag=-end for the end-of-session control")
    a = ap.parse_args()
    C = configs()
    if a.list:
        print("\n".join(C)); return
    data = json.load(open(OUT)) if os.path.exists(OUT) else {"configs": {}}
    for name in (a.only or list(C)):
        key = name + a.tag
        if key in data["configs"]: print("skip", key); continue
        try:
            data["configs"][key] = run_config(name, C[name], a.reps, a.cold)
        except Exception as e:
            data["configs"][key] = {"name": name, "error": str(e)}
            print("ERROR", key, e, flush=True)
        json.dump(data, open(OUT, "w"))
        time.sleep(2)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

