#!/usr/bin/env python3
"""Self-contained benchmark for the two headline configs (stdlib only, Windows/Linux).

For each prompt set it starts a FRESH llama-server per config, runs one discarded warm-up pass, then N timed passes
(greedy, chat template, thinking off, cache_prompt=false) and prints tok/s (mean +/- SD over passes), acceptance
and the ratio vs the plain bf16 run of the SAME prompt set. Normally started through scripts/reproduce.ps1 / reproduce.sh,
which build/locate the patched llama.cpp and the model first.

  python scripts/reproduce_bench.py --llama-bin <dir with llama-server> --model <bf16.gguf> [--passes 3] [--smoke] [--include-prose]
"""
import argparse, hashlib, json, math, os, shutil, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "results", "2026-10-26-bf16-b4-token-rank.txt")
BASE = ["-ngl", "99", "-fa", "on", "-np", "1", "-kvu", "-c", "4096"]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ENV_CLEAR = ["LLAMA_MTP_TRIM_FILE", "LLAMA_MTP_TRIM_K", "LLAMA_MTP_TRIM_FIXED", "LLAMA_MTP_DYN",
             "LLAMA_MTP_ADAPT", "LLAMA_MTP_ADAPT_LO", "LLAMA_MTP_ADAPT_INIT"]


def load_prompts():
    rd = lambda f: json.load(open(os.path.join(ROOT, "prompts", f), encoding="utf-8-sig"))
    p, e = rd("prompts.json"), rd("e-prompts.json")
    return {"code": e["code"], "copy": p["copy"], "prose": p["free"]}


def trim_env(k, **extra):
    d = {"LLAMA_MTP_TRIM_FILE": RANK, "LLAMA_MTP_TRIM_K": str(k)}
    d.update({a: str(b) for a, b in extra.items()})
    return d


def configs(include_prose):
    mtp = lambda n: ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n), "-bs"]
    ngram_mtp = ["--spec-type", "ngram-simple,draft-mtp", "--spec-draft-n-max", "8", "-bs",
                 "--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "8"]
    C = [
        ("code-plain", "code", "plain bf16", BASE, {}),
        ("code-mtp-trim32k", "code", "HEADLINE code: MTP, 32k draft head, 2 drafts, -bs", BASE + mtp(2), trim_env(32768)),
        ("copy-plain", "copy", "plain bf16", BASE, {}),
        ("copy-mtp-trim8k", "copy", "MTP alone, 8k draft head, 2 drafts, -bs (control)", BASE + mtp(2), trim_env(8192)),
        ("copy-ngram-mtp", "copy", "HEADLINE copy-heavy: ngram-simple,draft-mtp", BASE + ngram_mtp,
         trim_env(8192, LLAMA_MTP_ADAPT=1, LLAMA_MTP_ADAPT_INIT=2)),
    ]
    if include_prose:
        C += [
            ("prose-plain", "prose", "plain bf16", BASE, {}),
            ("prose-mtp-trim8k", "prose", "MTP, 8k draft head, 2 drafts, -bs (good for prose)", BASE + mtp(2), trim_env(8192)),
            ("prose-ngram-mtp", "prose", "ngram-simple,draft-mtp (NOT for prose)", BASE + ngram_mtp,
             trim_env(8192, LLAMA_MTP_ADAPT=1, LLAMA_MTP_ADAPT_INIT=2)),
        ]
    return C


def chatml(u):
    return "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def http(base, path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(base, p, n_predict):
    r = http(base, "/completion", {"prompt": chatml(p["user"]), "n_predict": n_predict, "temperature": 0.0, "seed": 1,
                                   "cache_prompt": False, "stream": False})
    t = r["timings"]
    return {"id": p["id"], "text": r["content"], "n": t["predicted_n"], "ms": t["predicted_ms"],
            "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0)}


def gpu_state():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        n, u, m = [x.strip() for x in o.split(",")]
        return {"name": n, "util": float(u), "mem_mib": float(m)}
    except Exception:
        return None


def wait_idle(require, maxwait=90):
    t0 = time.time()
    while True:
        s = gpu_state()
        if s is None or (s["util"] <= 5 and s["mem_mib"] < 400):
            return s
        if time.time() - t0 > maxwait:
            if require:
                raise SystemExit("GPU busy (%s) - aborting (--require-idle)" % s)
            print("WARNING: GPU not idle (%s); numbers may be unreliable" % s, flush=True)
            return s
        time.sleep(5)


def start_server(exe, model, args, env, port, logpath):
    e = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR}
    e.update(env)
    log = open(logpath, "w")
    cmd = [exe, "-m", model] + args + ["--no-webui", "--host", "127.0.0.1", "--port", str(port)]
    p = subprocess.Popen(cmd, stdout=log, stderr=log, env=e)
    base = "http://127.0.0.1:%d" % port
    for _ in range(240):
        time.sleep(1)
        if p.poll() is not None:
            raise RuntimeError("llama-server exited early, see " + logpath)
        try:
            if OPENER.open(base + "/health", timeout=2).status == 200:
                return p, base, " ".join(cmd[1:])
        except Exception:
            pass
    p.kill()
    raise RuntimeError("llama-server start timeout, see " + logpath)


def stop(p):
    p.terminate()
    try:
        p.wait(20)
    except Exception:
        p.kill()


def pass_tps(run):
    return sum(r["n"] for r in run) / (sum(r["ms"] for r in run) / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bin", required=True, help="directory containing llama-server (patched build)")
    ap.add_argument("--model", required=True, help="Qwen3.5-0.8B bf16 GGUF (with MTP block)")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="1 timed pass, 64 tokens, 1 warm-up prompt, code+copy only")
    ap.add_argument("--include-prose", action="store_true", help="also run the ordinary-prose sets (shows when NOT to use ngram)")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--out", default=os.path.join(ROOT, "reproduce-out"))
    ap.add_argument("--require-idle", action="store_true", help="abort instead of warning when the GPU is busy")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    exe = os.path.join(a.llama_bin, "llama-server.exe" if os.name == "nt" else "llama-server")
    if not os.path.exists(exe):
        sys.exit("llama-server not found at " + exe)
    if not os.path.exists(a.model):
        sys.exit("model not found: " + a.model)
    if not os.path.exists(RANK):
        sys.exit("token-rank file missing: " + RANK)
    if a.smoke:
        a.passes, a.n_predict = 1, 64
    P = load_prompts()
    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results, plain_texts = {}, {}
    cfgs = [c for c in configs(a.include_prose and not a.smoke) if not (a.smoke and c[1] == "prose")]
    print("llama-server: %s\nmodel: %s\npasses=%d n_predict=%d%s\n" % (exe, a.model, a.passes, a.n_predict, "  [SMOKE]" if a.smoke else ""), flush=True)
    for name, st, desc, args, env in cfgs:
        gs = wait_idle(a.require_idle)
        proc, base, cmd = start_server(exe, a.model, args, env, a.port, os.path.join(a.out, "server-%s-%s.log" % (name, stamp)))
        try:
            warm = P[st][:1] if a.smoke else P[st]
            for p in warm:
                complete(base, p, a.n_predict)
            passes, texts = [], {}
            for i in range(a.passes):
                run = [complete(base, p, a.n_predict) for p in P[st]]
                passes.append(run)
                for r in run:
                    texts.setdefault(r["id"], []).append(hashlib.md5(r["text"].encode("utf-8")).hexdigest())
                acc = sum(r["draft_acc"] for r in run); dn = sum(r["draft_n"] for r in run)
                print("%-18s pass %d: %7.2f tok/s%s" % (name, i + 1, pass_tps(run), ("  accept %.1f%%" % (100.0 * acc / dn)) if dn else ""), flush=True)
        finally:
            stop(proc)
        tps = [pass_tps(r) for r in passes]
        dn = sum(r["draft_n"] for run in passes for r in run); acc = sum(r["draft_acc"] for run in passes for r in run)
        results[name] = {"set": st, "desc": desc, "cmd": cmd, "env": {k: v for k, v in env.items() if k != "LLAMA_MTP_TRIM_FILE"},
                         "tps": tps, "mean": statistics.mean(tps), "sd": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
                         "accept": (100.0 * acc / dn) if dn else None, "hashes": texts, "gpu": gs}
        if name.endswith("-plain"):
            plain_texts[st] = {k: v[0] for k, v in texts.items()}
    # report
    rows = ["| config | set | what | tok/s (mean +/- SD) | vs plain (same set) | acceptance | output == plain |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        pl = results.get(r["set"] + "-plain")
        ratio = r["mean"] / pl["mean"] if pl else float("nan")
        same = tot = 0
        for pid, hs in r["hashes"].items():
            for h in hs:
                tot += 1; same += (h == plain_texts.get(r["set"], {}).get(pid))
        rows.append("| %s | %s | %s | %.2f +/- %.2f | %.2fx | %s | %d/%d |" % (
            name, r["set"], r["desc"], r["mean"], r["sd"], ratio, ("%.1f%%" % r["accept"]) if r["accept"] is not None else "-", same, tot))
    table = "\n".join(rows)
    print("\n" + table)
    print("\nNotes: ratios are vs the plain bf16 run of the same prompt set in this invocation (absolute tok/s are hardware specific).")
    print("'output == plain' counts (prompt x pass) outputs byte-identical to the plain greedy output; batched verification can change fp rounding, so < 100% is expected (see README).")
    if a.smoke:
        print("SMOKE mode (1 pass, %d tokens): ratios are indicative only." % a.n_predict)
    with open(os.path.join(a.out, "reproduce-%s.json" % stamp), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "results": results}, f, indent=1)
    with open(os.path.join(a.out, "reproduce-%s.md" % stamp), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("saved: %s" % os.path.join(a.out, "reproduce-%s.{json,md}" % stamp))


if __name__ == "__main__":
    main()
