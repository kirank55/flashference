#!/usr/bin/env python3
r"""E-series (2026-10-27) experiments on bf16 Qwen3.5-0.8B: reuses scripts/bf16_bench.py methodology
(fresh server per config, GPU-idle check, 1 discarded warm-up pass + 5 timed passes, chat template, thinking off, cache_prompt=false, n_predict 256).
Each config names a prompt SET: free (F1-F6), code (K1-K4), lang (L1-L4), rep (copy-heavy C1-C6 from prompts.json).
Usage: python scripts\e_bench.py --only name ... [--list] [--tag SUFFIX] [--reps 5] [--session E1]
Raw data: results\2026-10-27-bf16-E.raw.json (keyed by config name+tag). Resumable.
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16_bench as bb

ROOT = bb.ROOT
OUT = os.path.join(ROOT, "results", "2026-10-27-bf16-E.raw.json")
_P = json.load(open(os.path.join(ROOT, "prompts", "prompts.json"), encoding="utf-8-sig"))
_E = json.load(open(os.path.join(ROOT, "prompts", "e-prompts.json"), encoding="utf-8-sig"))
SETS = {"free": _P["free"], "rep": _P["copy"], "code": _E["code"], "lang": _E["lang"]}
RANK = bb.RANK
RANK_ENV = lambda K, f=None: {"LLAMA_MTP_TRIM_FILE": f or RANK, "LLAMA_MTP_TRIM_K": str(K)}
OWN = "llama-cpp-b4"
BA = bb.BASE_ARGS
mt = lambda k, extra=None: ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k), "-bs"] + (extra or [])
T07 = {"temperature": 0.7, "top_p": 0.9, "seed": 1}

def configs():
    C = {}
    def S(name, desc, set_, args=None, env=None, lc=None, req=None, **kw):
        C[name] = dict(kind="server", desc=desc, set=set_, model=bb.BF16, args=BA if args is None else args, env=env, lc=lc, req=req, **kw)
    # ---- E1: robustness of B4 -------------------------------------------------
    for st in ("free", "code", "lang"):
        S("e1-%s-plain" % st, "%s: plain bf16 (release build)" % st, st)
        S("e1-%s-untrim-n2" % st, "%s: MTP n-max 2 -bs, untrimmed (own build)" % st, st, BA + mt(2), lc=OWN)
        S("e1-%s-trim8k-n2" % st, "%s: MTP n-max 2 -bs, trimmed 8k (own build)" % st, st, BA + mt(2), RANK_ENV(8192), OWN)
        S("e1-%s-trim32k-n2" % st, "%s: MTP n-max 2 -bs, trimmed 32k (own build)" % st, st, BA + mt(2), RANK_ENV(32768), OWN)
    for st in ("code", "lang"):
        S("e1-%s-trim8k-n1" % st, "%s: MTP n-max 1 -bs, trimmed 8k" % st, st, BA + mt(1), RANK_ENV(8192), OWN)
    # temperature 0.7 top-p 0.9 (seed 1) on free text
    S("e1-t07-plain", "free T=0.7 top-p 0.9: plain bf16 (release, CPU sampling)", "free", req=T07)
    S("e1-t07-plain-bs", "free T=0.7 top-p 0.9: plain bf16 with -bs (own build)", "free", BA + ["-bs"], lc=OWN, req=T07)
    for dm in ("greedy", "probabilistic"):
        S("e1-t07-untrim-n2-%s" % dm, "free T=0.7: MTP n-max 2 -bs untrimmed, draft sampling %s" % dm, "free", BA + mt(2, ["--spec-draft-sampling", dm]), lc=OWN, req=T07)
        S("e1-t07-trim8k-n2-%s" % dm, "free T=0.7: MTP n-max 2 -bs trimmed 8k, draft sampling %s" % dm, "free", BA + mt(2, ["--spec-draft-sampling", dm]), RANK_ENV(8192), OWN, req=T07)
    S("e1-t07-trim8k-n1-greedy", "free T=0.7: MTP n-max 1 -bs trimmed 8k, draft sampling greedy", "free", BA + mt(1), RANK_ENV(8192), OWN, req=T07)
    for n, c in C.items(): c["name"] = n
    return C

def wait_idle(maxwait=3600):
    """wait (poll 15 s) until the GPU is idle (an external process was seen using it on 2026-10-27 ~00:20); no system setting is touched"""
    t0 = time.time()
    while True:
        s = bb.gpu_check()
        if s["util"] <= 2 and s["mem_mib"] < 300 and not s["stray"]: return
        if time.time() - t0 > maxwait: raise SystemExit("GPU busy for %d s: %s" % (maxwait, s))
        print("waiting for idle GPU:", s, flush=True); time.sleep(15)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*"); ap.add_argument("--list", action="store_true"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--tag", default="")
    ap.add_argument("--module", default="")
    a = ap.parse_args()
    C = configs()
    if a.module:
        mod = __import__(a.module); C.update(mod.configs(sys.modules[__name__]))
        for n, c in C.items(): c["name"] = n
    if a.list: print("\n".join(C)); return
    data = json.load(open(OUT)) if os.path.exists(OUT) else {"configs": {}}
    for name in (a.only or list(C)):
        key = name + a.tag
        if key in data["configs"] and "error" not in data["configs"][key]: print("skip", key); continue
        c = C[name]
        bb.PROMPTS = SETS[c["set"]]
        try:
            for attempt in range(3):
                wait_idle()
                rec = bb.run_server(name, c, a.reps, a.tag)
                ps = [bb.agg(run) for run in rec["runs"]]
                med = sorted(ps)[len(ps) // 2]
                rec["attempt"] = attempt + 1; rec["spread_pct"] = round(100 * (max(ps) - min(ps)) / med, 2)
                rec["set"] = c["set"]; rec["req"] = c.get("req")
                if (rec["spread_pct"] <= 4.0 and min(ps) >= 55.0) or a.reps < 3: break   # <55 tok/s = slower than plain bf16 = external GPU contention (seen with a steady 16 tok/s)
                print("NOISY (spread %.1f%%) -> retry" % rec["spread_pct"], flush=True)
                rec["noisy_discarded"] = True
                time.sleep(20)
            data["configs"][key] = rec
        except SystemExit: raise
        except Exception as e:
            data["configs"][key] = {"name": name, "error": str(e)}; print("ERROR", key, e, flush=True)
        cur = json.load(open(OUT)) if os.path.exists(OUT) else {"configs": {}}   # re-read: other queues may have written meanwhile
        cur["configs"][key] = data["configs"][key]; data = cur
        json.dump(data, open(OUT, "w")); time.sleep(3)
    print("DONE", flush=True)

if __name__ == "__main__": main()

