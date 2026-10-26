#!/usr/bin/env python3
"""Tables for the bf16 B1/B2/B3 sessions. Usage: python scripts\bf16_report.py B1|B2|B3  (prints markdown)"""
import gzip, json, math, os, statistics as st, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWN = {"B1": "2026-10-26-bf16-B1-overhead.raw.json", "B2": "2026-10-26-bf16-B2-backend-sampling.raw.json", "B3": "2026-10-26-bf16-B3-mtp.raw.json"}
REF = {"bench-disable-graphs": "bench-default", "bench-f16": "bench-default", "bench-t4": "bench-default", "bench-prev-style": "bench-default",
       "bs-np-auto": "np-auto", "bs-topk1": "req-nonsampling-topk1", "q4mtp-n1-pdef": "q4mtp-control", "q4mtp-n2-pdef": "q4mtp-control", "q4mtp-n3-pdef": "q4mtp-control",
       "f16-weights": "control", "gen512": "control"}

def load(s):
    p = os.path.join(ROOT, "results", RAWN[s])
    if os.path.exists(p + ".gz") and not os.path.exists(p): return json.load(gzip.open(p + ".gz", "rt", encoding="utf8"))
    return json.load(open(p, encoding="utf8"))

def passes(rec):
    return [sum(r["n"] for r in run) / (sum(r["ms"] for r in run) / 1000) for run in rec["runs"]]

def stats(rec):
    if rec["kind"] == "bench":
        t = rec["tests"][-1]; return t["avg_ts"], t["stddev_ts"], t["samples_ts"]
    x = passes(rec); return st.mean(x), (st.stdev(x) if len(x) > 1 else 0.0), x

def ratio(m, s, cm, cs):
    r = m / cm; return r, r * math.sqrt((s / m) ** 2 + (cs / cm) ** 2)

def mtpstats(rec):
    n = sum(r["n"] for run in rec["runs"] for r in run); ms = sum(r["ms"] for run in rec["runs"] for r in run)
    dn = sum(r["draft_n"] for run in rec["runs"] for r in run); da = sum(r["draft_acc"] for run in rec["runs"] for r in run)
    if not dn: return None
    cycles = n - da
    return {"acc": da / dn, "drafted_per_tok": dn / n, "tok_per_cycle": n / cycles, "cycle_ms": ms / cycles}

def loops(rec, ref=None):
    run = rec["runs"][0]
    fl = [r["id"] for r in run if r["loop"]]
    return "%d/6 %s" % (len(fl), ",".join(fl)) if fl else "0/6"

def same(rec, ctl):
    if rec["kind"] != "server" or ctl["kind"] != "server": return "n/a"
    a = [[r["hash"] for r in run] for run in rec["runs"]]; b = ctl["runs"][0]
    ok = sum(1 for run in a for i, h in enumerate(run) if h == b[i]["hash"]); return "%d/%d" % (ok, sum(len(r) for r in a))

def main(s):
    D = load(s)["configs"]; C = D.get("control"); E = D.get("control-end")
    if not C: print("no control"); return
    cm, cs, _ = stats(C)
    if E and "error" not in E:
        em, es, _ = stats(E); rm = (cm + em) / 2; rs = math.sqrt(cs ** 2 + es ** 2) / 2
        print("Control (server, bf16 plain): start %.2f +/- %.2f, end %.2f +/- %.2f tok/s; drift %.1f%%; bracket mean %.2f\n" % (cm, cs, em, es, 100 * (em - cm) / cm, rm))
    else: rm, rs = cm, cs; print("Control: %.2f +/- %.2f (no end control yet)\n" % (cm, cs))
    hdr = "| config | what | tok/s mean +/- SD | speedup vs control | vs its reference | greedy output = control | loops (pass 1) | extra |\n|---|---|---|---|---|---|---|---|"
    print(hdr)
    for k, rec in D.items():
        if "error" in rec: print("| %s | ERROR: %s | | | | | | |" % (k, rec["error"][:80])); continue
        m, sd, _ = stats(rec); r, rsd = ratio(m, sd, rm, rs)
        ref = REF.get(k); rt = "-"
        if ref and ref in D and "error" not in D[ref]:
            rr = ratio(m, sd, *stats(D[ref])[:2]); rt = "%.3fx +/- %.3f (vs %s)" % (rr[0], rr[1], ref)
        ex = ""
        if rec["kind"] == "server":
            ms_ = mtpstats(rec)
            if ms_: ex = "accept %.1f%%, %.2f tok/cycle, cycle %.2f ms" % (100 * ms_["acc"], ms_["tok_per_cycle"], ms_["cycle_ms"])
            ex += (" n_predict=%d" % rec["n_predict"]) if rec["n_predict"] != 256 else ""
        else:
            t = rec["tests"][-1]; ex = "bench threads=%s pp=%d tg=%d" % (t["n_threads"], t["n_prompt"], t["n_gen"])
        print("| %s | %s | %.2f +/- %.2f | %.3fx +/- %.3f | %s | %s | %s | %s |" % (k, rec["desc"], m, sd, r, rsd, rt, same(rec, C), loops(rec) if rec["kind"] == "server" else "-", ex))

if __name__ == "__main__": main(sys.argv[1])