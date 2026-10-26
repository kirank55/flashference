#!/usr/bin/env python3
"""Markdown table for E-series raw data.
python scripts\e_report.py --prefix e1- [--plain set=cfg,...] [--best set=cfg,...] [--ref cfg=refcfg,...] [--keys k1 k2 ...]
plain/best/ref are config keys in results\2026-10-27-bf16-E.raw.json. ratios use the pass-mean tok/s of the named config.
'out = X' = number of (prompt x pass) cells whose output text hash equals X's (same prompt, same pass index)."""
import argparse, json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "results", "2026-10-27-bf16-E.raw.json")))["configs"]

def passes(rec): return [sum(r["n"] for r in run) / (sum(r["ms"] for r in run) / 1000) for run in rec["runs"]]
def acc(rec):
    n = sum(r["draft_n"] for run in rec["runs"] for r in run); a = sum(r["draft_acc"] for run in rec["runs"] for r in run)
    return (100.0 * a / n) if n else None, n
def tokcyc(rec):
    nt = sum(r["n"] for run in rec["runs"] for r in run); nd = sum(r["draft_n"] for run in rec["runs"] for r in run)
    a = sum(r["draft_acc"] for run in rec["runs"] for r in run)
    return None
def hashes(rec): return [[r["hash"] for r in run] for run in rec["runs"]]
def eq(a, b):
    ha, hb = hashes(a), hashes(b); tot = ok = 0
    for ra, rb in zip(ha, hb):
        for x, y in zip(ra, rb): tot += 1; ok += (x == y)
    return "%d/%d" % (ok, tot)
def kv(s): return dict(x.split("=", 1) for x in s.split(",")) if s else {}

OK = {k for k, v in D.items() if 'runs' in v}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=""); ap.add_argument("--plain", default=""); ap.add_argument("--best", default=""); ap.add_argument("--ref", default=""); ap.add_argument("--keys", nargs="*")
    a = ap.parse_args(); plain, best, ref = kv(a.plain), kv(a.best), kv(a.ref)
    keys = a.keys or [k for k in D if k.startswith(a.prefix)]
    print("| config | set | tok/s mean +/- SD | vs plain (same set) | vs ref best | accept | out = plain | out = ref | what |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = D[k]
        if "error" in r: print("| %s | ERROR %s |" % (k, r["error"])); continue
        p = passes(r); m, sd = st.mean(p), st.pstdev(p) if len(p) < 2 else st.stdev(p)
        s = r["set"]; pk, bk, rk = plain.get(s), best.get(s), ref.get(k)
        vp = "%.3fx" % (m / st.mean(passes(D[pk]))) if pk in OK else "-"
        vb = "%.3fx" % (m / st.mean(passes(D[bk]))) if bk in OK else "-"
        ac, nd = acc(r)
        ep = eq(r, D[pk]) if pk in D and pk != k else ("-" if pk != k else "self")
        er = eq(r, D[rk]) if rk in OK else "-"
        print("| %s | %s | %.2f +/- %.2f | %s | %s | %s | %s | %s | %s |" % (k, s, m, sd, vp, vb, "-" if ac is None else "%.1f%%" % ac, ep, er, r.get("desc", "")))
if __name__ == "__main__": main()
