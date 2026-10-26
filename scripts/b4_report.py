#!/usr/bin/env python3
"""Tables for the bf16 B4 session (trimmed MTP draft head). Usage: python scripts\b4_report.py  (prints markdown)"""
import json, math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf16_report import stats, ratio, mtpstats, loops, ROOT
D = json.load(open(os.path.join(ROOT, "results", "2026-10-26-bf16-b4-trim-draft.raw.json"), encoding="utf8"))["configs"]
# adjacent control brackets (run order): block 1 = control .. trim32k-n1 ; block 2a = mid .. trim2k-n2 ; block 2b = mid2 .. own-control, end
BR = {}
B1 = ["control", "mtp-n1-bs", "own-n1-bs", "trim16k-n1-bs", "trim8k-n1-bs", "trim32k-n1-bs"]
B2A = ["trim8k-n2-bs", "trim16k-n2-bs", "trim32k-n2-bs", "trim4k-n1-bs", "trim2k-n1-bs", "trim4k-n2-bs", "trim2k-n2-bs"]
B2B = ["mtp-n2-bs", "own-n2-bs", "trim16k-n1-nobs", "own-control"]
B3X = ["trim8k-n3-bs", "trim8k-n4-bs"]
for k in B3X: BR[k] = ("control-c3a", "control-c3b")
for k in B1: BR[k] = ("control", "control-mid")
for k in B2A: BR[k] = ("control-mid", "control-mid2")
for k in B2B: BR[k] = ("control-mid2", "control-end")
def brk(a, b):
    if a not in D or b not in D or "error" in D[a] or "error" in D[b]: return None
    am, as_, _ = stats(D[a]); bm, bs, _ = stats(D[b]); return (am + bm) / 2, math.sqrt(as_ ** 2 + bs ** 2) / 2
def hashes(rec): return [[r["hash"] for r in run] for run in rec["runs"]]
def same(rec, ref):
    if ref not in D or "error" in D[ref]: return "n/a"
    rb = hashes(D[ref])[0]; ok = sum(1 for run in hashes(rec) for i, h in enumerate(run) if h == rb[i]); return "%d/%d" % (ok, sum(len(r) for r in hashes(rec)))
print("Controls (release build, plain bf16 server): " + ", ".join("%s %.2f +/- %.2f" % (k, *stats(D[k])[:2]) for k in ("control", "control-mid", "control-mid2", "control-end", "control-c3a", "control-c3b") if k in D and "error" not in D[k]) + "\n")
print("| config | what | tok/s mean +/- SD | speedup vs adjacent control bracket | vs previous best (mtp-n1-bs, same block) | accept | tok/cycle | cycle ms | output = plain control | output = untrimmed MTP (mtp-n1-bs / own-n1-bs) |\n|---|---|---|---|---|---|---|---|---|---|")
order = [k for k in D if not k.startswith("control")]
for k in B1[1:] + B2A + B2B + B3X:
    if k not in D: continue
    rec = D[k]
    if "error" in rec: print("| %s | ERROR %s |" % (k, rec["error"][:80])); continue
    m, sd, _ = stats(rec); b = brk(*BR[k]); x = mtpstats(rec)
    r = ratio(m, sd, *b) if b else (float("nan"), 0)
    base = "mtp-n1-bs" if k in B1 else None
    vb = "%.3fx" % (m / stats(D[base])[0]) if base and base in D else "-"
    refu = ("mtp-n2-bs" if k.endswith("n2-bs") else "mtp-n1-bs")
    refo = ("own-n2-bs" if k.endswith("n2-bs") else "own-n1-bs")
    print("| %s | %s | %.2f +/- %.2f | %.3fx +/- %.3f | %s | %s | %s | %s | %s | %s / %s |" % (k, rec["desc"], m, sd, r[0], r[1], vb,
          ("%.1f%%" % (100 * x["acc"])) if x else "-", ("%.2f" % x["tok_per_cycle"]) if x else "-", ("%.2f" % x["cycle_ms"]) if x else "-",
          same(rec, "control"), same(rec, refu), same(rec, refo)))
