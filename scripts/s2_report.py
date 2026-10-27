#!/usr/bin/env python3
"""Tables for S2 from results/2026-10-26-s2-spec-0.8b.raw.json(.gz) -> prints markdown (redirect into docs/experiments/2026-10-26-s2-spec-0.8b.md).
Control handling: speed-up = mean decode tok/s of a config / pooled mean of the plain controls of the same model file listed for that table
(start/end/end2 controls); drift = (max-min)/mean of the control means. Reps whose decode speed is < 60% of the median rep of the same config
(machine interruption, e.g. the PC going unreachable) are excluded and listed."""
import json, os, re, statistics as st, sys, math, gzip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "2026-10-26-s2-spec-0.8b.raw.json")
D = json.load(gzip.open(RAW + ".gz", "rt") if os.path.exists(RAW + ".gz") and not os.path.exists(RAW) else open(RAW))["configs"]
P = json.load(open(os.path.join(ROOT, "prompts", "prompts.json")))
TYPES = ("copy", "free")
EXCL = []

def runs_ok(rec, t):
    tp = []
    for run in rec["runs"]:
        rs = [r for r in run if r["type"] == t]; tp.append(sum(r["n"] for r in rs) / sum(r["ms"] for r in rs) * 1000)
    if len(tp) < 3: return list(range(len(tp)))
    med = st.median(tp); ok = [i for i, x in enumerate(tp) if x >= 0.6 * med]
    for i, x in enumerate(tp):
        if i not in ok: EXCL.append("%s %s rep %d (%.0f vs median %.0f tok/s)" % (rec["name"], t, i + 1, x, med))
    return ok

def per_run(rec, t):
    out = []
    for i in runs_ok(rec, t):
        rs = [r for r in rec["runs"][i] if r["type"] == t]
        n = sum(r["n"] for r in rs); ms = sum(r["ms"] for r in rs); dn = sum(r["draft_n"] for r in rs); da = sum(r["draft_acc"] for r in rs); wall = sum(r["wall"] for r in rs)
        out.append({"tps": n / ms * 1000, "wall_tps": n / wall, "n": n, "dn": dn, "da": da})
    return out

def ms(xs): return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

def summ(rec, t):
    pr = per_run(rec, t); n = sum(x["n"] for x in pr); dn = sum(x["dn"] for x in pr); da = sum(x["da"] for x in pr)
    return {"tps": ms([x["tps"] for x in pr]), "acc": da / dn if dn else None, "tps_step": n / (n - da) if n > da else None, "wall": ms([x["wall_tps"] for x in pr]), "dpt": dn / n if n else 0}

def ctrl(keys, t):
    ks = [k for k in keys if k in D and "runs" in D[k]]
    S = [summ(D[k], t) for k in ks]
    means = [s["tps"][0] for s in S]
    return {"keys": ks, "mean": st.mean(means), "sd": max(s["tps"][1] for s in S), "means": means, "drift": (max(means) - min(means)) / st.mean(means) if len(means) > 1 else None,
            "wall": st.mean(s["wall"][0] for s in S)}

def same_text(rec, ck, t):
    ref = D[ck[0]]["runs"][0]; cs = {r["id"]: r["hash"] for r in ref}
    ids = [r for r in rec["runs"][0] if r["type"] == t]
    return sum(1 for r in ids if cs.get(r["id"]) == r["hash"]) / len(ids)

def table(names, ckeys, title, note=""):
    names = [n for n in names if n in D]
    if not names or not any(k in D and "runs" in D[k] for k in ckeys): return
    print("\n### " + title)
    if note: print(note)
    print("| config | params | type | accept | tokens/step | r = tok/step / speed-up | decode tok/s mean +/- SD (5 reps) | speed-up vs plain mean +/- SD | wall-based speed-up | identical text vs plain (rep 1) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in TYPES:
        c = ctrl(ckeys, t)
        print("| plain (%s) | - | %s | n/a | 1.00 | 1.00 | %.1f +/- %.1f (control means %s; drift %s) | 1.00x | 1.00x | - |" % (" + ".join(c["keys"]), t, c["mean"], c["sd"], "/".join("%.0f" % x for x in c["means"]),
              ("%.1f%%" % (100 * c["drift"])) if c["drift"] is not None else "n/a"))
    for nme in names:
        rec = D[nme]
        if "error" in rec:
            print("| %s | - | - | ERROR: %s | | | | | | |" % (nme, rec["error"][:100])); continue
        for t in TYPES:
            c = ctrl(ckeys, t); s = summ(rec, t)
            sp = s["tps"][0] / c["mean"]; spsd = sp * math.sqrt((s["tps"][1] / s["tps"][0]) ** 2 + (c["sd"] / c["mean"]) ** 2)
            r = (s["tps_step"] / sp) if s["tps_step"] else None
            print("| %s | %s | %s | %s | %s | %s | %.1f +/- %.1f | %.2fx +/- %.2f | %.2fx | %d%% |" % (nme, rec["params"], t, ("%.1f%%" % (100 * s["acc"])) if s["acc"] is not None else "n/a",
                  ("%.2f" % s["tps_step"]) if s["tps_step"] else "n/a", ("%.2f" % r) if r else "n/a", s["tps"][0], s["tps"][1], sp, spsd, s["wall"][0] / c["wall"], 100 * same_text(rec, c["keys"], t)))

def best_table():
    fams = [("ngram-simple", "ngram-simple-n", ["control-q4km", "control-q4km-end", "control-q4km-end2"]), ("ngram-map-k", "ngram-map-k-n", ["control-q4km", "control-q4km-end", "control-q4km-end2"]),
            ("ngram-map-k4v", "ngram-map-k4v", ["control-q4km", "control-q4km-end", "control-q4km-end2"]), ("ngram-mod (warm)", "ngram-mod", ["control-q4km", "control-q4km-end", "control-q4km-end2"]),
            ("draft-mtp (unsloth MTP file, pass 2)", "mtp-nmax", ["control-mtpfile-end", "control-mtpfile-end2"]), ("draft-mtp + ngram-simple (pass 2)", "combo", ["control-mtpfile-end", "control-mtpfile-end2"])]
    print("\n### Best setting per family and task type (highest speed-up vs plain)\n")
    print("| family | type | best config | accept | tokens/step | speed-up vs plain | decode tok/s |")
    print("|---|---|---|---|---|---|---|")
    for fam, pre, ck in fams:
        for t in TYPES:
            best = None
            c = ctrl(ck, t)
            if not c["keys"]: continue
            for k, rec in D.items():
                if "runs" not in rec or not k.startswith(pre) or k.endswith("-cold") or k.endswith("-end") or k.endswith("-end2"): continue
                if fam.startswith("draft-mtp (") and not k.endswith("-r2"): continue
                if fam.startswith("draft-mtp +") and not k.endswith("-r2"): continue
                if fam.startswith("ngram-map-k") and fam == "ngram-map-k" and k.startswith("ngram-map-k4v"): continue
                if fam == "ngram-mod (warm)" and not k.startswith("ngram-mod"): continue
                s = summ(rec, t); sp = s["tps"][0] / c["mean"]
                if best is None or sp > best[0]: best = (sp, k, s)
            if best:
                sp, k, s = best
                print("| %s | %s | %s | %s | %s | %.2fx | %.0f |" % (fam, t, k, ("%.0f%%" % (100 * s["acc"])) if s["acc"] is not None else "n/a", ("%.2f" % s["tps_step"]) if s["tps_step"] else "n/a", sp, s["tps"][0]))

def maxes():
    print("\n### Highest single-reply decode speed seen (one prompt, one rep, >= 100 tokens; decode tok/s = predicted_n / predicted_ms; warm ngram-mod excluded for free text because it replays identical earlier replies)\n")
    print("| type | tok/s | config | prompt | rep | tokens |")
    print("|---|---|---|---|---|---|")
    for t in TYPES:
        best = None
        for k, rec in D.items():
            if "runs" not in rec: continue
            for i, run in enumerate(rec["runs"]):
                for r in run:
                    if r["type"] != t or r["n"] < 100 or (t == "free" and k.startswith("ngram-mod") and not k.endswith("-cold")): continue
                    v = r["n"] / r["ms"] * 1000
                    if best is None or v > best[0]: best = (v, k, r["id"], i + 1, r["n"])
        print("| %s | %.0f | %s | %s | %d | %d |" % ((t,) + best))
    for t in TYPES:
        v = max(((summ(r, t)["tps"][0], k) for k, r in D.items() if "runs" in r and not k.endswith("-cold")), key=lambda x: x[0])
        print("\nBest configuration mean over all %s prompts (5 reps): %.1f tok/s (%s)" % (t, v[0], v[1]))

def cold_table():
    cold = [k for k in D if k.endswith("-cold") and "runs" in D[k]]
    if not cold: return
    print("\n### ngram-mod learns across requests: cold (no warm-up, first pass over each prompt, 1 rep) vs warm (warm-up pass + 5 reps over the same prompts)\n")
    print("| config | type | cold decode tok/s | cold accept | cold tokens/step | warm decode tok/s (5 reps) | warm accept | warm tokens/step | rep-1 vs rep-5 warm tok/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in cold:
        base = k[:-5]
        for t in TYPES:
            a = summ(D[k], t); b = summ(D[base], t)
            tp = [x["tps"] for x in per_run(D[base], t)]
            print("| %s | %s | %.0f | %.0f%% | %.2f | %.0f | %.0f%% | %.2f | %.0f -> %.0f |" % (base, t, a["tps"][0], 100 * (a["acc"] or 0), a["tps_step"] or 1, b["tps"][0], 100 * (b["acc"] or 0), b["tps_step"] or 1, tp[0], tp[-1]))

def copy_fraction(best, ck):
    print("\n### Copy fraction vs n-gram acceptance per prompt (%s; copy fraction = share of output word 4-grams already in prompt + earlier output of the plain control reply)\n" % best)
    ctl = D["control-q4km"]; rec = D.get(best)
    allp = {p["id"]: p for t in TYPES for p in P[t]}
    print("| prompt | type | copy fraction | %s accept | tokens/step | per-prompt speed-up vs plain control (same prompt) |" % best)
    print("|---|---|---|---|---|---|")
    for r in ctl["runs"][0]:
        ow = re.findall(r"[a-z0-9']+", r["text"].lower()); pw = re.findall(r"[a-z0-9']+", allp[r["id"]]["user"].lower())
        seen = set(tuple(pw[i:i + 4]) for i in range(len(pw) - 3)); hit = tot = 0
        for i in range(len(ow) - 3):
            g = tuple(ow[i:i + 4]); tot += 1; hit += g in seen; seen.add(g)
        cf = hit / tot if tot else 0
        if rec and "runs" in rec:
            rs = [x for run in rec["runs"] for x in run if x["id"] == r["id"]]
            cs = [x for k in ck if k in D for run in D[k]["runs"] for x in run if x["id"] == r["id"]]
            dn = sum(x["draft_n"] for x in rs); da = sum(x["draft_acc"] for x in rs); n = sum(x["n"] for x in rs)
            sp = (n / sum(x["ms"] for x in rs)) / (sum(x["n"] for x in cs) / sum(x["ms"] for x in cs))
            print("| %s | %s | %.2f | %s | %.2f | %.2fx |" % (r["id"], r["type"], cf, ("%.0f%%" % (100 * da / dn)) if dn else "n/a", n / (n - da) if n > da else 1.0, sp))


def repro_table():
    pairs = [k[:-3] for k in D if k.endswith("-r2") and "runs" in D[k] and k[:-3] in D and "runs" in D[k[:-3]]]
    print("\n### Reproducibility: pass 1 vs repeat (-r2) of the same config (acceptance and tokens/step are machine-independent; tok/s depends on machine state)\n")
    print("| config | type | accept p1 | accept p2 | tokens/step p1 | tokens/step p2 | decode tok/s p1 | decode tok/s p2 |")
    print("|---|---|---|---|---|---|---|---|")
    for k in pairs:
        for t in TYPES:
            a = summ(D[k], t); b = summ(D[k + "-r2"], t)
            f = lambda s: ("%.1f%%" % (100 * s["acc"])) if s["acc"] is not None else "n/a"
            g = lambda s: ("%.2f" % s["tps_step"]) if s["tps_step"] else "n/a"
            print("| %s | %s | %s | %s | %s | %s | %.0f +/- %.0f | %.0f +/- %.0f |" % (k, t, f(a), f(b), g(a), g(b), a["tps"][0], a["tps"][1], b["tps"][0], b["tps"][1]))
if __name__ == "__main__":
    Q = ["control-q4km", "control-q4km-end", "control-q4km-end2"]
    ng = [k for k in D if k.startswith("ngram") and not k.endswith("-cold") and not k.endswith("-r2")]
    print("<!-- generated by scripts/s2_report.py -->")
    table([k for k in ng if k.startswith("ngram-simple")], Q, "n-gram: ngram-simple (n x m)")
    table([k for k in ng if k.startswith("ngram-map-k-")], Q, "n-gram: ngram-map-k (n x m)")
    table([k for k in ng if k.startswith("ngram-map-k4v")], Q, "n-gram: ngram-map-k4v")
    table([k for k in ng if k.startswith("ngram-mod")], Q, "n-gram: ngram-mod (warm: the server sees the same prompts again in the warm-up pass and every rep; see the cold table)")
    table(["ngram-simple-n3-m16-r2"], Q, "Re-check of the headline n-gram config later in the session (pass 2)")
    table([k for k in D if k.startswith("mtp") and k.endswith("-r2")] + [k for k in D if k.startswith("combo") and k.endswith("-r2")], ["control-mtpfile-end", "control-mtpfile-end2"], "MTP and MTP+n-gram, pass 2 (unsloth MTP GGUF Q4_K_M; plain control = same file, bracketing controls)")
    table(["bart-mtp-nmax1-r2"], Q, "MTP head inside the bartowski Q4_K_M file, pass 2 (plain control = same file)")
    table([k for k in D if (k.startswith("mtp") or k.startswith("combo")) and not k.endswith("-r2") and "runs" in D[k]], ["control-mtpfile"], "MTP and MTP+n-gram, pass 1 (SUPERSEDED: its only start control ran at ~100 tok/s, the end control of the same file at ~142; shown for transparency)")
    table([k for k in D if k.startswith("bart-mtp") and not k.endswith("-r2")], ["control-q4km"], "bartowski-file MTP, pass 1 (SUPERSEDED: ran in the same slow machine state as MTP pass 1; compared with the start control of the same file)")
    cold_table(); repro_table(); best_table(); maxes()
    copy_fraction("ngram-simple-n3-m16", Q); copy_fraction("ngram-mod-match8", Q)
    if EXCL: print("\nReps excluded as machine interruptions (< 60% of the median rep): " + "; ".join(sorted(set(EXCL))))
