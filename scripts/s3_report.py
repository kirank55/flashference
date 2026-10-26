#!/usr/bin/env python3
"""Tables for S3 from results/2026-10-26-s3-sot-0.8b.raw.json -> prints markdown (redirect into results/2026-10-26-s3-sot-0.8b.md)."""
import json, os, math, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "results", "2026-10-26-s3-sot-0.8b.raw.json")))["servers"]

if "np4sw" in D and "np4" in D:  # sweeps were run as a separate server session (np4sw) with fewer reps; merge for the tables
    D["np4"]["cfgs"] = D["np4"]["cfgs"] + D["np4sw"]["cfgs"]

def ms(xs):
    xs = list(xs); return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

def entry_stats(e):
    sw = [s["reply_words"] / s["wall"] for s in e["sot"]]; qw = [s["words"] / s["wall"] for s in e["seq"]]
    sp = ms(sw)[0] / ms(qw)[0]
    rel = math.sqrt((ms(sw)[1] / ms(sw)[0]) ** 2 + (ms(qw)[1] / ms(qw)[0]) ** 2)
    ideal = st.mean(s["ideal_np"] for s in e["sot"])
    return {"wall": ms(s["wall"] for s in e["sot"])[0], "wall_sd": ms(s["wall"] for s in e["sot"])[1], "words": st.mean(s["reply_words"] for s in e["sot"]), "tokens": st.mean(s["reply_tokens"] for s in e["sot"]),
            "seq_wall": ms(s["wall"] for s in e["seq"])[0], "seq_words": st.mean(s["words"] for s in e["seq"]), "seq_tokens": st.mean(s["tokens"] for s in e["seq"]),
            "speedup": sp, "sp_sd": sp * rel, "ideal": ideal, "ideal_inf": st.mean(s["ideal_inf"] for s in e["sot"]),
            "eff_tps": st.mean(s["eff_tps"] for s in e["sot"]), "ov": st.mean(s["overlap_max"] for s in e["sot"]), "rolls": st.mean(s["rolls"] for s in e["sot"]),
            "npts": st.mean(s["n_points"] for s in e["sot"]), "t_outline": st.mean(s["t_outline"] for s in e["sot"]), "agg": st.mean(s["agg_tps_expand"] for s in e["sot"]),
            "per_stream": st.mean(s["per_stream_tps"] for s in e["sot"]), "seq_pps": st.mean(s["pps"] for s in e["seq"]), "trunc": st.mean(s["n_trunc"] for s in e["sot"]),
            "prompt_n": st.mean(s["prompt_n"] for s in e["sot"]), "cache_n": st.mean(s["cache_n"] for s in e["sot"]),
            "qs": e["quality_sot"], "qq": e["quality_seq"], "max_eff": max(s["eff_tps"] for s in e["sot"])}

def sel(srv, **kw):
    out = []
    for e in D.get(srv, {}).get("cfgs", []):
        if all(e[k] == v for k, v in kw.items()): out.append(e)
    return out

def agg(entries, key):
    return st.mean(entry_stats(e)[key] for e in entries)

def qagg(entries, which, key):
    return st.mean(entry_stats(e)[which][key] for e in entries)

def fmt(x, p=2): return ("%." + str(p) + "f") % x

V2 = ["naive", "distinct_ctx_ic"]
VN = {"naive": "(a) naive outline", "distinct_ctx_ic": "(b) distinct outline + context + intro + conclusion", "distinct_ctx_ic_prefix": "(b) + cache_prompt prefix reuse"}

def main_table():
    print("\n### Main comparison (np=4, N=6 requested, ~100-token sections, 6 topics T1-T6; 3 reps each, one warm-up discarded; mean over topics)\n")
    print("| variant | points | total wall s | words | tokens | ideal (critical path, np=4) | realised speed-up per word vs sequential (mean over topics; range) | realised/ideal | tok/wall | outline overlap (max pairwise Jaccard) | outline re-rolls | sections truncated | distinct-2gram | distinct-4gram | repeated-sentence rate | keyword coverage | auto rubric 0-2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in V2:
        es = sel("np4", phase="main", variant=v)
        if not es: continue
        S = [entry_stats(e) for e in es]; sp = [s["speedup"] for s in S]
        print("| %s | %s | %s | %s | %s | %s | %sx (%s-%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (VN[v], fmt(agg(es, "npts"), 1), fmt(agg(es, "wall")), fmt(agg(es, "words"), 0), fmt(agg(es, "tokens"), 0), fmt(agg(es, "ideal")), fmt(st.mean(sp)), fmt(min(sp)), fmt(max(sp)),
              fmt(st.mean(s["speedup"] / s["ideal"] for s in S)), fmt(agg(es, "eff_tps"), 0), fmt(agg(es, "ov")), fmt(agg(es, "rolls"), 1), fmt(agg(es, "trunc"), 1), fmt(qagg(es, "qs", "distinct2")), fmt(qagg(es, "qs", "distinct4")), fmt(qagg(es, "qs", "rep_sent_rate"), 3), fmt(qagg(es, "qs", "kw_coverage")), fmt(qagg(es, "qs", "rubric_auto"))))
    for v in V2:
        es = sel("np4", phase="main", variant=v)
        if not es: continue
        print("| sequential control matched to %s (plain decode) | - | %s | %s | %s | 1.00 | 1.00x | - | %s | - | - | - | %s | %s | %s | %s | %s |" % (v, fmt(st.mean(entry_stats(e)["seq_wall"] for e in es)), fmt(agg(es, "seq_words"), 0), fmt(agg(es, "seq_tokens"), 0),
              fmt(st.mean(entry_stats(e)["seq_tokens"] / entry_stats(e)["seq_wall"] for e in es), 0), fmt(qagg(es, "qq", "distinct2")), fmt(qagg(es, "qq", "distinct4")), fmt(qagg(es, "qq", "rep_sent_rate"), 3), fmt(qagg(es, "qq", "kw_coverage")), fmt(qagg(es, "qq", "rubric_auto"))))
    print("\nPer topic (np=4, N=6, L=100): realised speed-up per word (mean of 3 reps) / distinct-4gram / repeated-sentence rate; the last columns give the matched sequential reply (words, distinct-4gram / rep rate)\n")
    print("| topic | naive | (b) distinct+ctx+intro+concl | sequential matched to naive | sequential matched to (b) |")
    print("|---|---|---|---|---|")
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        cells = []
        for v in V2:
            es = sel("np4", phase="main", variant=v, topic=t)
            if es:
                s = entry_stats(es[0]); cells.append(("%sx / %s / %s (%dw, ideal %s)" % (fmt(s["speedup"]), fmt(s["qs"]["distinct4"]), fmt(s["qs"]["rep_sent_rate"], 2), s["words"], fmt(s["ideal"]))))
            else: cells.append("-")
        for v in V2:
            es = sel("np4", phase="main", variant=v, topic=t)
            if es:
                s = entry_stats(es[0]); q = s["qq"]; cells.append("%dw, %s / %s" % (s["seq_words"], fmt(q["distinct4"]), fmt(q["rep_sent_rate"], 2)))
            else: cells.append("-")
        print("| %s | %s |" % (t, " | ".join(cells)))

def grid_table():
    t3 = ["T1", "T2", "T3"]
    print("\n### Reduced grid: sections N x section length L (variant (b), np=4; topics T1-T3; 3 reps; N=6/L=100 from the main run restricted to T1-T3)\n")
    print("| N requested | L (token cap per section) | points | words | total wall s | ideal (np=4) | ideal (infinite slots) | realised speed-up per word | realised/ideal | eff tok/s (reply tokens/wall) | sections truncated | distinct-4gram | rep. sentence rate | auto rubric |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for L in (50, 100, 200):
        for n in (4, 6, 8):
            es = [e for e in D.get("np4", {}).get("cfgs", []) if e["variant"] == "distinct_ctx_ic" and e["N"] == n and e["L"] == L and e["topic"] in t3 and e["phase"] in ("main", "sweeps")]
            if not es: print("| %d | %d | skipped | | | | | | | | | | | |" % (n, L)); continue
            S = [entry_stats(e) for e in es]
            print("| %d | %d | %s | %s | %s | %s | %s | %sx (+/- %s) | %s | %s | %s | %s | %s | %s |" % (n, L, fmt(agg(es, "npts"), 1), fmt(agg(es, "words"), 0), fmt(agg(es, "wall")), fmt(agg(es, "ideal")), fmt(agg(es, "ideal_inf")), fmt(st.mean(s["speedup"] for s in S)), fmt(st.mean(s["sp_sd"] for s in S)),
                  fmt(st.mean(s["speedup"] / s["ideal"] for s in S)), fmt(agg(es, "eff_tps"), 0), fmt(agg(es, "trunc"), 1), fmt(qagg(es, "qs", "distinct4")), fmt(qagg(es, "qs", "rep_sent_rate"), 3), fmt(qagg(es, "qs", "rubric_auto"))))

def np_table():
    t3 = ["T1", "T2", "T3"]
    print("\n### -np sweep (variant (b), L=100; topics T1-T3; fresh server per np; ratios vs the sequential plain reply of the same words on the same server)\n")
    print("| N requested | np | total wall s | words | ideal (critical path with np slots) | realised speed-up per word | realised/ideal | aggregate tok/s during expansion / sequential tok/s | per-stream tok/s / sequential tok/s | eff tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in (6, 8):
        for np_ in (1, 4, 8):
            if np_ == 4: es = [e for e in D.get("np4", {}).get("cfgs", []) if e["variant"] == "distinct_ctx_ic" and e["N"] == n and e["L"] == 100 and e["topic"] in t3]
            else: es = [e for e in D.get("np%d" % np_, {}).get("cfgs", []) if e["variant"] == "distinct_ctx_ic" and e["N"] == n and e["L"] == 100 and e["topic"] in t3]
            if not es: print("| %d | %d | skipped | | | | | | | |" % (n, np_)); continue
            S = [entry_stats(e) for e in es]
            print("| %d | %d | %s | %s | %s | %sx | %s | %sx | %sx | %s |" % (n, np_, fmt(agg(es, "wall")), fmt(agg(es, "words"), 0), fmt(agg(es, "ideal")), fmt(st.mean(s["speedup"] for s in S)), fmt(st.mean(s["speedup"] / s["ideal"] for s in S)),
                  fmt(st.mean(s["agg"] / s["seq_pps"] for s in S)), fmt(st.mean(s["per_stream"] / s["seq_pps"] for s in S)), fmt(agg(es, "eff_tps"), 0)))

def prefix_table():
    srv = "np4pf"
    if srv not in D: return
    print("\n### Prefix reuse (cache_prompt=true, every slot primed with the shared instruction+outline prefix; same server session, alternating-by-topic runs; np=4)\n")
    print("| variant | topics | total wall s | words | speed-up per word vs sequential | prompt tokens evaluated per reply | cached prompt tokens per reply | prime time s | distinct-4gram | rep. sentence rate |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v in ("distinct_ctx_ic", "distinct_ctx_ic_prefix"):
        es = [e for e in D[srv]["cfgs"] if e["variant"] == v]
        if not es: continue
        S = [entry_stats(e) for e in es]
        print("| %s | %d | %s | %s | %sx | %s | %s | %s | %s | %s |" % (VN[v], len(es), fmt(agg(es, "wall")), fmt(agg(es, "words"), 0), fmt(st.mean(s["speedup"] for s in S)), fmt(agg(es, "prompt_n"), 0), fmt(agg(es, "cache_n"), 0),
              fmt(st.mean(x["t_prime"] for e in es for x in e["sot"])), fmt(qagg(es, "qs", "distinct4")), fmt(qagg(es, "qs", "rep_sent_rate"), 3)))

def controls():
    print("\n### Plain-decode controls (256-token electric-cars reply; decode tok/s, mean +/- SD of 5) and drift within each server session\n")
    print("| server session | start | end | drift |")
    print("|---|---|---|---|")
    for k, r in D.items():
        a = r.get("control_start"); b = r.get("control_end")
        if not a: continue
        sa = ms(x["pps"] for x in a); sb = ms(x["pps"] for x in b) if b else None
        print("| %s | %.1f +/- %.1f | %s | %s |" % (k, sa[0], sa[1], ("%.1f +/- %.1f" % sb) if sb else "n/a", ("%.1f%%" % (100 * abs(sa[0] - sb[0]) / sa[0])) if sb else "n/a"))

def maxes():
    best = None
    for k, r in D.items():
        for e in r.get("cfgs", []):
            for s in e["sot"]:
                if best is None or s["eff_tps"] > best[0]: best = (s["eff_tps"], k, e["variant"], e["N"], e["L"], e["topic"], s["reply_words"], s["wall"])
    if best: print("\nHighest single SoT reply (effective tok/s = reply tokens / wall incl. outline): %.0f tok/s (%s, %s, N=%d, L=%d, %s, %d words, %.2f s)" % best)

if __name__ == "__main__":
    print("<!-- generated by scripts/s3_report.py -->")
    main_table(); grid_table(); np_table(); prefix_table(); controls(); maxes()
