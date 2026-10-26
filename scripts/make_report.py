#!/usr/bin/env python3
"""Turn results/2026-10-26-parallel-subtasks.raw.json into markdown tables (stdout)."""
import json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results", "2026-10-26-parallel-subtasks.raw.json")))
cfgs = d["configs"]
def label(c): return "np=%d%s" % (c["np"], " (re-check at end)" if c["drift_check"] else "")
def ms(xs, f="%.1f"):
    xs = list(xs); sd = st.stdev(xs) if len(xs) > 1 else 0
    return (f + " ± " + f + " (" + f + "–" + f + ")") % (st.mean(xs), sd, min(xs), max(xs))
def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"
print("Format: mean ± sd (min–max) over %d timed runs.\n" % d["reps"])
print("### Baseline A: one ~400-word reply, single stream (n_predict 700)\n")
print(table(["server", "tokens", "words", "wall s", "tok/s (tokens/wall)", "decode tok/s (server timings)"],
    [[label(c), "%d" % c["baseline"][0]["tokens"], c["baseline"][0]["words"], ms((r["wall"] for r in c["baseline"]), "%.2f"),
      ms(r["tps_wall"] for r in c["baseline"]), ms(r["pps"] for r in c["baseline"])] for c in cfgs]))
base1 = [c for c in cfgs if c["np"] == 1 and not c["drift_check"]][0]
for key, n in (("exp1_8", 8), ("exp1_16", 16)):
    print("### Experiment 1: %d independent sub-tasks (n_predict 100 each); np=1 is sequential, np>1 all %d requests submitted concurrently\n" % (n, n))
    b = st.mean(r["wall"] for r in base1[key])
    print(table(["server", "total tokens", "job wall s", "aggregate tok/s", "per-stream decode tok/s", "mean / max request latency s", "speed-up vs np=1"],
        [[label(c), "%d" % st.mean(r["tokens"] for r in c[key]), ms((r["wall"] for r in c[key]), "%.2f"), ms(r["agg_tps"] for r in c[key]),
          ms(r["per_stream_tps"] for r in c[key]), "%.2f / %.2f" % (st.mean(r["mean_latency"] for r in c[key]), st.mean(r["max_latency"] for r in c[key])),
          "%.2fx" % (b / st.mean(r["wall"] for r in c[key]))] for c in cfgs]))
bw = st.mean(r["wall"] for r in base1["baseline"])
bw_per100 = 100 * bw / base1["baseline"][0]["words"]
print("Baseline A (np=1) = %.2f s per 100 words.\n" % bw_per100)
for v in ("short", "long"):
    key = "exp2_" + v
    print("### Experiment 2 (%s sections): skeleton-of-thought, 6-point outline then parallel expansion (np=1 expands sequentially)\n" % v)
    print(table(["server", "points", "outline s", "expand s", "total wall s", "reply tokens", "reply words", "effective tok/s (reply tokens / total wall)", "incl. outline tokens", "per-stream decode tok/s", "s per 100 reply words", "words-normalised speed-up vs Baseline A (np=1)"],
        [[label(c), "%.1f" % st.mean(r["n_points"] for r in c[key]), "%.2f" % st.mean(r["t_outline"] for r in c[key]), "%.2f" % st.mean(r["t_expand"] for r in c[key]),
          ms((r["t_total"] for r in c[key]), "%.2f"), "%d" % st.mean(r["reply_tokens"] for r in c[key]), "%d" % st.mean(r["reply_words"] for r in c[key]),
          ms(r["eff_tps"] for r in c[key]), "%.1f" % st.mean(r["agg_tps_all"] for r in c[key]), "%.1f" % st.mean(r["per_stream_tps"] for r in c[key]),
          "%.2f" % (100 * st.mean(r["t_total"] for r in c[key]) / st.mean(r["reply_words"] for r in c[key])),
          "%.2fx" % (bw_per100 / (100 * st.mean(r["t_total"] for r in c[key]) / st.mean(r["reply_words"] for r in c[key])))] for c in cfgs]))
print("### GPU during each server's whole run (sampled every ~0.4 s; includes warm-up)\n")
print(table(["server", "VRAM after warm-up MiB (whole GPU)", "peak VRAM MiB", "max temp C", "mean power W (busy)", "mean SM clock MHz (busy)", "GPU idle before start (mem MiB / util % / temp C)"],
    [[label(c), "%d" % c["vram_after_warmup_mib"], "%d" % c["gpu"]["peak_mem_mib"], "%d" % c["gpu"]["max_temp_c"], c["gpu"]["mean_power_w_busy"], c["gpu"]["mean_sm_mhz_busy"],
      "%d / %d / %d" % (c["idle_before"][0], c["idle_before"][1], c["idle_before"][2])] for c in cfgs]))
print("Server command (np=N): `" + cfgs[0]["cmd"].replace(os.path.join(ROOT, ""), "") + "`  (-c = 2048 x np)\n")

sp = os.path.join(ROOT, "results", "2026-10-26-parallel-subtasks-saturation.raw.json")
if os.path.exists(sp):
    sd_ = json.load(open(sp))
    print("### Saturation check: %d independent sub-tasks (cycled prompts, n_predict 100) submitted at once, %d timed runs\n" % (sd_["tasks"], sd_["reps"]))
    print(table(["server", "total tokens", "job wall s", "aggregate tok/s", "per-stream decode tok/s", "peak VRAM MiB", "mean power W (busy)", "max temp C"],
        [["np=%d" % c["np"], "%d" % st.mean(r["tokens"] for r in c["runs"]), ms((r["wall"] for r in c["runs"]), "%.2f"), ms(r["agg_tps"] for r in c["runs"]),
          ms(r["per_stream_tps"] for r in c["runs"]), "%d" % c["gpu"]["peak_mem_mib"], c["gpu"]["mean_power_w_busy"], "%d" % c["gpu"]["max_temp_c"]] for c in sd_["configs"]]))
