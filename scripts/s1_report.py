#!/usr/bin/env python3
"""S1 tables from results/2026-10-26-s1-quants-0.8b.raw.json -> markdown on stdout."""
import json, os, math, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "results", "2026-10-26-s1-quants-0.8b.raw.json")))
p1, p2, Q = D["pass1"], D["pass2"], D["quality"]
order = sorted(p1, key=lambda k: p1[k]["size"])
def pooled(k, t):
    a, b = p1[k][t], p2[k][t]
    return (a["mean"] + b["mean"]) / 2, max(a["sd"], b["sd"])
q8 = pooled("Q8_0", "tg")[0]; bf = pooled("bf16", "tg")[0]; q8p = pooled("Q8_0", "pp")[0]
print("| Quant | file MiB | bits/weight | decode tok/s pass1 | decode tok/s pass2 (reverse order) | drift | decode mean (SD) | vs Q8_0 | vs bf16 | prefill tok/s (pp512) | PPL (20x512 wikitext-2, +/- se) | mean KLD vs bf16 (+/- se) | same-top-token % | RMS dp % |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(order, key=lambda k: -p1[k]["size"]):
    s = p1[k]["size"]; n = p1[k]["params"]; m, sd = pooled(k, "tg"); pp = pooled(k, "pp")[0]
    q = Q[k]
    print("| %s | %.0f | %.2f | %.1f +/- %.1f | %.1f +/- %.1f | %+.1f%% | %.1f (%.1f) | %.2fx | %.2fx | %.0f | %.2f +/- %.2f | %.4f +/- %.4f | %.1f +/- %.1f | %.2f |" % (
        k, s / 1048576, s * 8 / n, p1[k]["tg"]["mean"], p1[k]["tg"]["sd"], p2[k]["tg"]["mean"], p2[k]["tg"]["sd"], 100 * (p2[k]["tg"]["mean"] / p1[k]["tg"]["mean"] - 1),
        m, sd, m / q8, m / bf, pp, q["ppl"][0], q["ppl"][1], q["kld"][0], q["kld"][1], q["same_top"][0], q["same_top"][1], q["rms_dp"][0] if q.get("rms_dp") else 0))
# affine fit t_token(ms) = a + b * MiB
def fit(keys, label):
    x = [p1[k]["size"] / 1e9 for k in keys]  # GB
    y = [1000.0 / pooled(k, "tg")[0] for k in keys]
    n = len(x); mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x); b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx; a0 = my - b * mx
    res = [c - (a0 + b * a) for a, c in zip(x, y)]; sse = sum(r * r for r in res); sst = sum((c - my) ** 2 for c in y)
    r2 = 1 - sse / sst; s2 = sse / (n - 2); seb = math.sqrt(s2 / sxx); sea = math.sqrt(s2 * (1 / n + mx * mx / sxx))
    tcrit = {12: 2.179, 11: 2.201, 10: 2.228, 9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776, 3: 3.182}.get(n - 2, 2.2)
    print("| %s | %d | %.2f +/- %.2f | %.2f +/- %.2f (1/b = %.0f GB/s) | %.3f |" % (label, n, a0, tcrit * sea, b, tcrit * seb, 1000 / b, r2))
print("\n| fit set | n points | a (ms/token, intercept, 95% CI) | b (ms per GB of file, 95% CI) | R^2 |")
print("|---|---|---|---|---|")
fit(order, "all 14 files")
fit([k for k in order if k not in ("bf16",)], "quantised only (no bf16)")
fit([k for k in order if k in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K")], "K-quants only (+Q8_0)")
fit([k for k in order if k in ("Q4_0", "Q4_1", "Q8_0")], "legacy quants only (Q4_0, Q4_1, Q8_0)")
print("\nbuild:", p1["Q8_0"].get("build"), " params(llama-bench):", p1["Q8_0"]["params"])

