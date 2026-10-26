"""E5 feasibility study (CPU, numpy): how often can a cheap reduced head CERTIFY the argmax of the exact bf16 head?
Inputs: logs/e5_hidden_<tag>.npz (h = final hidden per generated position, tok = actual next token), head = tied token_embd of the bf16 GGUF.
Certificates (all rigorous bounds, no statistics):
  quant-b : head rows stored with block-32 symmetric b-bit quantisation (Q8_0-like); logit error |h.e_i| <= |h| |e_i| (Cauchy-Schwarz, |e_i| = true row error norm).
            certified iff s_j - b_j > max_{i != j}(s_i + b_i), j = argmax(s - b)... (j = argmax of lower bound)
  lowrank-r : rank-r projection V_r of the head; score a_i = (h V_r).(w_i V_r); residual |h.R_i| <= |h_perp| |R_i|. Candidates = rows whose upper bound >= best lower bound;
              exact full-precision rows are then evaluated only for the candidates (lossless pruning).
usage: python scripts\e5_analysis.py <tag>"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_read
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1]
d = np.load(os.path.join(ROOT, "logs", "e5_hidden_%s.npz" % tag))
H, TOK = d["h"].astype(np.float32), d["tok"]
W = gguf_read.read_tensor(os.path.join(ROOT, "models", "Qwen_Qwen3.5-0.8B-bf16.gguf"), "token_embd.weight")
N, V = H.shape[0], W.shape[0]
print("positions", N, "vocab", V, "|h| mean %.2f" % np.linalg.norm(H, axis=1).mean(), flush=True)
hn = np.linalg.norm(H, axis=1)

def quant_err(W, bits):
    """block-32 symmetric quantisation along the row (like Q8_0 / Q5_0 / Q4_0): returns dequantised W and row error norms"""
    qmax = 2 ** (bits - 1) - 1
    Wb = W.reshape(W.shape[0], -1, 32)
    sc = np.abs(Wb).max(axis=2, keepdims=True) / qmax
    sc = np.where(sc == 0, 1, sc).astype(np.float16).astype(np.float32)
    Q = np.clip(np.round(Wb / sc), -qmax - 1, qmax) * sc
    E = (Wb - Q).reshape(W.shape[0], -1)
    return Q.reshape(W.shape), np.linalg.norm(E, axis=1)

def stats(name, cert, extra=""):
    print("%-22s certified %6.2f%%  (%d / %d)%s" % (name, 100 * cert.mean(), cert.sum(), cert.size, extra), flush=True)

# exact scores top-2 (float32 matmul), in chunks
top1 = np.zeros(N, np.int64); gap = np.zeros(N, np.float32); z1 = np.zeros(N, np.float32)
CH = 128
t0 = time.time()
for a in range(0, N, CH):
    Z = H[a:a + CH] @ W.T
    p = np.argpartition(-Z, 2, axis=1)[:, :2]
    zz = np.take_along_axis(Z, p, 1); o = np.argsort(-zz, axis=1)
    top1[a:a + CH] = np.take_along_axis(p, o, 1)[:, 0]
    s = np.take_along_axis(zz, o, 1); gap[a:a + CH] = s[:, 0] - s[:, 1]; z1[a:a + CH] = s[:, 0]
print("full head matmul %.1fs; argmax == generated next token: %.2f%%; top1 logit mean %.2f; gap quantiles 1/5/25/50/75%%: %s" % (
    time.time() - t0, 100 * (top1 == TOK).mean(), z1.mean(), np.round(np.quantile(gap, [.01, .05, .25, .5, .75]), 3)), flush=True)
res = {"positions": int(N), "gap_quantiles": [float(x) for x in np.quantile(gap, [.01, .05, .25, .5, .75])], "argmax_eq_generated": float((top1 == TOK).mean())}
for q in (0.05, 0.1, 0.2, 0.5, 1.0): print("  gap < %.2f: %.1f%%" % (q, 100 * (gap < q).mean()))

for bits in (8, 6, 5, 4):
    Q, en = quant_err(W, bits)
    cert = np.zeros(N, bool); agree = np.zeros(N, bool); candn = np.zeros(N, np.int64); inc = np.zeros(N, bool)
    for a in range(0, N, CH):
        S = H[a:a + CH] @ Q.T
        B = hn[a:a + CH, None] * en[None, :]
        lo, up = S - B, S + B
        j = lo.argmax(1); lj = lo[np.arange(len(j)), j]
        cmask = up >= lj[:, None]
        candn[a:a + CH] = cmask.sum(1); inc[a:a + CH] = cmask[np.arange(len(j)), top1[a:a + CH]]
        up[np.arange(len(j)), j] = -np.inf
        cert[a:a + CH] = lj > up.max(1)
        agree[a:a + CH] = (S.argmax(1) == top1[a:a + CH])
    bpw = bits + 16.0 / 32
    stats("quant-%d (%.1f bit/w)" % (bits, bpw), cert, " ; plain argmax of quantised head == exact argmax: %.2f%% ; mean |e|/|w| %.4f ; bytes vs bf16 %.3f" % (
        100 * agree.mean(), (en / np.linalg.norm(W, axis=1)).mean(), bpw / 16))
    assert inc.all(), "bound violated"
    fr = lambda c: 100 * (candn <= c).mean()
    print("   quant-%d candidate pruning: candidates mean %.1f median %.0f p90 %.0f p99 %.0f; <=1 %.1f%%  <=4 %.1f%%  <=16 %.1f%%  <=64 %.1f%%  <=256 %.1f%%  <=1024 %.1f%%" % (bits, candn.mean(), np.median(candn), np.quantile(candn, .9), np.quantile(candn, .99), fr(1), fr(4), fr(16), fr(64), fr(256), fr(1024)), flush=True)
    res["quant%d" % bits] = {"cert": float(cert.mean()), "argmax_agree": float(agree.mean()), "bytes_frac": bpw / 16, "cand_mean": float(candn.mean()), "le16": fr(16) / 100, "le64": fr(64) / 100, "le256": fr(256) / 100, "le1024": fr(1024) / 100}
    del Q

# low-rank candidate pruning
t0 = time.time()
C = (W.T @ W).astype(np.float64)
ev, evec = np.linalg.eigh(C); evec = evec[:, ::-1].astype(np.float32); ev = ev[::-1]
print("eigh %.1fs; energy captured by rank 64/128/256/512: %s" % (time.time() - t0, [round(float(ev[:r].sum() / ev.sum()), 4) for r in (64, 128, 256, 512)]), flush=True)
for r in (128, 256, 384, 512, 768):
    Vr = evec[:, :r]
    WP = W @ Vr                                            # [V, r]
    rho = np.sqrt(np.maximum((W * W).sum(1) - (WP * WP).sum(1), 0))
    cand_n = np.zeros(N, np.int64); ok = np.zeros(N, bool)
    for a in range(0, N, CH):
        h = H[a:a + CH]; hp = h @ Vr
        hperp = np.sqrt(np.maximum((h * h).sum(1) - (hp * hp).sum(1), 0))
        S = hp @ WP.T
        B = hperp[:, None] * rho[None, :]
        lo = (S - B).max(1)
        cand = (S + B) >= lo[:, None]
        cand_n[a:a + CH] = cand.sum(1)
        ok[a:a + CH] = cand[np.arange(len(h)), top1[a:a + CH]]
    assert ok.all(), "bound violated"
    print("lowrank-%-4d candidates mean %.0f median %.0f p90 %.0f ; rows after prune <=64: %.1f%% ; <=1024: %.1f%% ; bytes first pass %.3f of bf16 (+ candidate rows)" % (
        r, cand_n.mean(), np.median(cand_n), np.quantile(cand_n, .9), 100 * (cand_n <= 64).mean(), 100 * (cand_n <= 1024).mean(), r / 1024), flush=True)
    res["lowrank%d" % r] = {"cand_mean": float(cand_n.mean()), "cand_median": float(np.median(cand_n)), "le64": float((cand_n <= 64).mean()), "bytes_frac": r / 1024}
json.dump(res, open(os.path.join(ROOT, "results", "2026-10-27-bf16-E5-analysis-%s.json" % tag), "w"), indent=1)

