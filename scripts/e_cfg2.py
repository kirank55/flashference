"""E2/E3/E4 configs (loaded with e_bench.py --module e_cfg2). New build: llama-cpp-e (B4 trim patch + E2 adaptive n + E4 dynamic vocab, env driven)."""
def configs(e):
    C = {}
    LE = "llama-cpp-e"; BA = e.BA; mt = e.mt; R = e.RANK_ENV
    def S(name, desc, set_, args=None, env=None, lc=LE, req=None):
        C[name] = dict(kind="server", desc=desc, set=set_, model=e.bb.BF16, args=BA if args is None else args, env=env, lc=lc, req=req)
    def X(K, **kw): d = R(K); d.update({k: str(v) for k, v in kw.items()}); return d
    # ---- E2a: adaptive draft length / confidence threshold on the trimmed 8k head
    for st in ("free",):
        S("e2-%s-base" % st, "%s: trim8k n-max 2 (control on the E build)" % st, st, BA + mt(2), X(8192))
        S("e2-%s-n3" % st, "%s: trim8k n-max 3 fixed" % st, st, BA + mt(3), X(8192))
        S("e2-%s-n4" % st, "%s: trim8k n-max 4 fixed" % st, st, BA + mt(4), X(8192))
        for p in (0.3, 0.5, 0.7):
            S("e2-%s-n3-p%s" % (st, p), "%s: trim8k n-max 3, --spec-draft-p-min %s" % (st, p), st, BA + mt(3, ["--spec-draft-p-min", str(p)]), X(8192))
        S("e2-%s-n4-p0.5" % st, "%s: trim8k n-max 4, p-min 0.5" % st, st, BA + mt(4, ["--spec-draft-p-min", "0.5"]), X(8192))
        S("e2-%s-adapt4" % st, "%s: trim8k history-adaptive n in [1,4]" % st, st, BA + mt(4), X(8192, LLAMA_MTP_ADAPT=1))
        S("e2-%s-adapt3" % st, "%s: trim8k history-adaptive n in [1,3]" % st, st, BA + mt(3), X(8192, LLAMA_MTP_ADAPT=1))
    # ---- E4: context-adaptive draft vocabulary (fixed Kf rows + context rows, total 8192)
    for st in ("free", "lang", "code"):
        S("e4-%s-fix4k-dyn4k" % st, "%s: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2" % st, st, BA + mt(2), X(8192, LLAMA_MTP_TRIM_FIXED=4096, LLAMA_MTP_DYN=16))
        S("e4-%s-fix2k-dyn6k" % st, "%s: 2048 fixed + 6144 context/script rows, n-max 2" % st, st, BA + mt(2), X(8192, LLAMA_MTP_TRIM_FIXED=2048, LLAMA_MTP_DYN=16))
    # ---- E3: n-gram / prompt lookup drafter in front of MTP
    NG = lambda n, m: ["--spec-ngram-simple-size-n", str(n), "--spec-ngram-simple-size-m", str(m)]
    for st in ("free", "rep"):
        S("e3-%s-mtp" % st, "%s: trim8k MTP n-max 2 (control)" % st, st, BA + mt(2), X(8192))
        S("e3-%s-ngs-mtp-n4m8" % st, "%s: ngram-simple(N4,M8) then MTP trim8k, n-max 8" % st, st,
          BA + ["--spec-type", "ngram-simple,draft-mtp", "--spec-draft-n-max", "8", "-bs"] + NG(4, 8), X(8192, LLAMA_MTP_ADAPT=1, LLAMA_MTP_ADAPT_INIT=2))
        S("e3-%s-ngs-only" % st, "%s: ngram-simple(N4,M8) only, n-max 8" % st, st, BA + ["--spec-type", "ngram-simple", "--spec-draft-n-max", "8", "-bs"] + NG(4, 8))
        S("e3-%s-ngmod-mtp" % st, "%s: ngram-mod then MTP trim8k" % st, st,
          BA + ["--spec-type", "ngram-mod,draft-mtp", "--spec-draft-n-max", "8", "-bs"], X(8192, LLAMA_MTP_ADAPT=1, LLAMA_MTP_ADAPT_INIT=2))
    # ---- E5: speed ceiling of a Q8_0 output head (token_embd q8_0 => NOT lossless; measures what an exact Q8-first-pass head could save at most)
    Q8 = "Qwen_Qwen3.5-0.8B-bf16-q8head.gguf"
    C["e5-plain-q8head"] = dict(kind="server", desc="free: plain, bf16 body + Q8_0 token_embd/output head (speed ceiling, lossy)", set="free", model=Q8, args=BA, env=None, lc=None, req=None)
    C["e5-trim8k-q8head"] = dict(kind="server", desc="free: trim8k MTP n2 -bs, bf16 body + Q8_0 head (speed ceiling, lossy)", set="free", model=Q8, args=BA + mt(2), env=X(8192), lc=LE, req=None)
    C["e3-rep-plain"] = dict(kind="server", desc="rep: plain bf16 (release build)", set="rep", model=e.bb.BF16, args=BA, env=None, lc=None, req=None)
    return C
