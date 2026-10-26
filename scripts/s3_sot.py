#!/usr/bin/env python3
r"""S3: improved skeleton-of-thought (SoT) for ONE reply, Qwen3.5-0.8B Q4_K_M, llama-server (stdlib only).
Variants: naive | distinct | distinct_ctx | distinct_ctx_ic | distinct_ctx_ic_prefix (see VARIANTS).
Greedy (temperature 0, seed 1) except outline re-rolls (temperature 0.8, seeds 11/12). Fresh server per -np.
Usage: python scripts\s3_sot.py [--phase main sweeps np end] [--reps 5]
Raw results: results/2026-10-26-s3-sot-0.8b.raw.json ; sample texts: results/2026-10-26-s3-samples.json
"""
import argparse, json, os, re, subprocess, sys, threading, time, statistics as st, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "llama-cpp", "llama-server.exe")
PORT = 8092
BASE = "http://127.0.0.1:%d" % PORT
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OUT = os.path.join(ROOT, "results", "2026-10-26-s3-sot-0.8b.raw.json")
SAMPLES = os.path.join(ROOT, "results", "2026-10-26-s3-samples.json")
MODEL = os.path.join(ROOT, "models", "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf")

TOPICS = [
 ("T1", "how electric cars work", ["battery", "motor", "inverter", "regenerat", "charg", "range", "efficien", "drivetrain"]),
 ("T2", "how vaccines work", ["immune", "antigen", "antibod", "memory", "pathogen", "dose", "herd", "t cell"]),
 ("T3", "how a bill becomes a law", ["committee", "vote", "senate", "house", "president", "veto", "sign", "introduc"]),
 ("T4", "how photosynthesis works", ["chlorophyll", "sunlight", "carbon dioxide", "glucose", "oxygen", "chloroplast", "water", "calvin"]),
 ("T5", "how internet packets are routed", ["packet", "router", "ip address", "header", "hop", "protocol", "tcp", "table"]),
 ("T6", "how binary search works in code", ["sorted", "middle", "half", "log", "index", "low", "high", "compare"]),
]
VARIANTS = ["naive", "distinct_ctx_ic"]  # (a) naive outline, (b) distinct outline + context + intro + conclusion; "..._prefix" = (b) with cache_prompt prefix reuse
LEN_PHRASE = {50: ("2 sentences (about 30 words)", 70), 100: ("3 to 4 sentences (about 60 words)", 140),
              200: ("two short paragraphs (about 120 words)", 280), 400: ("three paragraphs (about 250 words)", 560)}
STOP = set("a an the of to and or in on for with by is are was were be as at it its this that these those from into than then can how what which who their they them his her use uses used using also".split())

def chatml(user): return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def http(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r: return json.loads(r.read())

def trim_sentence(text):
    m = list(re.finditer(r"[.!?][\"')\]]*(\s|$)", text))
    return text[:m[-1].end()].strip() if m else text.strip()

def complete(user, n_predict, cache=False, temp=0.0, seed=1):
    t0 = time.perf_counter()
    r = http("/completion", {"prompt": chatml(user), "n_predict": n_predict, "temperature": temp, "seed": seed, "cache_prompt": cache, "stream": False})
    wall = time.perf_counter() - t0
    tm = r["timings"]
    text = r["content"].strip()
    trunc = r.get("stop_type") == "limit"
    return {"text": trim_sentence(text) if trunc else text, "truncated": trunc, "tokens": tm["predicted_n"], "pps": tm["predicted_per_second"],
            "prompt_n": tm["prompt_n"], "cache_n": tm.get("cache_n", 0), "wall": wall}

def run_jobs(jobs, parallel, cache=False):
    res = [None] * len(jobs)
    t0 = time.perf_counter()
    if parallel:
        def w(i): res[i] = complete(jobs[i][0], jobs[i][1], cache=cache)
        ths = [threading.Thread(target=w, args=(i,)) for i in range(len(jobs))]
        [t.start() for t in ths]; [t.join() for t in ths]
    else:
        for i, j in enumerate(jobs): res[i] = complete(j[0], j[1], cache=cache)
    return res, time.perf_counter() - t0

# ---------- text metrics ----------
def words(t): return re.findall(r"[A-Za-z0-9']+", t.lower())
def cwords(t): return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words(t) if w not in STOP}
def jacc(a, b): return len(a & b) / len(a | b) if a and b and (a | b) else 0.0

def quality(text, kws):
    w = words(text)
    def ngr(n): return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    d4 = ngr(4); d2 = ngr(2)
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(words(s)) >= 4]
    seen, rep = [], 0
    for s in sents:
        cw = cwords(s)
        if any(jacc(cw, p) >= 0.8 for p in seen): rep += 1
        seen.append(cw)
    low = text.lower()
    cov = sum(1 for k in kws if k in low) / len(kws)
    return {"words": len(w), "distinct4": len(set(d4)) / len(d4) if d4 else 1.0, "distinct2": len(set(d2)) / len(d2) if d2 else 1.0,
            "rep_sent_rate": rep / len(sents) if sents else 0.0, "n_sent": len(sents), "kw_coverage": cov}

def rubric_auto(q):
    cov = 0 if q["kw_coverage"] < 0.4 else (1 if q["kw_coverage"] < 0.7 else 2)
    d4 = 2 if q["distinct4"] >= 0.95 else (1 if q["distinct4"] >= 0.85 else 0)
    rs = 2 if q["rep_sent_rate"] <= 0.02 else (1 if q["rep_sent_rate"] <= 0.10 else 0)
    return round((cov + d4 + rs) / 3, 3)

def outline_overlap(titles):
    sets = [cwords(t) for t in titles]
    ps = [jacc(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return (max(ps), st.mean(ps)) if ps else (0.0, 0.0)

def parse_points(text, n):
    pts = []
    for l in text.splitlines():
        m = re.match(r"^\s*\d+[.)]\s*(.+?)\s*$", l)
        if m: pts.append(re.sub(r"[*_`#]", "", m.group(1)).strip().rstrip(".:"))
    return pts[:n]

# ---------- prompts ----------
def outline_prompt(variant, topic, n):
    if variant == "naive":
        return ("List exactly %d key points for an explanation of %s. Output only a numbered list 1-%d, each point a short phrase of at most 8 words. No other text." % (n, topic, n), 20 * n + 30)
    ex = "1. Heat absorption by refrigerant\n2. Compressor raises pressure\n3. Condenser releases heat\n4. Expansion valve cools"
    return ("Plan an explanation of %s. List exactly %d sections. Rules: every section must cover a DIFFERENT aspect, with no overlap between sections; "
            "give each section a title of 2 to 5 words; output only a numbered list 1-%d containing the titles, no sentences, no other text.\n"
            "Example for 'how a refrigerator works' with 4 sections:\n%s" % (topic, n, n, ex), 14 * n + 30)

def section_prompt(variant, topic, outline, titles, k, L, prefix_layout=False):
    n = len(titles); lp = LEN_PHRASE[L][0]
    if variant == "naive":
        return ("You are writing one section of a short explanation of %s. The full outline is:\n%s\n\nWrite one detailed paragraph of %s covering only point %d (\"%s\"). Do not write an introduction or conclusion and do not cover the other points." % (topic, outline, lp, k + 1, titles[k]))
    prev = titles[k - 1] if k > 0 else "(none, this is the first section)"
    nxt = titles[k + 1] if k + 1 < n else "(none, this is the last section)"
    pre = "You are writing a structured explanation of %s. The full outline is:\n%s\n\n" % (topic, outline)
    if variant == "distinct":
        return pre + "Write one section of %s covering only point %d (\"%s\"). Do not write an introduction or conclusion and do not cover the other points." % (lp, k + 1, titles[k])
    req = ("Write section %d of %d (\"%s\") only: %s. Begin directly with the substance of this point: no introduction, no restating the topic, no conclusion. "
           "Do not cover other sections' points. The previous section is \"%s\"; the next section is \"%s\"." % (k + 1, n, titles[k], lp, prev, nxt))
    return pre + req

def intro_prompt(topic, outline):
    return ("Write a short introduction of exactly 2 sentences (about 35 words) for an explanation of %s that will cover these sections:\n%s\nDo not explain the sections themselves." % (topic, outline), 80)
def concl_prompt(topic, outline):
    return ("Write a short conclusion of exactly 2 sentences (about 35 words) for an explanation of %s that covered these sections:\n%s\nDo not introduce new facts." % (topic, outline), 80)

def lpt_makespan(tokens, np_):
    loads = [0] * max(1, np_)
    for t in sorted(tokens, reverse=True):
        i = loads.index(min(loads)); loads[i] += t
    return max(loads)

# ---------- one SoT reply ----------
def sot(topic, kws, n, L, variant, np_):
    par = np_ > 1
    t0 = time.perf_counter()
    rolls, best = 0, None
    op, ont = outline_prompt(variant, topic, n)
    outline_tokens = 0
    attempts = 1 if variant == "naive" else 3
    for a in range(attempts):
        o = complete(op, ont, temp=0.0 if a == 0 else 0.8, seed=1 if a == 0 else 10 + a)
        outline_tokens += o["tokens"]; rolls += 1
        pts = parse_points(o["text"], n)
        ov = outline_overlap(pts) if len(pts) > 1 else (1.0, 1.0)
        if best is None or ov[0] < best[1][0] or len(pts) > len(best[0]) and ov[0] <= best[1][0] + 0.01: best = (pts, ov)
        if variant != "naive" and len(pts) >= max(2, n - 1) and ov[0] <= 0.5: break
    pts, ov = best
    if variant != "naive":  # dedup: drop later points that overlap > 0.5 with an earlier kept one
        kept = []
        for p in pts:
            if all(jacc(cwords(p), cwords(q)) <= 0.5 for q in kept): kept.append(p)
        pts = kept
    t_outline = time.perf_counter() - t0
    if len(pts) < 2: pts = (pts + ["overview", "key details"])[:2]
    outline = "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(pts))
    cache = variant.endswith("prefix")
    jobs = [(section_prompt(variant, topic, outline, pts, k, L), int(LEN_PHRASE[L][1])) for k in range(len(pts))]
    kinds = ["sec"] * len(jobs)
    if "ic" in variant.split("_"):
        jobs = [(intro_prompt(topic, outline)[0], 80)] + jobs + [(concl_prompt(topic, outline)[0], 80)]
        kinds = ["intro"] + kinds + ["concl"]
    t_prime = 0.0
    if cache:  # prime every slot with the shared prefix (cost included in wall time)
        prefix = "You are writing a structured explanation of %s. The full outline is:\n%s\n\n" % (topic, outline)
        tp = time.perf_counter()
        run_jobs([(prefix, 1)] * np_, par, cache=True)
        t_prime = time.perf_counter() - tp
    res, t_expand = run_jobs(jobs, par, cache=cache)
    wall = time.perf_counter() - t0
    reply = "\n\n".join(r["text"] for r in res)
    toks = [r["tokens"] for r in res]
    reply_tokens = sum(toks)
    mk = lpt_makespan(toks, np_)
    out = {"variant": variant, "n_req": n, "n_points": len(pts), "L": L, "np": np_, "wall": wall, "t_outline": t_outline, "t_expand": t_expand, "t_prime": t_prime,
           "rolls": rolls, "outline_tokens": outline_tokens, "job_tokens": toks, "reply_tokens": reply_tokens, "reply_words": len(words(reply)),
           "ideal_inf": reply_tokens / (outline_tokens + max(toks)), "ideal_np": reply_tokens / (outline_tokens + mk),
           "overlap_max": ov[0], "overlap_mean": ov[1], "per_stream_tps": st.mean(r["pps"] for r in res), "agg_tps_expand": reply_tokens / t_expand,
           "eff_tps": reply_tokens / wall, "prompt_n": sum(r["prompt_n"] for r in res), "cache_n": sum(r["cache_n"] for r in res),
           "outline": outline, "n_trunc": sum(1 for r in res if r["truncated"])}
    return out, reply

def seq_reply(topic, target_words):
    p = "Write a clear, well-organised explanation of %s in about %d words." % (topic, target_words)
    r = complete(p, int(target_words * 2) + 60)
    return {"wall": r["wall"], "tokens": r["tokens"], "words": len(words(r["text"])), "pps": r["pps"], "truncated": r["truncated"]}, r["text"]

# ---------- server ----------
def smi():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    u, m = [float(x) for x in o.split(",")]; return u, m

def idle():
    for _ in range(120):
        u, m = smi()
        if u < 5 and m < 300: return {"util": u, "mem_mib": m}
        time.sleep(2)
    raise RuntimeError("GPU not idle")

def start_server(np_):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log = open(os.path.join(ROOT, "logs", "s3-np%d.log" % np_), "w")
    cmd = [SERVER, "-m", MODEL, "-ngl", "99", "-fa", "on", "-np", str(np_), "-c", str(2048 * np_), "--no-webui", "--host", "127.0.0.1", "--port", str(PORT)]
    p = subprocess.Popen(cmd, stdout=log, stderr=log)
    for _ in range(120):
        time.sleep(1)
        if p.poll() is not None: raise RuntimeError("server exited early")
        try:
            if OPENER.open(BASE + "/health", timeout=2).status == 200: return p, " ".join(cmd[1:])
        except Exception: pass
    p.kill(); raise RuntimeError("server start timeout")

# ---------- driver ----------
PREFIX_CFG = (6, 100); PREFIX_TOPICS = 6

def plain_control(reps):
    runs = []
    for _ in range(reps):
        r = complete("Write a ~400-word explanation of how electric cars work.", 256)
        runs.append({"tokens": r["tokens"], "pps": r["pps"], "wall": r["wall"]})
    return runs

def plan(phase, topics):
    cfgs = []  # (variant, N, L, topic_ids)
    allt = [t[0] for t in topics]; t3 = allt[:3]
    if phase == "main":
        for v in VARIANTS: cfgs.append((v, 6, 100, allt))
    elif phase == "sweeps":  # rest of the {4,6,8} x {50,100,200} grid on T1-T3 (N=6 L=100 comes from main)
        for n in (4, 8): cfgs.append(("distinct_ctx_ic", n, 100, t3))
        for L in (50, 200):
            for n in (4, 6, 8): cfgs.append(("distinct_ctx_ic", n, L, t3))
    elif phase == "np":
        for n in (6, 8): cfgs.append(("distinct_ctx_ic", n, 100, t3))
    elif phase == "prefix":
        n, L = PREFIX_CFG
        for v in ("distinct_ctx_ic", "distinct_ctx_ic_prefix"): cfgs.append((v, n, L, allt[:PREFIX_TOPICS]))
    return cfgs

def run_server_phase(np_, phases, reps, data, samples, tag=""):
    idle_state = idle()
    proc, cmd = start_server(np_)
    key = "np%d%s" % (np_, tag)
    rec = {"np": np_, "cmd": cmd, "idle_before": idle_state, "control_start": None, "control_end": None, "cfgs": []}
    try:
        # warm-up (discarded): one SoT + one sequential
        t = TOPICS[0]; sot(t[1], t[2], 4, 100, "distinct_ctx_ic", np_); seq_reply(t[1], 200); complete("Hello", 8)
        rec["control_start"] = plain_control(5)
        seqcache = {}
        for ph in phases:
            for (variant, n, L, tids) in plan(ph, TOPICS):
                for tid in tids:
                    tp = [x for x in TOPICS if x[0] == tid][0]
                    entry = {"phase": ph, "variant": variant, "N": n, "L": L, "topic": tid, "sot": [], "seq": []}
                    for rep in range(reps):
                        s, reply = sot(tp[1], tp[2], n, L, variant, np_)
                        if rep == 0:
                            q = quality(reply, tp[2]); q["rubric_auto"] = rubric_auto(q); entry["quality_sot"] = q
                            target = max(25, int(round(s["reply_words"] / 25.0)) * 25); entry["seq_target_words"] = target
                            if tid in ("T1", "T2") and np_ == 4 and ph in ("main", "prefix"):
                                samples.setdefault(tid, {})[variant] = {"outline": s["outline"], "reply": reply}
                        entry["sot"].append(s)
                        sk = (tid, target)
                        sr, stext = seq_reply(tp[1], target)
                        if rep == 0 and "quality_seq" not in entry:
                            q = quality(stext, tp[2]); q["rubric_auto"] = rubric_auto(q); entry["quality_seq"] = q
                            if tid in ("T1", "T2") and np_ == 4 and ph in ("main",) and variant == "naive":
                                samples.setdefault(tid, {})["sequential_%d" % target] = {"reply": stext}
                        entry["seq"].append(sr)
                    rec["cfgs"].append(entry)
                    print("np=%d %s %s N=%d L=%d %s: sot %.2fs %dw seq %.2fs %dw" % (np_, ph, variant, n, L, tid, st.mean(x["wall"] for x in entry["sot"]),
                          entry["sot"][0]["reply_words"], st.mean(x["wall"] for x in entry["seq"]), entry["seq"][0]["words"]), flush=True)
                    data["servers"][key] = rec
                    json.dump(data, open(OUT, "w")); json.dump(samples, open(SAMPLES, "w"), indent=1)
        rec["control_end"] = plain_control(5)
        u, m = smi(); rec["vram_end_mib"] = m
    finally:
        proc.terminate()
        try: proc.wait(15)
        except Exception: proc.kill()
    data["servers"][key] = rec
    json.dump(data, open(OUT, "w")); json.dump(samples, open(SAMPLES, "w"), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", default=["np4:main", "np4sw:sweeps", "np1:np", "np8:np", "np4pf:prefix"])
    ap.add_argument("--prefix-cfg", default="6,100")
    ap.add_argument("--prefix-topics", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    global PREFIX_CFG, PREFIX_TOPICS
    PREFIX_CFG = tuple(int(x) for x in a.prefix_cfg.split(",")); PREFIX_TOPICS = a.prefix_topics
    data = json.load(open(OUT)) if os.path.exists(OUT) else {"servers": {}}
    samples = json.load(open(SAMPLES)) if os.path.exists(SAMPLES) else {}
    for item in a.plan:
        srv, phases = item.split(":")
        m = re.match(r"np(\d+)(.*)", srv)
        if not m: raise SystemExit("bad plan item " + item)
        if srv in data["servers"] and data["servers"][srv].get("control_end"): print("skip", srv); continue
        run_server_phase(int(m.group(1)), phases.split(","), a.reps, data, samples, tag=m.group(2))
        time.sleep(3)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
