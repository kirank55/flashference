#!/usr/bin/env python3
r"""Parallel sub-task experiments against llama-server (stdlib only).

Baseline A : one ~400-word reply, single stream.
Exp 1      : N independent short sub-tasks, sequential (np=1) vs parallel slots (-np 2/4/8/16).
Exp 2      : skeleton-of-thought: outline (1 call) -> expand points in parallel -> concatenate.
Usage: python scripts\parallel_subtasks.py [--np 1 2 4 8 16] [--reps 5] [--model Q4_K_M]
"""
import argparse, json, os, re, subprocess, sys, threading, time, statistics as st, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "llama-cpp", "llama-server.exe")
PORT = 8089
BASE = "http://127.0.0.1:%d" % PORT
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def chatml(user):  # Qwen chat format, thinking disabled (empty think block)
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def http(path, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())

def complete(user, n_predict):
    t0 = time.perf_counter()
    r = http("/completion", {"prompt": chatml(user), "n_predict": n_predict, "temperature": 0.0,
                             "seed": 1, "cache_prompt": False, "stream": False})
    wall = time.perf_counter() - t0
    tm = r["timings"]
    return {"text": r["content"].strip(), "tokens": tm["predicted_n"], "pps": tm["predicted_per_second"],
            "prompt_n": tm["prompt_n"], "wall": wall}

def run_jobs(jobs, parallel):
    """jobs = [(user_prompt, n_predict)]. Returns list of results + total wall."""
    res = [None] * len(jobs)
    t0 = time.perf_counter()
    if parallel:
        def w(i):
            res[i] = complete(*jobs[i])
        ths = [threading.Thread(target=w, args=(i,)) for i in range(len(jobs))]
        [t.start() for t in ths]; [t.join() for t in ths]
    else:
        for i, j in enumerate(jobs):
            res[i] = complete(*j)
    return res, time.perf_counter() - t0

# ---------------- GPU sampling ----------------
def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    try:
        return [float(x) for x in out.split(",")]
    except Exception:
        return None

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.s = []
    def run(self):
        while not self.stop:
            v = smi()
            if v: self.s.append(v)
            time.sleep(0.4)
    def summary(self):
        if not self.s: return {}
        busy = [x for x in self.s if x[1] > 30] or self.s
        return {"peak_mem_mib": max(x[0] for x in self.s), "max_temp_c": max(x[2] for x in self.s),
                "mean_power_w_busy": round(st.mean(x[3] for x in busy), 1),
                "mean_sm_mhz_busy": round(st.mean(x[4] for x in busy)), "mean_util_busy": round(st.mean(x[1] for x in busy))}

# ---------------- workloads ----------------
TEXTS = [
 "The city council voted on Tuesday to approve a new bike lane network spanning forty kilometres. Supporters say it will cut traffic and emissions, while shop owners worry that losing curbside parking will hurt sales. Construction begins in March and is expected to take eighteen months.",
 "Researchers have found that a common soil bacterium can break down certain plastics at room temperature. In lab tests, a thin film of polyethylene lost about ten percent of its mass in six weeks. The team cautions that scaling the process to industrial levels remains an open challenge.",
 "The company announced quarterly revenue of 4.2 billion dollars, up eight percent from last year, driven by cloud services. However, hardware sales fell for the third consecutive quarter, and executives warned that supply costs would stay elevated through the next year.",
 "A new study of sleep habits among university students found that those who went to bed after 2 a.m. scored lower on morning exams on average. The authors note that caffeine use and part-time work could partly explain the gap, and call for larger trials.",
 "The museum reopened its Egyptian wing after a two-year renovation. Visitors can now see forty newly conserved artifacts, including a painted coffin that had been in storage for decades. Tickets are free on the first Sunday of each month.",
]
TRANSL = [
 "Translate into French, then explain two grammar points in your translation: 'The train to the coast leaves at half past seven. Please bring a warm jacket, because it is usually windy by the sea. We will have lunch at a small restaurant near the harbour.'",
 "Translate into Spanish, then explain two grammar points in your translation: 'Our library is open every day except Monday. Members can borrow up to ten books for three weeks. If you need help finding a title, please ask the staff at the front desk.'",
 "Translate into German, then explain two grammar points in your translation: 'Thank you for your order. Your package will be shipped tomorrow and should arrive within five working days. You will receive a tracking number by email.'",
 "Translate into Italian, then explain two grammar points in your translation: 'The weather this weekend will be sunny with a light breeze. Temperatures will reach twenty degrees in the afternoon. It is a perfect time to go hiking in the hills.'",
]
CLASSIFY = [
 "Classify the sentiment (positive, negative or mixed) of this review, then justify in 4 to 5 sentences: 'The phone is fast and the camera is superb, but the battery barely lasts a day and the charger feels cheap.'",
 "Classify this support ticket as billing, technical, or account, then justify in 4 to 5 sentences: 'I was charged twice this month for my subscription and I can not find the invoice in my dashboard.'",
 "Classify the tone of this email (formal, casual, or angry), then justify in 4 to 5 sentences: 'Hey, just checking if you got the slides I sent over yesterday. No rush, but would be great to chat before Friday!'",
 "Classify this headline as politics, sports, science, or business, then justify in 4 to 5 sentences: 'Central bank holds interest rates steady as inflation cools to a two-year low.'",
]
EXPLAIN = [
 "Explain in about 100 words what a hash table is and why lookups are fast.",
 "Explain in about 100 words why the sky appears blue during the day.",
 "Explain in about 100 words the difference between a process and a thread.",
]
def exp1_tasks(n):
    items = [("Summarize the following text in about 90 words:\n\n" + t) for t in TEXTS] + TRANSL + CLASSIFY + EXPLAIN  # 16
    return [(p, 100) for p in (items * (n // len(items) + 1))[:n]]  # cycles if n > 16

TOPIC = "how electric cars work"
N_POINTS = 6
def baseline_job():
    return [("Write a ~400-word explanation of %s." % TOPIC, 700)]

LENGTHS = {"short": ("4 to 5 sentences (about 70 words)", 130), "long": ("5 to 6 sentences (about 100 words)", 200)}
def skeleton(parallel, variant):
    """Returns dict with timings + reply. parallel=False -> expand sequentially (for np=1)."""
    t0 = time.perf_counter()
    o = complete("List exactly %d key points for an explanation of %s. Output only a numbered list 1-%d, each point a short phrase of at most 8 words. No other text." % (N_POINTS, TOPIC, N_POINTS), 80)
    t_outline = time.perf_counter() - t0
    pts = [m.group(1).strip() for m in (re.match(r"^\s*\d+[.)]\s*(.+?)\s*$", l) for l in o["text"].splitlines()) if m][:N_POINTS]
    outline = "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(pts))
    jobs = [("You are writing one section of a short explanation of %s. The full outline is:\n%s\n\nWrite one detailed paragraph of %s covering only point %d (\"%s\"). Do not write an introduction or conclusion and do not cover the other points." % (TOPIC, outline, LENGTHS[variant][0], i + 1, p), LENGTHS[variant][1])
            for i, p in enumerate(pts)]
    res, t_expand = run_jobs(jobs, parallel)
    reply = "\n\n".join(r["text"] for r in res)
    total = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in res)
    return {"n_points": len(pts), "outline": outline, "t_outline": t_outline, "t_expand": t_expand, "t_total": total,
            "reply_tokens": toks, "outline_tokens": o["tokens"], "reply_words": len(reply.split()), "reply": reply,
            "eff_tps": toks / total, "agg_tps_all": (toks + o["tokens"]) / total,
            "per_stream_tps": st.mean(r["pps"] for r in res)}

# ---------------- server mgmt ----------------
def start_server(model, np_):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log = open(os.path.join(ROOT, "logs", "server-np%d.log" % np_), "w")
    cmd = [SERVER, "-m", os.path.join(ROOT, "models", "Qwen_Qwen3.5-0.8B-%s.gguf" % model), "-ngl", "99", "-fa", "on",
           "-np", str(np_), "-c", str(2048 * np_), "--no-webui", "--host", "127.0.0.1", "--port", str(PORT)]
    p = subprocess.Popen(cmd, stdout=log, stderr=log)
    for _ in range(120):
        time.sleep(1)
        if p.poll() is not None: raise RuntimeError("server exited early, see logs")
        try:
            if OPENER.open(BASE + "/health", timeout=2).status == 200: return p, " ".join(cmd[1:])
        except Exception: pass
    raise RuntimeError("server start timeout")

def stats(xs):
    xs = list(xs)
    return {"mean": st.mean(xs), "min": min(xs), "max": max(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0}

def wait_idle():
    for _ in range(60):
        v = smi()
        if v and v[1] < 5: return v
        time.sleep(1)
    return smi()

def saturation(a):
    data = {"model": a.model, "reps": a.reps, "tasks": a.sat, "configs": []}
    for np_ in a.np:
        wait_idle()
        proc, cmd = start_server(a.model, np_)
        samp = Sampler(); samp.start()
        cfg = {"np": np_, "cmd": cmd, "runs": []}
        try:
            run_jobs(exp1_tasks(a.sat), np_ > 1)  # warm-up, discarded
            for rep in range(a.reps):
                r, w = run_jobs(exp1_tasks(a.sat), np_ > 1)
                tot = sum(x["tokens"] for x in r)
                cfg["runs"].append({"wall": w, "tokens": tot, "agg_tps": tot / w, "per_stream_tps": st.mean(x["pps"] for x in r)})
                print("sat np=%d rep=%d agg=%.1f" % (np_, rep + 1, tot / w), flush=True)
            cfg["gpu"] = samp.summary()
        finally:
            samp.stop = True; proc.terminate()
            try: proc.wait(15)
            except Exception: proc.kill()
        data["configs"].append(cfg)
        time.sleep(3)
    out = os.path.join(ROOT, "results", "2026-10-26-parallel-subtasks-saturation.raw.json")
    json.dump(data, open(out, "w"), indent=1)
    print("DONE", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model", default="Q4_K_M")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "2026-10-26-parallel-subtasks.raw.json"))
    ap.add_argument("--sat", type=int, default=0, help="saturation mode: only Exp1 with this many tasks (cycled), all submitted at once")
    a = ap.parse_args()
    if a.sat:
        return saturation(a)
    data = {"model": a.model, "reps": a.reps, "configs": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    order = a.np + [1]  # final np=1 pass = drift check
    for idx, np_ in enumerate(order):
        idle = wait_idle()
        proc, cmd = start_server(a.model, np_)
        cfg = {"np": np_, "drift_check": idx == len(a.np), "cmd": cmd, "idle_before": idle}
        samp = Sampler(); samp.start()
        try:
            par = np_ > 1
            # warm-up (discarded): one of each workload
            run_jobs(baseline_job(), False); run_jobs(exp1_tasks(16), par); skeleton(par, "short"); skeleton(par, "long")
            cfg["vram_after_warmup_mib"] = smi()[0]
            cfg["baseline"] = []; cfg["exp1_8"] = []; cfg["exp1_16"] = []; cfg["exp2_short"] = []; cfg["exp2_long"] = []
            for rep in range(a.reps):
                r, w = run_jobs(baseline_job(), False)
                cfg["baseline"].append({"wall": w, "tokens": r[0]["tokens"], "pps": r[0]["pps"], "tps_wall": r[0]["tokens"] / w,
                                        "words": len(r[0]["text"].split()), "text": r[0]["text"] if rep == 0 else None})
                for n, key in ((8, "exp1_8"), (16, "exp1_16")):
                    r, w = run_jobs(exp1_tasks(n), par)
                    tot = sum(x["tokens"] for x in r)
                    cfg[key].append({"wall": w, "tokens": tot, "agg_tps": tot / w, "per_stream_tps": st.mean(x["pps"] for x in r),
                                     "mean_latency": st.mean(x["wall"] for x in r), "max_latency": max(x["wall"] for x in r)})
                for v in ("short", "long"):
                    s = skeleton(par, v)
                    if rep > 0: s["reply"] = None
                    cfg["exp2_" + v].append(s)
                print("np=%d rep=%d done" % (np_, rep + 1), flush=True)
            cfg["gpu"] = samp.summary()
        finally:
            samp.stop = True
            proc.terminate()
            try: proc.wait(15)
            except Exception: proc.kill()
        data["configs"].append(cfg)
        with open(a.out, "w") as f: json.dump(data, f, indent=1)
        time.sleep(3)
    data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(a.out, "w") as f: json.dump(data, f, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()




