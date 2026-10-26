"""CPU-only smoke test of the E2/E4 patch (no GPU used: -dev none). Usage: python scripts/e_smoke.py"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bf16_bench as bb
bb.PORT = 8097; bb.BASE = "http://127.0.0.1:8097"
E = json.load(open(os.path.join(bb.ROOT, "prompts", "e-prompts.json"), encoding="utf-8-sig"))
P = {p["id"]: p for p in E["lang"] + E["code"]}
R = lambda K, extra=None: dict({"LLAMA_MTP_TRIM_FILE": bb.RANK, "LLAMA_MTP_TRIM_K": str(K)}, **(extra or {}))
CF = {"untrim": {}, "trim8k": R(8192), "dyn4k+4k": R(8192, {"LLAMA_MTP_TRIM_FIXED": "4096", "LLAMA_MTP_DYN": "16"}),
      "trim8k-adapt4": R(8192, {"LLAMA_MTP_ADAPT": "1"})}
which = sys.argv[1:] or list(CF)
for name in which:
    n = 4 if "adapt" in name else 2
    args = ["-dev", "none", "-ngl", "0", "-fa", "off", "-np", "1", "-c", "1024", "-t", "8", "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]
    p, cmd = bb.start_server("smoke-" + name, "Qwen_Qwen3.5-0.8B-bf16.gguf", args, CF[name], "", lc="llama-cpp-e")
    try:
        for pid in ("L3-zh", "L4-hi", "K3"):
            t0 = time.time(); r = bb.complete(P[pid], 40)
            print(name, pid, "n", r["n"], "draft", r["draft_n"], "acc", r["draft_acc"], "%.0fs" % (time.time() - t0), "hash", hash(r["text"]) % 100000, flush=True)
    finally: bb.stop(p)

