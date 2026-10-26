"""E5 data: per-position final hidden states (post output-norm) of the bf16 target for the generated text of chosen configs.
Needs a llama-server running with --embeddings --pooling none on port 8095 (CPU is fine). Saves logs/e5_hidden_<tag>.npz (git-ignored)."""
import json, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_bench as eb, bf16_bench as bb
BASE = "http://127.0.0.1:8095"
def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=600).read())
def main(cfgs, tag):
    D = json.load(open(eb.OUT))["configs"]
    H, T, SRC = [], [], []
    for cfg in cfgs:
        rec = D[cfg]
        for p, r in zip(eb.SETS[rec["set"]], rec["runs"][0]):
            pre = bb.chatml(p["user"])
            ids_pre = post("/tokenize", {"content": pre, "add_special": False, "parse_special": True})["tokens"]
            ids_all = post("/tokenize", {"content": pre + r["text"], "add_special": False, "parse_special": True})["tokens"]
            e = post("/embedding", {"content": pre + r["text"], "embd_normalize": -1})
            emb = np.array(e[0]["embedding"], dtype=np.float32)
            assert emb.shape[0] == len(ids_all), (emb.shape, len(ids_all))
            # position t predicts token t+1; keep generated part: t >= len(ids_pre)-1, t < len-1
            sl = slice(len(ids_pre) - 1, len(ids_all) - 1)
            H.append(emb[sl]); T.append(np.array(ids_all[len(ids_pre):], dtype=np.int32)); SRC += ["%s/%s" % (cfg, p["id"])] * (sl.stop - sl.start)
            print(cfg, p["id"], emb.shape, len(ids_pre), len(ids_all), flush=True)
    np.savez(os.path.join(eb.ROOT, "logs", "e5_hidden_%s.npz" % tag), h=np.concatenate(H), tok=np.concatenate(T), src=np.array(SRC))
if __name__ == "__main__": main(sys.argv[2:], sys.argv[1])
