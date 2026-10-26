#!/usr/bin/env python3
r"""B4: frequency-ranked token subset for the trimmed MTP draft head.
Unseen tokens ranked by id ascending (BPE merge order proxy). Corpus (held out from F1-F6 bf16 outputs on purpose): wikitext-2 test text (data/wiki.test.raw), every string in the
s3-samples / parallel-subtasks raw jsons, and the repo markdown. Tokenised with llama-tokenize (same GGUF vocab).
Writes results/2026-10-26-bf16-b4-token-rank.txt (one id per line, most frequent first; EOG/special ids first) and prints coverage of
the plain-control F1-F6 outputs (not used for ranking) for K in 8k/16k/32k.
Usage: python scripts\b4_rank_tokens.py"""
import json, os, re, glob, gzip, subprocess, collections, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "llama-cpp-b4", "llama-tokenize.exe")
MODEL = os.path.join(ROOT, "models", "Qwen_Qwen3.5-0.8B-bf16.gguf")
OUT = os.path.join(ROOT, "results", "2026-10-26-bf16-b4-token-rank.txt")

def strings(o):
    if isinstance(o, str):
        if len(o) > 40: yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from strings(v)
    elif isinstance(o, list):
        for v in o: yield from strings(v)

def tokenize(text, special=False):
    p = os.path.join(ROOT, "data", "b4_tok_in.txt")
    open(p, "w", encoding="utf-8", newline="").write(text)
    args = [TOK, "-m", MODEL, "-f", p, "--ids", "--no-bos", "--log-disable"] + ([] if special else ["--no-parse-special"])
    o = subprocess.run(args, capture_output=True, text=True, encoding="utf-8").stdout
    return [int(x) for x in re.findall(r"-?\d+", o[o.index("["):o.rindex("]")])]

def main():
    parts = [open(os.path.join(ROOT, "data", "wiki.test.raw"), encoding="utf-8", errors="ignore").read()]
    for f in ["2026-10-26-s3-samples.json", "2026-10-26-parallel-subtasks.raw.json", "2026-10-26-s3-sot-0.8b.raw.json", "2026-10-26-s1-quants-0.8b.raw.json"]:
        parts += list(strings(json.load(open(os.path.join(ROOT, "results", f), encoding="utf-8"))))
    parts += list(strings(json.load(gzip.open(os.path.join(ROOT, "results", "2026-10-26-s2-spec-0.8b.raw.json.gz"), "rt", encoding="utf-8"))))
    for f in glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True):
        if "llama-src" in f or "llama-cpp" in f: continue
        parts.append(open(f, encoding="utf-8", errors="ignore").read())
    corpus = "\n".join(parts)
    ids = tokenize(corpus)
    cnt = collections.Counter(ids)
    special = []
    for s in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<think>", "</think>"):
        special += tokenize(s, special=True)
    ranked = special + [t for t, _ in cnt.most_common() if t not in special]
    # tokens never seen in the corpus: fall back to id order (byte-level BPE ids roughly follow merge order ~ frequency)
    seen = set(ranked)
    ranked += [t for t in range(248000) if t not in seen]
    open(OUT, "w", newline="\n").write("\n".join(map(str, ranked)) + "\n")
    print("corpus chars %d tokens %d distinct %d; specials %s" % (len(corpus), len(ids), len(cnt), special))
    # coverage on held-out plain-control outputs (F1-F6, first pass text)
    d = json.load(gzip.open(os.path.join(ROOT, "results", "2026-10-26-bf16-B3-mtp.raw.json.gz"), "rt", encoding="utf-8"))
    ctrl = d["configs"]["control"]["runs"][0]
    gen = []
    for r in ctrl: gen += tokenize(r["text"])
    rank = {t: i for i, t in enumerate(ranked)}
    for K in (4096, 8192, 16384, 32768, 65536):
        print("K=%6d coverage of F1-F6 plain output tokens: %.1f%% (n=%d)" % (K, 100 * sum(rank.get(t, 1e9) < K for t in gen) / len(gen), len(gen)))

main()
