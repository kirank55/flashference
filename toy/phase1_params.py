"""Phase 1.3: where are the parameters (and their bytes)?

Run from the repo root:  .venv/bin/python toy/phase1_params.py
"""
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT = Path("results/phase1/params.json")


def group(name):
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "self_attn" in name:
        return "attention"
    if "mlp" in name:
        return "mlp"
    if "norm" in name:
        return "norms"
    return "other"


model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)   # CPU is enough
cfg = model.config
tied = model.lm_head.weight is model.get_input_embeddings().weight

# remove_duplicate=False also lists lm_head.weight; the data_ptr check stops a
# tied matrix being counted twice.
groups = defaultdict(lambda: {"params": 0, "bytes": 0, "tensors": 0})
seen = {}
for name, p in model.named_parameters(remove_duplicate=False):
    g = group(name)
    if p.data_ptr() in seen:
        print(f"{name} shares storage with {seen[p.data_ptr()]} (not counted again)")
        groups[g]["tensors"] += 1
        continue
    seen[p.data_ptr()] = name
    groups[g]["params"] += p.numel()
    groups[g]["bytes"] += p.numel() * p.element_size()
    groups[g]["tensors"] += 1

total = sum(v["bytes"] for v in groups.values())
emb_head = groups["embedding"]["bytes"] + groups["lm_head"]["bytes"]

print(f"\nmodel={MODEL} dtype={next(model.parameters()).dtype}")
print(f"tie_word_embeddings={cfg.tie_word_embeddings}  lm_head is embed_tokens: {tied}")
print(f"vocab={cfg.vocab_size} hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}\n")
print(f"{'group':10s} {'tensors':>7s} {'params':>12s} {'bytes':>12s} {'share':>7s}")
for g in ["embedding", "attention", "mlp", "norms", "lm_head", "other"]:
    v = groups.get(g, {"params": 0, "bytes": 0, "tensors": 0})
    print(f"{g:10s} {v['tensors']:7d} {v['params']:12,d} {v['bytes']:12,d} {v['bytes'] / total:7.2%}")
print(f"{'total':10s} {'':7s} {sum(v['params'] for v in groups.values()):12,d} {total:12,d}")
print(f"\nembedding/head (one shared matrix if tied) = {emb_head:,d} bytes = {emb_head / total:.2%} of total")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "model": MODEL, "dtype": str(next(model.parameters()).dtype),
    "tie_word_embeddings": cfg.tie_word_embeddings, "lm_head_is_embedding": tied,
    "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
    "groups": dict(groups), "total_bytes": total,
    "embedding_head_bytes": emb_head, "embedding_head_fraction": emb_head / total,
}, indent=2))
print(f"wrote {OUT}")
