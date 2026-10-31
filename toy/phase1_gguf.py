"""Phase 1.4: look inside the real model (Qwen3.5-0.8B bf16 GGUF).

GGUFReader memory-maps the file, so nothing is loaded onto the GPU.
Run from the repo root:  .venv/bin/python toy/phase1_gguf.py [--keys] [--tensors]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from gguf import GGUFReader

PATH = Path("models/Qwen_Qwen3.5-0.8B-bf16.gguf")
OUT = Path("results/phase1/gguf.json")

ap = argparse.ArgumentParser()
ap.add_argument("--keys", action="store_true", help="print every metadata key")
ap.add_argument("--tensors", action="store_true", help="print every tensor")
args = ap.parse_args()

r = GGUFReader(PATH)
arch = r.fields["general.architecture"].contents()


def field(key):
    return r.fields[key].contents()


if args.keys:
    for k in r.fields:
        print(k)

# Prefix = "blk.N" for block tensors, else the tensor name minus ".weight".
groups = defaultdict(lambda: {"bytes": 0, "tensors": []})
for t in r.tensors:
    if args.tensors:
        print(t.name, [int(x) for x in t.shape], t.tensor_type.name, int(t.n_bytes))
    parts = t.name.split(".")
    prefix = ".".join(parts[:2]) if parts[0] == "blk" else t.name.removesuffix(".weight")
    groups[prefix]["bytes"] += int(t.n_bytes)
    groups[prefix]["tensors"].append(t.name)

total = sum(g["bytes"] for g in groups.values())
emb = r.tensors[[t.name for t in r.tensors].index("token_embd.weight")]
# GGUF shapes are [ne0, ne1] = [width, vocab] (fastest-moving dim first).
width, vocab = (int(x) for x in emb.shape)
has_output = "output" in groups      # a separate output.weight would mean an untied head

kinds = {}
for prefix, g in groups.items():
    if not prefix.startswith("blk."):
        continue
    names = " ".join(g["tensors"])
    # Careful: recurrent blocks also have a tensor called "attn_qkv"; ssm_* is the tell.
    if ".nextn." in names:
        kinds[prefix] = "mtp(attn)" if ".attn_q." in names else "mtp"
    elif ".ssm_" in names:
        kinds[prefix] = "recurrent"
    elif ".attn_q." in names:
        kinds[prefix] = "attention"
    else:
        kinds[prefix] = "unknown"

blk_ids = lambda k: sorted(int(p.split(".")[1]) for p, v in kinds.items() if v == k)
attn, rec = blk_ids("attention"), blk_ids("recurrent")
mtp = sorted(int(p.split(".")[1]) for p, v in kinds.items() if v.startswith("mtp"))
mtp_bytes = sum(groups[f"blk.{i}"]["bytes"] for i in mtp)
emb_frac_no_mtp = int(emb.n_bytes) / (total - mtp_bytes)

print(f"file={PATH}  arch={arch}  tensors={len(r.tensors)}")
print(f"block_count={field(f'{arch}.block_count')}  nextn_predict_layers={field(f'{arch}.nextn_predict_layers')}"
      f"  full_attention_interval={field(f'{arch}.full_attention_interval')}\n")
print(f"{'prefix':16s} {'kind':10s} {'tensors':>7s} {'bytes':>13s} {'share':>7s}")
order = sorted(groups, key=lambda p: (p.startswith("blk."), int(p.split(".")[1]) if p.startswith("blk.") else 0))
for p in order:
    g = groups[p]
    print(f"{p:16s} {kinds.get(p, ''):10s} {len(g['tensors']):7d} {g['bytes']:13,d} {g['bytes'] / total:7.2%}")
print(f"{'total':16s} {'':10s} {len(r.tensors):7d} {total:13,d}\n")

print(f"vocab={vocab} (tokenizer.ggml.tokens has {len(r.fields['tokenizer.ggml.tokens'].data)})  "
      f"width={width} (embedding_length={field(f'{arch}.embedding_length')})")
print(f"separate output.weight tensor: {has_output}  -> head is {'untied' if has_output else 'tied to token_embd'}")
print(f"token_embd: {emb.tensor_type.name} {int(emb.n_bytes):,d} bytes = {int(emb.n_bytes) / total:.2%} of all tensor bytes")
print(f"attention blocks: {attn}")
print(f"recurrent blocks: {rec}")
print(f"MTP block(s): {mtp} ({[kinds[f'blk.{i}'] for i in mtp]})  bytes={mtp_bytes:,d} ({mtp_bytes / total:.2%})")
print(f"token_embd share if the MTP block is left out: {emb_frac_no_mtp:.2%}")
print(f"file size on disk: {PATH.stat().st_size:,d} bytes")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "path": str(PATH), "arch": arch, "file_bytes": PATH.stat().st_size,
    "total_tensor_bytes": total, "n_tensors": len(r.tensors),
    "vocab": vocab, "width": width, "token_embd_type": emb.tensor_type.name,
    "token_embd_bytes": int(emb.n_bytes), "token_embd_fraction": int(emb.n_bytes) / total,
    "separate_output_tensor": has_output,
    "attention_blocks": attn, "recurrent_blocks": rec, "mtp_blocks": mtp, "mtp_bytes": mtp_bytes,
    "token_embd_fraction_without_mtp": emb_frac_no_mtp,
    "groups": {p: {"kind": kinds.get(p), "bytes": groups[p]["bytes"], "tensors": groups[p]["tensors"]}
               for p in order},
}, indent=2))
print(f"wrote {OUT}")
