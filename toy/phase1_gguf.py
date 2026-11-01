"""Phase 1.4: look inside the real model, Qwen3.5-0.8B bf16 GGUF (plan.md 1.4).

A GGUF file is a header (metadata: architecture, tokenizer, hyperparameters), a table describing
every tensor, and then the raw tensor bytes. This script reads models/Qwen_Qwen3.5-0.8B-bf16.gguf
and groups its tensors by name prefix: token_embd, blk.0 .. blk.23, blk.24 (the MTP / "nextn"
block), output_norm, other. A block counts as "mtp" if it has nextn.* tensors, else "recurrent"
(Gated DeltaNet) if it has ssm_* tensors, else "attention" if it has attention q/k/v weights.

Shapes are in GGUF (ggml) order: the first number is the row length. token_embd is [1024, 248320]
because the vocabulary is the number of rows, one 1024-wide vector per token.

  python toy/phase1_gguf.py [--model models/Qwen_Qwen3.5-0.8B-bf16.gguf] [--out results/1]

Writes <out>/gguf.json. Everything under "deterministic" must be identical on a re-run.
"""
import argparse, json, os, re
from datetime import datetime, timezone
from pathlib import Path

from gguf import GGUFReader, GGUFValueType

ROOT = Path(__file__).resolve().parent.parent
MIB = 1 << 20


def pyval(v):
    """Turn numpy scalars and arrays into plain Python values so json can write them."""
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [pyval(x) for x in v]
    return v


def field_value(f):
    """Scalars as values; arrays as their length and element type (token lists are huge)."""
    if f.types[0] == GGUFValueType.ARRAY:
        elem = f.types[-1].name
        n = len(f.data)
        if elem == "STRING":
            return {"array_of": elem, "length": n}
        return {"array_of": elem, "length": n, "values": [pyval(v) for v in f.contents()][:8]}
    return pyval(f.contents())


def group_of(name):
    """token_embd.* -> token_embd, blk.N.* -> blk.N, output_norm.* -> output_norm, else other."""
    if name.startswith("token_embd."):
        return "token_embd"
    m = re.match(r"blk\.(\d+)\.", name)
    if m:
        return f"blk.{m.group(1)}"
    if name.startswith("output_norm."):
        return "output_norm"
    return "other"


def group_order(g):
    """token_embd, output_norm, other, then blk.0 .. blk.24 in numeric order."""
    if g.startswith("blk."):
        return (1, int(g.split(".")[1]))
    return (0, {"token_embd": 0, "output_norm": 1, "other": 2}.get(g, 3))


def block_kind(inner_names):
    if any(n.startswith("nextn.") for n in inner_names):
        return "mtp"
    if any(n.startswith("ssm_") for n in inner_names):
        return "recurrent"
    if any(n.startswith(("attn_q.", "attn_k.", "attn_v.", "attn_qkv.")) for n in inner_names):
        return "attention"
    return "other"


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FF_MODEL", str(ROOT / "models/Qwen_Qwen3.5-0.8B-bf16.gguf")))
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()

    r = GGUFReader(a.model)
    fields = {k: field_value(f) for k, f in r.fields.items() if not k.startswith("GGUF.")}
    tensors = [{"name": t.name, "shape": [int(x) for x in t.shape], "type": t.tensor_type.name,
                "n_elements": int(t.n_elements), "n_bytes": int(t.n_bytes)} for t in r.tensors]
    tensors.sort(key=lambda t: t["name"])

    # ---- per tensor type: how many tensors and bytes of BF16, F32, ...
    types = {}
    for t in tensors:
        s = types.setdefault(t["type"], {"n_tensors": 0, "n_bytes": 0, "n_elements": 0})
        s["n_tensors"] += 1
        s["n_bytes"] += t["n_bytes"]
        s["n_elements"] += t["n_elements"]
    types = {k: types[k] for k in sorted(types)}

    # ---- groups by name prefix
    groups = {g: {"n_tensors": 0, "n_bytes": 0, "n_elements": 0, "types": {}, "tensor_names": []}
              for g in ("token_embd", "output_norm", "other")}
    for t in tensors:
        g = group_of(t["name"])
        s = groups.setdefault(g, {"n_tensors": 0, "n_bytes": 0, "n_elements": 0, "types": {}, "tensor_names": []})
        s["n_tensors"] += 1
        s["n_bytes"] += t["n_bytes"]
        s["n_elements"] += t["n_elements"]
        s["types"][t["type"]] = s["types"].get(t["type"], 0) + 1
        s["tensor_names"].append(t["name"])
    groups = {g: {**groups[g], "tensor_names": sorted(groups[g]["tensor_names"])}
              for g in sorted(groups, key=group_order)}

    # ---- one record per transformer block
    blocks = {}
    for g, s in groups.items():
        if not g.startswith("blk."):
            continue
        inner = [n.split(".", 2)[2] for n in s["tensor_names"]]
        blocks[g] = {"block": int(g.split(".")[1]), "n_tensors": s["n_tensors"],
                     "n_bytes": s["n_bytes"], "n_elements": s["n_elements"],
                     "kind": block_kind(inner), "tensor_names": sorted(inner)}
    blocks = {g: blocks[g] for g in sorted(blocks, key=lambda x: blocks[x]["block"])}

    # ---- block numbers and byte sizes per kind
    kinds = {}
    for g, b in blocks.items():
        k = kinds.setdefault(b["kind"], {"blocks": [], "bytes": {}, "total_bytes": 0})
        k["blocks"].append(b["block"])
        k["bytes"][str(b["block"])] = b["n_bytes"]
        k["total_bytes"] += b["n_bytes"]
    kinds = {k: {**kinds[k], "n_blocks": len(kinds[k]["blocks"])} for k in sorted(kinds)}

    total = sum(t["n_bytes"] for t in tensors)
    embd = next(t for t in tensors if t["name"] == "token_embd.weight")
    arch = fields["general.architecture"]
    special = {k: v for k, v in fields.items()
               if re.search(r"nextn|mtp|full_attention", k, re.I)}
    metadata = {
        "architecture": arch,
        "block_count": fields.get(f"{arch}.block_count"),
        "embedding_length": fields.get(f"{arch}.embedding_length"),
        "context_length": fields.get(f"{arch}.context_length"),
        "vocab_size": len(r.fields["tokenizer.ggml.tokens"].data),
        "nextn_mtp_and_full_attention_keys": special,
    }
    file_size = os.path.getsize(a.model)
    mtp_bytes = sum(b["n_bytes"] for b in blocks.values() if b["kind"] == "mtp")

    det = {
        "model_file": Path(a.model).name,
        "file_bytes": file_size,
        "tensor_data_offset": int(r.data_offset),
        "n_tensors": len(tensors),
        "total_tensor_bytes": total,
        "total_params": sum(t["n_elements"] for t in tensors),
        "bytes_not_in_tensors": file_size - total,
        "tensor_types": types,
        "token_embd": {"shape": embd["shape"], "type": embd["type"], "n_bytes": embd["n_bytes"],
                       "share_of_tensor_bytes": embd["n_bytes"] / total},
        "output_weight_present": any(t["name"] == "output.weight" for t in tensors),
        "groups": groups,
        "blocks": blocks,
        "block_kinds": kinds,
        "mtp_bytes": mtp_bytes,
        "metadata": metadata,
        "tensors": tensors,
    }

    print(f"file {file_size:,} bytes; {len(tensors)} tensors; {total:,} tensor bytes "
          f"({total / MIB:.1f} MiB); {det['total_params']:,} parameters")
    print(f"token_embd {embd['n_bytes']:,} bytes ({100 * det['token_embd']['share_of_tensor_bytes']:.2f} %); "
          f"separate output.weight: {det['output_weight_present']}")
    print(f"vocab {metadata['vocab_size']:,}; width {metadata['embedding_length']}; "
          f"blocks {metadata['block_count']}; MTP bytes {mtp_bytes:,}")
    for k, v in kinds.items():
        print(f"  {k:9s} blocks {v['blocks']} total {v['total_bytes']:,} bytes")
    print(f"bytes not in tensors: {det['bytes_not_in_tensors']:,}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_gguf.py", "started_utc": started, "deterministic": det}
    (out / "gguf.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
