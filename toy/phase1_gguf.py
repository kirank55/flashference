"""Phase 1.4: look inside the real model, Qwen3.5-0.8B bf16 GGUF (plan.md 1.4).

Lists the metadata keys and every tensor (name, shape, type, bytes), then groups tensors by name
prefix: token_embd, blk.0 .. blk.23, blk.24 (the MTP / "nextn" block), output_norm. A block counts
as attention if it has attn_q / attn_k / attn_v weights, and as recurrent (Gated DeltaNet) if it
has ssm_* tensors. Shapes are printed in GGUF (ggml) order: the first number is the row length.

  python toy/phase1_gguf.py [--model models/Qwen_Qwen3.5-0.8B-bf16.gguf] [--out results/1]

Writes <out>/gguf.json. Everything under "deterministic" must be identical on a re-run.
"""
import argparse, json, os, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gguf import GGUFReader, GGUFValueType

ROOT = Path(__file__).resolve().parent.parent
MIB = 1 << 20


def field_value(f):
    """Scalars and strings as values; long arrays as their length and element type."""
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        if len(f.data) <= 16 and f.types[-1] != GGUFValueType.STRING:
            return [v.item() if hasattr(v, "item") else v for v in f.contents()]
        return {"array_of": f.types[-1].name, "length": len(f.data)}
    v = f.contents()
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def group_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return f"blk.{m.group(1)}" if m else name.rsplit(".", 1)[0]     # e.g. token_embd, output_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FF_MODEL", str(ROOT / "models/Qwen_Qwen3.5-0.8B-bf16.gguf")))
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r = GGUFReader(a.model)

    fields = {k: field_value(f) for k, f in r.fields.items() if not k.startswith("GGUF.")}
    tensors = [{"name": t.name, "shape": [int(x) for x in t.shape], "type": t.tensor_type.name,
                "n_elements": int(t.n_elements), "n_bytes": int(t.n_bytes)} for t in r.tensors]
    groups = defaultdict(lambda: {"n_tensors": 0, "n_bytes": 0, "types": Counter()})
    for t in tensors:
        g = groups[group_of(t["name"])]
        g["n_tensors"] += 1
        g["n_bytes"] += t["n_bytes"]
        g["types"][t["type"]] += 1
    blocks = {}
    for g in groups:
        if g.startswith("blk."):
            names = {t["name"].split(".", 2)[2] for t in tensors if group_of(t["name"]) == g}
            attn = any(n.startswith(("attn_q.", "attn_k.", "attn_v.")) for n in names)
            rec = any(n.startswith("ssm_") for n in names)
            inner = "attention" if attn and not rec else "recurrent" if rec and not attn else "other"
            mtp = any(n.startswith("nextn.") for n in names)        # the MTP block's extra tensors
            blocks[g] = {"kind": "mtp" if mtp else inner, "inner_layer": inner, "tensors": sorted(names)}

    total = sum(t["n_bytes"] for t in tensors)
    embd = groups["token_embd"]["n_bytes"]
    arch = fields.get("general.architecture")
    nblk = fields.get(f"{arch}.block_count")
    vocab = fields["tokenizer.ggml.tokens"]["length"]
    embd_t = next(t for t in tensors if t["name"] == "token_embd.weight")
    file_size = os.path.getsize(a.model)
    det = {
        "file_bytes": file_size,
        "tensor_data_offset": int(r.data_offset),          # where tensor data starts: header + metadata + padding
        "n_fields": len(fields), "fields": fields,
        "n_tensors": len(tensors), "tensors": tensors,
        "groups": {g: {**v, "types": dict(v["types"])} for g, v in sorted(groups.items(), key=lambda kv: kv[0])},
        "blocks": dict(sorted(blocks.items(), key=lambda kv: int(kv[0].split(".")[1]))),
        "total_tensor_bytes": total,
        "token_embd_bytes": embd,
        "token_embd_shape": embd_t["shape"],
        "output_weight_present": any(t["name"] == "output.weight" for t in tensors),
        "head_fraction_of_tensor_bytes": embd / total,
        "vocab_size": vocab,
        "embedding_width": embd_t["shape"][0],
        "block_count_field": nblk,
        "attention_blocks": [g for g, b in blocks.items() if b["kind"] == "attention"],
        "recurrent_blocks": [g for g, b in blocks.items() if b["kind"] == "recurrent"],
        "other_blocks": [g for g, b in blocks.items() if b["kind"] in ("other", "mtp")],
    }
    det["attention_blocks"].sort(key=lambda g: int(g.split(".")[1]))
    det["recurrent_blocks"].sort(key=lambda g: int(g.split(".")[1]))
    mtps = [g for g, b in blocks.items() if b["kind"] == "mtp"]
    mtp = mtps[0] if len(mtps) == 1 else None
    det["mtp_block"] = mtp
    det["nextn_predict_layers_field"] = fields.get(f"{arch}.nextn_predict_layers")
    det["mtp_block_bytes"] = groups[mtp]["n_bytes"] if mtp else None
    # A plain (non-speculative) token runs every block except the MTP one and reads the tied head once.
    det["bytes_used_by_a_plain_token"] = total - (det["mtp_block_bytes"] or 0)

    print("metadata keys:")
    for k, v in fields.items():
        print(f"  {k} = {v}")
    print(f"\n{len(tensors)} tensors:")
    for t in tensors:
        print(f"  {t['name']:40s} {str(t['shape']):22s} {t['type']:5s} {t['n_bytes']:>12,}")
    print("\nby group:")
    for g, v in det["groups"].items():
        kind = blocks.get(g, {}).get("kind", "")
        print(f"  {g:12s} {v['n_tensors']:3d} tensors {v['n_bytes']:>13,} bytes {v['n_bytes'] / MIB:8.2f} MiB  {kind}")
    print(f"\nfile {file_size:,} bytes; tensor data starts at byte {det['tensor_data_offset']:,}")
    print(f"total tensor bytes {total:,} ({total / MIB:.1f} MiB); token_embd {embd:,} ({embd / MIB:.1f} MiB) "
          f"= {100 * det['head_fraction_of_tensor_bytes']:.2f} % of tensor bytes; output.weight present: "
          f"{det['output_weight_present']}")
    print(f"vocab {vocab:,}, width {det['embedding_width']}; attention blocks {det['attention_blocks']}")
    print(f"recurrent blocks: {len(det['recurrent_blocks'])}; other: {det['other_blocks']}; "
          f"MTP block {mtp}: {det['mtp_block_bytes']:,} bytes")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_gguf.py", "started_utc": started, "model": str(Path(a.model).name),
           "deterministic": det}
    (out / "gguf.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
