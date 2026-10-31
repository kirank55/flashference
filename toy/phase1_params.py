"""Phase 1.3: where are the parameters? (plan.md 1.3)

Sums parameter counts and bytes by group (embedding, attention, MLP, norms, lm_head) for the
SmolLM2 target (360M) and draft (135M), in the dtype the checkpoints are stored in (bf16).

named_parameters() skips a tensor it has already listed, so with tied embeddings lm_head.weight
never shows up on its own. remove_duplicate=False lists it, and comparing storage pointers proves
it is the same matrix as embed_tokens.weight.

  python toy/phase1_params.py [--out results/1]

Writes <out>/params.json. Everything under "deterministic" must be identical on a re-run.
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct"}
GROUPS = ("embedding", "attention", "mlp", "norms", "lm_head")


def group_of(name):
    if "embed_tokens" in name:
        return "embedding"
    if "self_attn" in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "norms"
    if name.startswith("lm_head"):
        return "lm_head"
    raise ValueError(f"no group for {name}")


def count(model):
    params = {g: 0 for g in GROUPS}
    nbytes = {g: 0 for g in GROUPS}
    names = {g: [] for g in GROUPS}
    for name, p in model.named_parameters():          # each distinct tensor once
        g = group_of(name)
        params[g] += p.numel()
        nbytes[g] += p.numel() * p.element_size()
        if len(names[g]) < 4:
            names[g].append(name)
    all_names = [n for n, _ in model.named_parameters(remove_duplicate=False)]
    head, emb = model.get_output_embeddings().weight, model.get_input_embeddings().weight
    total_bytes = sum(nbytes.values())
    return {
        "dtype": str(next(model.parameters()).dtype),
        "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "lm_head_listed_by_named_parameters": any(n.startswith("lm_head") for n, _ in model.named_parameters()),
        "lm_head_listed_with_remove_duplicate_false": any(n.startswith("lm_head") for n in all_names),
        "lm_head_is_embed_tokens": head.data_ptr() == emb.data_ptr() and head.shape == emb.shape,
        "embedding_shape": list(emb.shape),
        "params": params, "bytes": nbytes, "example_names": names,
        "total_params": sum(params.values()), "total_bytes": total_bytes,
        "embedding_head_byte_fraction": (nbytes["embedding"] + nbytes["lm_head"]) / total_bytes,
        "n_layers": model.config.num_hidden_layers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    det = {}
    for key, repo in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16)   # the stored dtype
        det[key] = count(model)
        r = det[key]
        print(f"\n== {key} ({r['dtype']}, {r['n_layers']} layers): {r['total_params']:,} params, {r['total_bytes']:,} bytes")
        for g in GROUPS:
            print(f"  {g:9s} {r['params'][g]:>12,} params {r['bytes'][g]:>13,} bytes "
                  f"{100 * r['bytes'][g] / r['total_bytes']:6.2f} %")
        print(f"  tie_word_embeddings={r['config_tie_word_embeddings']}; lm_head is embed_tokens: "
              f"{r['lm_head_is_embed_tokens']}; embedding+head share of bytes: {100 * r['embedding_head_byte_fraction']:.2f} %")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_params.py", "started_utc": started, "deterministic": det}
    (out / "params.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
