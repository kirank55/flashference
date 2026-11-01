"""Phase 1.3: where are the parameters? (plan.md 1.3)

A model's speed is mostly a question of how many bytes it must read per token. This script
loads the SmolLM2 target (360M) and draft (135M) in bf16 (2 bytes per parameter, the stored
dtype) on the CPU and sums parameter counts and bytes by group: the input embedding table,
the attention blocks, the MLP blocks, the layer norms, and the output LM head.

SmolLM2 ties its embeddings: the input table and the LM head are the *same* matrix. PyTorch's
named_parameters() lists a shared tensor only once, so lm_head.weight shows up with 0 bytes of
its own; data_ptr equality proves the two module attributes point at one storage.

  python toy/phase1_params.py [--out results/1]

Writes <out>/params.json. Everything under "deterministic" must be identical on a re-run.
"""
import argparse, gc, json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct"}
GROUPS = ("embedding", "attention", "mlp", "norms", "lm_head", "other")


def group_of(name):
    """Map a parameter name to one of the five real groups, or to "other" (which must stay empty)."""
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
    return "other"


def count(model):
    params = {g: 0 for g in GROUPS}
    nbytes = {g: 0 for g in GROUPS}
    names = {g: [] for g in GROUPS}
    for name, p in model.named_parameters():          # each distinct tensor once
        g = group_of(name)
        params[g] += p.numel()
        nbytes[g] += p.numel() * p.element_size()
        names[g].append(name)
    assert params["other"] == 0, f"ungrouped parameters: {names['other']}"

    all_names = [n for n, _ in model.named_parameters(remove_duplicate=False)]
    head, emb = model.get_output_embeddings().weight, model.get_input_embeddings().weight
    total_bytes = sum(nbytes.values())
    return {
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "lm_head_listed_by_named_parameters": any(n.startswith("lm_head") for n, _ in model.named_parameters()),
        "lm_head_listed_with_remove_duplicate_false": any(n.startswith("lm_head") for n in all_names),
        "lm_head_same_storage_as_embed_tokens": head.data_ptr() == emb.data_ptr() and head.shape == emb.shape,
        "embedding_shape": list(emb.shape),
        "params": params, "bytes": nbytes, "example_names": {g: names[g][:4] for g in GROUPS},
        "total_params": sum(params.values()), "total_bytes": total_bytes,
        "embedding_share_percent": 100 * nbytes["embedding"] / total_bytes,
        "embedding_plus_lm_head_share_percent": 100 * (nbytes["embedding"] + nbytes["lm_head"]) / total_bytes,
        "n_layers": model.config.num_hidden_layers,
        "note": "named_parameters() lists a tied tensor once, so lm_head shows 0 bytes of its own.",
    }


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()

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
              f"{r['lm_head_same_storage_as_embed_tokens']}; embedding share: {r['embedding_share_percent']:.2f} %")
        del model
        gc.collect()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_params.py", "started_utc": started, "deterministic": det}
    (out / "params.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
