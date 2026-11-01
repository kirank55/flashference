"""Phase 1 report: markdown tables from the JSON files in --dir (default results/1).

The executor embeds this output in NOTEBOOK.md between `<!-- gen: python toy/phase1_report.py --X -->`
markers, so the output must be byte-identical for the same input: fixed formatting, sorted keys,
and no times or paths.

  python toy/phase1_report.py --dir results/1 [--tokens] [--greedy] [--params] [--gguf] [--sample]

With no --X option it prints every table that has a file.
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ("embedding", "attention", "mlp", "norms", "lm_head", "other")


def load(d, name):
    p = Path(d) / name
    return json.loads(p.read_text()) if p.exists() else None


def cell(x):
    return str(x).replace("|", "\\|").replace("\n", "\\n")


def table(headers, rows, right=()):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(("---:" if i in right else "---") for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def fi(n):
    return "—" if n is None else f"{n:,}"


def ff(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def fp(x):
    return "—" if x is None else f"{x:.2f}%"


def show_tokens(d):
    doc = load(d, "tokens.json")
    if not doc:
        return ""
    det = doc["deterministic"]
    rows = []
    for text in sorted(det["texts"]):
        s = det["results"]["smollm2-360m"][text]
        q = det["results"]["qwen3.5-0.8b"][text]
        rows.append([text, s["n_tokens"], q["n_tokens"], ff(s["n_tokens"] / q["n_tokens"], 2),
                     ff(s["n_chars"] / s["n_tokens"], 2), ff(q["n_chars"] / q["n_tokens"], 2),
                     s["pieces_with_partial_chars"], q["pieces_with_partial_chars"],
                     "yes" if s["decodes_back_exactly"] else "no",
                     "yes" if q["decodes_back_exactly"] else "no"])
    vs = det["vocab_size"]
    return "\n\n".join([
        table(["text", "SmolLM2 tokens", "Qwen3.5 tokens", "ratio S/Q", "chars/token S", "chars/token Q",
               "partial S", "partial Q", "exact S", "exact Q"], rows, right={1, 2, 3, 4, 5, 6, 7}),
        f"vocabulary (len(tok)): SmolLM2-360M {fi(vs['smollm2-360m'])}, SmolLM2-135M {fi(vs['smollm2-135m'])}, "
        f"Qwen3.5-0.8B {fi(vs['qwen3.5-0.8b'])}; SmolLM2-135M ids equal SmolLM2-360M ids for every text: "
        + ("yes" if det["smollm2_135m_ids_equal_360m"] else "no")])


def show_greedy(d):
    blocks = []
    for dev in ("cpu", "cuda"):
        doc = load(d, f"greedy-{dev}.json")
        if not doc:
            continue
        det, meas = doc["deterministic"], doc.get("measured", {})
        med = meas.get("median_s", {})
        rows = [[m, ff(med.get(m)), ff(meas.get("min_s", {}).get(m)),
                 ff(meas.get("max_s", {}).get(m)), ff(meas.get("median_ms_per_token", {}).get(m), 1)]
                for m in ("nocache", "cached", "generate")]
        rc = med.get("nocache") / med["cached"] if med.get("cached") else None
        rg = med.get("generate") / med["cached"] if med.get("cached") else None
        # per-step medians across repeats: step 1 runs the whole prompt in both loops; after that,
        # nocache step k runs prompt + k - 1 positions while cached runs one
        st = meas.get("step_seconds", {})
        n_prompt = len(det["prompt_ids"])
        med_step = {m: [sorted(x)[len(x) // 2] for x in zip(*st[m])] for m in ("nocache", "cached") if st.get(m)}
        step_rows = []
        if len(med_step) == 2:
            n_steps = min(len(med_step["nocache"]), len(med_step["cached"]))
            for k in sorted({1, 2, n_steps}):
                step_rows.append([k, fi(n_prompt + k - 1), ff(med_step["nocache"][k - 1], 4),
                                  fi(n_prompt if k == 1 else 1), ff(med_step["cached"][k - 1], 4)])
        blocks.append("\n\n".join([
            f"**greedy-{dev}**",
            table(["method", "median s", "min s", "max s", "ms/token"], rows, right={1, 2, 3, 4}),
            table(["step", "positions run (nocache)", "nocache median s", "positions run (cached)", "cached median s"],
                  step_rows, right={0, 1, 2, 3, 4}),
            table(["metric", "value"], [["prompt tokens", fi(len(det["prompt_ids"]))],
                                        ["tokens generated", fi(det["n_generated"])],
                                        ["timed repeats per method", fi(meas.get("repeats"))],
                                        ["clock", meas.get("clock", "—")],
                                        ["nocache/cached", ff(rc, 2)], ["generate/cached", ff(rg, 2)],
                                        ["all three ids equal", "yes" if det["all_three_equal"] else "no"]],
                  right={1}),
        ]))
    return "\n\n".join(blocks)


def show_params(d):
    doc = load(d, "params.json")
    if not doc:
        return ""
    det = doc["deterministic"]
    rows = []
    for model in sorted(det):
        r = det[model]
        rows.append([model, *[fi(r["bytes"].get(g, 0)) for g in GROUPS], fi(r["total_bytes"]),
                     fi(r["total_params"]), fp(r["embedding_share_percent"]),
                     "yes" if r["config_tie_word_embeddings"] else "no",
                     "yes" if r["lm_head_same_storage_as_embed_tokens"] else "no"])
    return table(["model", *GROUPS, "total bytes", "total params", "embedding share", "tied", "same storage"],
                 rows, right=set(range(1, 10)))


def show_gguf(d):
    doc = load(d, "gguf.json")
    if not doc:
        return ""
    det = doc["deterministic"]
    summary = [
        ["file bytes", fi(det["file_bytes"])],
        ["tensor bytes", fi(det["total_tensor_bytes"])],
        ["parameters", fi(det["total_params"])],
        ["tensors", fi(det["n_tensors"])],
        ["token_embd bytes", fi(det["token_embd"]["n_bytes"])],
        ["token_embd share", fp(100 * det["token_embd"]["share_of_tensor_bytes"])],
        ["separate output.weight", "yes" if det["output_weight_present"] else "no"],
        ["MTP bytes", fi(det["mtp_bytes"])],
        ["bytes not in tensors", fi(det["bytes_not_in_tensors"])],
    ]
    kinds = []
    for kind in sorted(det["block_kinds"]):
        k = det["block_kinds"][kind]
        per = ", ".join(f"{n}: {fi(k['bytes'][str(n)])}" for n in sorted(k["blocks"]))
        kinds.append([kind, ", ".join(str(n) for n in sorted(k["blocks"])), per, fi(k["total_bytes"])])
    types = [[t, fi(s["n_tensors"]), fi(s["n_bytes"])] for t, s in sorted(det["tensor_types"].items())]
    return "\n\n".join([
        "**summary**", table(["item", "value"], summary, right={1}),
        "**blocks by kind**", table(["kind", "blocks", "bytes per block", "total bytes"], kinds, right={3}),
        "**bytes by tensor type**", table(["type", "tensors", "bytes"], types, right={1, 2}),
    ])


def show_sample(d):
    doc = load(d, "sample.json")
    if not doc:
        return ""
    det = doc["deterministic"]
    toy, model = det["toy"], det["model"]
    rows = [[int(i), ff(p, 6), ff(p * toy["draws"], 2), toy["observed_counts"][int(i)],
             ff(toy["observed_counts"][int(i)] / toy["draws"], 6)]
            for i, p in zip(toy["nucleus_ids"], toy["nucleus_probs"])]
    seed_rows = [[s["seed"], "yes" if s["differs_from_greedy"] else "no",
                  "—" if s["first_position_differing_from_greedy"] is None else s["first_position_differing_from_greedy"],
                  s["text"]] for s in model["samples"]]
    return "\n\n".join([
        "**toy**",
        table(["token id", "probability", "expected draws", "observed draws", "observed freq"], rows,
              right={0, 1, 2, 3, 4}),
        f"chi-square p = {ff(toy['chi_square_p_value'], 6)}; draws outside nucleus = {toy['draws_outside_nucleus']}; "
        f"sampler's nucleus equals the reference: {'yes' if toy.get('sampler_nucleus_matches_reference') else 'no'}",
        "**model**",
        f"greedy text: {cell(model['greedy_text'])}",
        table(["seed", "differs from greedy", "first differing position", "text"], seed_rows, right={0, 2}),
        f"distinct samples = {model['distinct_samples']}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results/1"))
    for name in ("tokens", "greedy", "params", "gguf", "sample"):
        ap.add_argument(f"--{name}", action="store_true")
    a = ap.parse_args()
    wanted = [n for n in ("tokens", "greedy", "params", "gguf", "sample") if getattr(a, n)]
    if not wanted:
        wanted = ["tokens", "greedy", "params", "gguf", "sample"]
    fns = {"tokens": show_tokens, "greedy": show_greedy, "params": show_params,
           "gguf": show_gguf, "sample": show_sample}
    out = [s for n in wanted if (s := fns[n](a.dir))]
    if out:
        print("\n\n".join(out))


if __name__ == "__main__":
    main()
