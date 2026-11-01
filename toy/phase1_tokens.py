"""Phase 1.1: token round-trip (plan.md 1.1).

An LLM never sees text. A *tokenizer* cuts text into vocabulary pieces ("tokens") and gives
each piece an integer id; the model predicts the next id. The same text can need very different
numbers of tokens in different vocabularies, and a single id can hold only part of one UTF-8
character (then decoding that id alone shows the replacement character U+FFFD).

This script tokenizes six short texts with three tokenizers: SmolLM2's (shared by the 135M
draft and the 360M target of Phases 1-3; 49,152 tokens) and Qwen3.5-0.8B's (the real target of
Phases 4-9; about 248k tokens). For every text it records the ids, what each id decodes to on
its own, the raw vocabulary entries, and whether decode(ids) gives the original text back.

  python toy/phase1_tokens.py [--out results/1]

Writes <out>/tokens.json. Everything under "deterministic" must be identical on a re-run.
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TOKENIZERS = {
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
}
TEXTS = {
    "english": "The quick brown fox jumps over the lazy dog.",
    "code": "def greedy(model, ids, n):\n    for _ in range(n):\n        ids.append(model(ids).argmax())",
    "emoji": "Speculative decoding is fast 🚀🙂👍🏽",
    "hindi": "भाषा मॉडल एक समय में एक टोकन बनाता है।",
    "chinese": "语言模型一次生成一个词元。",
    "german": "Sprachmodelle erzeugen ein Token nach dem anderen.",
}


def round_trip(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]          # each id decoded on its own
    return {
        "ids": ids,
        "pieces": pieces,                             # what each single id decodes to
        "tokens": tok.convert_ids_to_tokens(ids),     # the raw vocabulary entries
        "decodes_back_exactly": tok.decode(ids) == text,
        "n_tokens": len(ids),
        "n_chars": len(text),
        "n_utf8_bytes": len(text.encode("utf-8")),
        # a lone id that decodes to U+FFFD holds only part of a character's UTF-8 bytes
        "pieces_with_partial_chars": sum("\ufffd" in p for p in pieces),
    }


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/1"))
    a = ap.parse_args()

    det = {"texts": TEXTS, "vocab_size": {}, "results": {}}
    for name, repo in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        det["vocab_size"][name] = len(tok)
        det["results"][name] = {k: round_trip(tok, t) for k, t in TEXTS.items()}
    small = det["results"]
    det["smollm2_135m_ids_equal_360m"] = all(
        small["smollm2-135m"][k]["ids"] == small["smollm2-360m"][k]["ids"] for k in TEXTS)

    print(f"vocab sizes: {det['vocab_size']}; 135M ids == 360M ids: {det['smollm2_135m_ids_equal_360m']}")
    for k, text in TEXTS.items():
        print(f"\n== {k}: {text!r} ({len(text)} chars, {len(text.encode())} UTF-8 bytes)")
        for name in ("smollm2-360m", "qwen3.5-0.8b"):
            r = det["results"][name][k]
            print(f"  {name:13s} {r['n_tokens']:3d} tokens, {r['n_chars'] / r['n_tokens']:.2f} chars/token, "
                  f"{r['pieces_with_partial_chars']} partial-character pieces, exact: {r['decodes_back_exactly']}")
            print("   ", " ".join(f"{i}:{p!r}" for i, p in zip(r["ids"], r["pieces"])))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"script": "toy/phase1_tokens.py", "started_utc": started, "deterministic": det}
    (out / "tokens.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
