"""Phase 1.1: token round-trip.

Tokenize a few strings, print the ids, then decode each id on its own.
Run from the repo root:  .venv/bin/python toy/phase1_tokens.py
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT = Path("results/phase1/tokens.json")

SAMPLES = {
    "english": "The quick brown fox jumps over the lazy dog.",
    "code": "def add(a, b):\n    return a + b  # sum",
    "emoji": "I love pizza 🍕 and rockets 🚀!",
    "hindi": "मुझे हिंदी पढ़ना पसंद है।",
}

tok = AutoTokenizer.from_pretrained(MODEL)
print(f"tokenizer: {MODEL}  vocab size: {len(tok)}\n")

rows = {}
for name, text in SAMPLES.items():
    ids = tok.encode(text, add_special_tokens=False)
    # Decoding one id at a time: a multi-byte character split across ids shows as '�'.
    pieces = [tok.decode([i]) for i in ids]
    roundtrip = tok.decode(ids) == text
    rows[name] = {
        "text": text,
        "n_chars": len(text),
        "n_utf8_bytes": len(text.encode("utf-8")),
        "n_tokens": len(ids),
        "ids": ids,
        "pieces": pieces,
        "roundtrip_ok": roundtrip,
    }
    print(f"== {name}: {text!r}")
    print(f"   ids ({len(ids)}): {ids}")
    for i, p in zip(ids, pieces):
        print(f"   {i:>6}  {p!r}")
    print(f"   chars={len(text)} utf8_bytes={len(text.encode('utf-8'))} "
          f"tokens={len(ids)} chars/token={len(text) / len(ids):.2f} roundtrip={roundtrip}\n")

print("Note on uneven splits:")
for name, r in rows.items():
    bad = sum("�" in p for p in r["pieces"])
    print(f"  {name:8s} {r['n_chars']:3d} chars -> {r['n_tokens']:3d} tokens "
          f"({r['n_chars'] / r['n_tokens']:.2f} chars/token); "
          f"{bad} ids decode to a partial character on their own")
print("  English words are often one token each; Hindi and emoji fall back to byte-level"
      " pieces, so the same meaning costs many more tokens (= more decode steps).")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"tokenizer": MODEL, "vocab_size": len(tok), "samples": rows},
                          ensure_ascii=False, indent=2))
print(f"\nwrote {OUT}")
