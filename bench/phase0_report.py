"""Phase 0.6 table: one row per llama-bench run, from results/0/llama-bench-*.json.

  python bench/phase0_report.py            markdown table (used inside NOTEBOOK.md gen markers)
  python bench/phase0_report.py --summary  the derived numbers that predictions P-001..P-003 test
"""
import json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
runs = []
for f in sorted((ROOT / "results/0").glob("llama-bench-[0-9]*.json")):
    r = json.loads(f.read_text())[0]
    runs.append((f.name, r))

if "--summary" in sys.argv:
    means = [r["avg_ts"] for _, r in runs]
    print(f"mean of run means: {statistics.mean(means):.2f} tok/s")
    print(f"between-run spread (max/min - 1): {100 * (max(means) / min(means) - 1):.2f} %")
    print(f"largest within-run SD / mean: {max(100 * r['stddev_ts'] / r['avg_ts'] for _, r in runs):.2f} %")
else:
    print("| run | test_time (UTC) | tok/s mean | SD | SD % of mean | slowest .. fastest repetition |")
    print("|---|---|---:|---:|---:|---|")
    for name, r in runs:
        s = r["samples_ts"]
        print(f"| {name} | {r['test_time']} | {r['avg_ts']:.2f} | {r['stddev_ts']:.2f} | "
              f"{100 * r['stddev_ts'] / r['avg_ts']:.2f} | {min(s):.2f} .. {max(s):.2f} |")
