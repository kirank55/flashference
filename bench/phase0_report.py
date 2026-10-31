"""Phase 0.6 tables, from results/0/llama-bench-*.json (and the clock evidence in results/0/).

  python bench/phase0_report.py            one row per llama-bench run, raw and screened
  python bench/phase0_report.py --summary  the derived numbers that predictions P-001..P-003 test
  python bench/phase0_report.py --clock    the clock measurement that P-004 and P-005 test

Screening rule. It was written after the Phase 0 outlier had been seen, so for the Phase 0 data
it is post hoc; from Phase 0 on it applies to every llama-bench run as stated here, before the data
are seen. llama-bench times each repetition with the wall clock
(std::chrono::high_resolution_clock is system_clock in libstdc++), and this WSL2 wall clock is
stepped back ~0.6 s every ~32 s (results/0/clock-steps.json). A repetition is excluded only if
(a) it is more than 0.3 s shorter than its run's median repetition (half the smallest step measured)
and (b) the journal records a clock change inside that run's time window
(results/0/timesync-snapshot.txt). Raw numbers are always shown next to screened ones.
"""
import json, re, statistics, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R0 = ROOT / "results/0"


def clock_changes():
    text = (R0 / "timesync-snapshot.txt").read_text()
    out = []
    for m in re.finditer(r"^(\w{3} \d+ \d\d:\d\d:\d\d\.\d+) \S+ systemd-resolved\[\d+\]: Clock change detected", text, re.M):
        out.append(datetime.strptime("2026 " + m.group(1), "%Y %b %d %H:%M:%S.%f").replace(tzinfo=timezone.utc))
    return out


def runs():
    steps = clock_changes()
    rows = []
    for f in sorted(R0.glob("llama-bench-[0-9]*.json")):
        r = json.loads(f.read_text())[0]
        ns = r["samples_ns"]
        start = datetime.strptime(r["test_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        end = start + timedelta(seconds=1 + sum(ns) / 1e9 + 0.5)    # test_time is truncated to the second
        in_window = [s for s in steps if start <= s <= end]
        med = statistics.median(ns)
        drop = [i for i, x in enumerate(ns) if x < med - 0.3e9] if in_window else []
        keep = [x for i, x in enumerate(ns) if i not in drop]
        ts_all = [r["n_gen"] / (x / 1e9) for x in ns]
        ts_keep = [r["n_gen"] / (x / 1e9) for x in keep]
        rows.append({"file": f.name, "test_time": r["test_time"], "ts_all": ts_all, "ts_keep": ts_keep,
                     "steps_in_window": in_window, "dropped": drop})
    return rows


def pct_sd(v):
    return 100 * statistics.stdev(v) / statistics.mean(v)


def table():
    print("| run | test_time (UTC) | tok/s, all 5 reps | SD %, all 5 | clock changes in run window | "
          "excluded rep (tok/s) | tok/s, kept reps | SD %, kept |")
    print("|---|---|---:|---:|---|---|---:|---:|")
    for x in runs():
        ch = ", ".join(s.strftime("%H:%M:%S.%f")[:-3] for s in x["steps_in_window"]) or "none"
        ex = ", ".join(f"#{i + 1} ({x['ts_all'][i]:.2f})" for i in x["dropped"]) or "none"
        print(f"| {x['file']} | {x['test_time']} | {statistics.mean(x['ts_all']):.2f} | {pct_sd(x['ts_all']):.2f} | "
              f"{ch} | {ex} | {statistics.mean(x['ts_keep']):.2f} | {pct_sd(x['ts_keep']):.2f} |")


def summary():
    rs = runs()
    for label, key in (("raw, all 15 repetitions", "ts_all"), ("screened", "ts_keep")):
        means = [statistics.mean(x[key]) for x in rs]
        print(f"- {label}: mean of run means {statistics.mean(means):.2f} tok/s; between-run spread "
              f"(max/min − 1) {100 * (max(means) / min(means) - 1):.2f} %; largest within-run SD "
              f"{max(pct_sd(x[key]) for x in rs):.2f} % of the run mean")
    print(f"- repetitions excluded: {sum(len(x['dropped']) for x in rs)} of {sum(len(x['ts_all']) for x in rs)}")


def clock():
    doc = json.loads((R0 / "clock-steps.json").read_text())
    s = doc["summary"]
    t = [x["guest_monotonic_s"] for x in doc["steps"]]
    gaps = [b - a for a, b in zip(t, t[1:])]                # from the raw stamps: round once
    print("| quantity | value |")
    print("|---|---:|")
    print(f"| window (guest CLOCK_MONOTONIC) | {s['window_guest_monotonic_s']:.1f} s |")
    print(f"| wall-clock steps | {s['n_steps']} |")
    print(f"| step sizes | {', '.join(f'{v / 1000:+.3f} s' for v in s['step_ms'])} |")
    print(f"| time between steps (guest monotonic) | {', '.join(f'{v:.2f} s' for v in gaps)} |")
    print(f"| host elapsed / guest monotonic elapsed | {s['host_elapsed_s']:.3f} s / {s['guest_monotonic_elapsed_s']:.3f} s |")
    print(f"| guest CLOCK_MONOTONIC rate vs Windows clock | {s['guest_minus_host_rate_pct']:+.2f} % |")
    raw = s['window_guest_monotonic_s'] - s['monotonic_minus_raw_over_window_us'] / 1e6
    print(f"| CLOCK_MONOTONIC gain over CLOCK_MONOTONIC_RAW in the window | "
          f"{s['monotonic_minus_raw_over_window_us'] / 1e6:.3f} s ({100 * (s['window_guest_monotonic_s'] / raw - 1):+.2f} %) |")


if __name__ == "__main__":
    {"--summary": summary, "--clock": clock}.get(sys.argv[1] if len(sys.argv) > 1 else "", table)()
