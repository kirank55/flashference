"""Kernel clock adjustment over time (Phase 0 review fix; tests prediction P-006).

Samples adjtimex(2) read-only every --interval seconds: tick (µs per 10 ms USER_HZ tick; nominal
10000, above it CLOCK_MONOTONIC and the wall clock run fast) and freq (ppm).

  python bench/tick_watch.py [--seconds 120] [--interval 0.5] [--out results/0/tick-watch.json]
"""
import argparse, ctypes, ctypes.util, json, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Timex(ctypes.Structure):     # struct timex up to tick; modes = 0 means read only
    _fields_ = [("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
                ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
                ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
                ("time", ctypes.c_long * 2), ("tick", ctypes.c_long), ("pad", ctypes.c_long * 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results/0/tick-watch.json"))
    a = ap.parse_args()
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    t0, rows = time.monotonic(), []
    while time.monotonic() - t0 < a.seconds:
        t = Timex()
        libc.adjtimex(ctypes.byref(t))
        rows.append({"t_s": round(time.monotonic() - t0, 3), "tick_us": t.tick, "freq_ppm": round(t.freq / 65536, 3)})
        time.sleep(a.interval)
    ticks = [r["tick_us"] for r in rows]
    summary = {"samples": len(rows), "tick_min": min(ticks), "tick_max": max(ticks),
               "tick_changes": sum(x != y for x, y in zip(ticks, ticks[1:])),
               "rate_from_tick_pct_min": 100 * (min(ticks) / 10000 - 1),
               "rate_from_tick_pct_max": 100 * (max(ticks) / 10000 - 1),
               "freq_ppm_min": min(r["freq_ppm"] for r in rows), "freq_ppm_max": max(r["freq_ppm"] for r in rows)}
    out = Path(a.out)
    out.write_text(json.dumps({"script": "bench/tick_watch.py", "started_utc": started, "summary": summary,
                               "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
