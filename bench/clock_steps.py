"""WSL2 clock behaviour (Phase 0.6 follow-up; tests predictions P-004 and P-005).

Two measurements over the same window:
1. Wall-clock steps. Samples CLOCK_REALTIME and CLOCK_MONOTONIC about every 1 ms. Between steps the
   two advance together, so a jump of more than 1 ms in (REALTIME - MONOTONIC) is a step of the wall
   clock. Each reading brackets REALTIME between two MONOTONIC reads and is dropped if the thread
   was descheduled in between (gap > 100 us), so scheduling hiccups don't look like steps.
2. Clock rate against Windows. powershell.exe prints the host's UTC ticks every 10 s; each line is
   stamped with the guest's MONOTONIC time on arrival. Comparing elapsed times gives the rate.

  python bench/clock_steps.py [--seconds 120] [--out results/0/clock-steps.json]
"""
import argparse, json, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ns = time.clock_gettime_ns
RT, MONO, RAW = time.CLOCK_REALTIME, time.CLOCK_MONOTONIC, time.CLOCK_MONOTONIC_RAW


def iso(t_ns):
    return datetime.fromtimestamp(t_ns / 1e9, timezone.utc).isoformat(timespec="milliseconds")


def sample_steps(stop, steps, stats):
    prev, kept, dropped = None, 0, 0
    while not stop.is_set():
        m1, r, m2 = ns(MONO), ns(RT), ns(MONO)
        if m2 - m1 > 100_000:
            dropped += 1
        else:
            off = r - (m1 + m2) // 2
            if prev is not None and abs(off - prev) > 1_000_000:
                steps.append({"guest_monotonic_s": m1 / 1e9, "realtime_after_step": iso(r),
                              "step_ms": (off - prev) / 1e6})
            prev = off
            kept += 1
        time.sleep(0.001)
    stats.update(samples_kept=kept, samples_dropped=dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "results/0/clock-steps.json"))
    a = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_lines = a.seconds // 10 + 1
    ps = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command",
         f"for ($i = 0; $i -lt {n_lines}; $i++) {{ [DateTime]::UtcNow.Ticks; [Console]::Out.Flush(); "
         f"if ($i -lt {n_lines - 1}) {{ Start-Sleep -Milliseconds 10000 }} }}"],
        stdout=subprocess.PIPE, text=True)
    stop, steps, stats = threading.Event(), [], {}
    th = threading.Thread(target=sample_steps, args=(stop, steps, stats))
    start = {"mono": ns(MONO), "raw": ns(RAW), "rt": ns(RT)}
    th.start()
    host = []
    for line in ps.stdout:                      # one line per 10 s, stamped on arrival
        m, r = ns(MONO), ns(RT)
        if line.strip().isdigit():
            ticks = int(line.strip())           # 100 ns units since 0001-01-01 UTC
            host.append({"host_ticks": ticks, "guest_monotonic_ns": m, "guest_realtime_ns": r,
                         "guest_realtime_minus_host_ms": (r - (ticks - 621355968000000000) * 100) / 1e6})
    ps.wait()
    stop.set()
    th.join()
    end = {"mono": ns(MONO), "raw": ns(RAW), "rt": ns(RT)}

    host_s = (host[-1]["host_ticks"] - host[0]["host_ticks"]) / 1e7
    guest_s = (host[-1]["guest_monotonic_ns"] - host[0]["guest_monotonic_ns"]) / 1e9
    gaps = [round(b["guest_monotonic_s"] - a_["guest_monotonic_s"], 3) for a_, b in zip(steps, steps[1:])]
    summary = {
        "window_guest_monotonic_s": (end["mono"] - start["mono"]) / 1e9,
        "n_steps": len(steps),
        "step_ms": [round(s["step_ms"], 3) for s in steps],
        "seconds_between_steps": gaps,
        "host_readings": len(host),
        "host_elapsed_s": host_s,
        "guest_monotonic_elapsed_s": guest_s,
        "guest_minus_host_rate_pct": 100 * (guest_s / host_s - 1),
        "monotonic_minus_raw_over_window_us": ((end["mono"] - start["mono"]) - (end["raw"] - start["raw"])) / 1e3,
        **stats,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"script": "bench/clock_steps.py", "started_utc": started, "seconds": a.seconds,
           "summary": summary, "steps": steps, "host": host}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
