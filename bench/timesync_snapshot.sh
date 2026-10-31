#!/usr/bin/env bash
# Raw evidence for the WSL2 clock problem found in Phase 0.6: who adjusts the clock, and when.
#   bench/timesync_snapshot.sh "<journal since>" "<journal until>" > results/0/timesync-snapshot.txt
set -u
since=${1:?since}; until=${2:?until}
echo "# timesync snapshot $(date -u +%FT%TZ)"
echo; echo "## systemd-resolved clock-change events, $since .. $until (journal, realtime clock)"
journalctl -o short-precise --since "$since" --until "$until" 2>/dev/null | grep "Clock change detected"
echo; echo "## systemd-timesyncd log"
journalctl -o short-precise -u systemd-timesyncd 2>/dev/null | tail -12
echo; echo "## timedatectl timesync-status"
timedatectl timesync-status 2>&1
echo; echo "## kernel clock adjustment (adjtimex): tick in us per 10 ms USER_HZ tick; freq in ppm"
python3 - <<'PY'
import ctypes, ctypes.util
class tv(ctypes.Structure): _fields_ = [("s", ctypes.c_long), ("us", ctypes.c_long)]
class timex(ctypes.Structure):
    _fields_ = [("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
                ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
                ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
                ("time", tv), ("tick", ctypes.c_long), ("pad", ctypes.c_long * 20)]
t = timex(); ctypes.CDLL(ctypes.util.find_library("c")).adjtimex(ctypes.byref(t))
print(f"tick={t.tick} (nominal 10000: clock rate {100 * (t.tick / 10000 - 1):+.3f} %) freq={t.freq / 65536:+.3f} ppm status={t.status:#x}")
PY
