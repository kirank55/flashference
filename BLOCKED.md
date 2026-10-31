# BLOCKED

## 2026-10-31 WSL2 clock runs ~1.9 % fast and is stepped back every 32 s
severity: soft
what: inside WSL the kernel clock tick is 10188–10192 µs instead of the nominal 10000, so CLOCK_MONOTONIC and the wall clock gain ~1.9 %. systemd-timesyncd polls every 32 s, finds the clock ~0.6 s ahead and steps it back. One llama-bench repetition was cut 0.622 s short by a step. Evidence: NOTEBOOK.md "Phase 0.5–0.6", results/0/clock-steps.json, results/0/timesync-snapshot.txt.
tried: measured it, and worked around it: same-session ratios, CUDA events for absolute GPU timing, a screening rule for llama-bench, and the tick recorded at every preflight (STATE.md Decisions).
owner action: none required (for information). Optional: to find out whether a fresh WSL boot has a normal clock, run `wsl --shutdown` in Windows, reopen WSL, then run `timedatectl timesync-status` in the repo. The Offset should be a few ms, not about −600 ms. Also run `bench/timesync_snapshot.sh "10 min ago" now`; it should show tick=10000. If the problem comes back, `sudo systemctl disable --now systemd-timesyncd` stops the stepping (WSL2 takes its time from Windows), but that doesn't fix the tick.
meanwhile: work continues. Ratios are unaffected. Absolute rates carry a stated ~1.9 % correction.
