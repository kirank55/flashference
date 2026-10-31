#!/usr/bin/env bash
# preflight.sh: environment and GPU checks (autoplan.md §4.2).
#
#   autopilot/preflight.sh            session start: all checks, including the dirty-tree check
#   autopilot/preflight.sh --gpu      just before a GPU run: skips the dirty-tree check
#   add --wait                        on 10/20, re-check every 10 min for up to 1 h
#   FF_LOCK_HELD=1 in the environment  the caller holds the GPU lock for this run (gpu-run.sh, job.sh)
#
# Prints one JSON line of readings to stdout. Exit codes:
#   0 OK | 11 our own GPU job holds the lock | 12 FF_LOCK_HELD=1 but the lock is free | 10 GPU busy (median of 5 samples over 10 s:
#   utilization > 5 % or memory used > 500 MiB) | 20 not on AC power | 30 uncommitted changes
#   | 40 under 20 GB disk free, or nvidia-smi fails
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK=/tmp/flashference-gpu.lock
mode=start; wait=0
for a in "$@"; do
  case $a in
    --gpu) mode=gpu ;;
    --wait) wait=1 ;;
    *) echo "preflight.sh: unknown arg $a" >&2; exit 2 ;;
  esac
done

median() { tr ' ' '\n' | sort -n | awk '{v[NR]=$1} END {print v[int((NR+1)/2)]}'; }

check_once() {
  local code=0 util=() mem=() line disk dirty=false lock=false bat smi
  disk=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
  smi=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,driver_version,pstate \
        --format=csv,noheader,nounits 2>/dev/null) || smi=""
  if [[ -z $smi || ${disk:-0} -lt 20 ]]; then code=40; fi
  if [[ $code == 0 && $mode == start && -n $(git -C "$ROOT" status --porcelain) ]]; then
    dirty=true; code=30
  fi
  # FF_LOCK_HELD=1: the caller says it holds the lock for this run, so a held lock isn't "another job".
  # gpu_lock records which case this was, so rule 8 can be checked from the raw files.
  local gpu_lock
  if flock -n "$LOCK" true 2>/dev/null; then
    gpu_lock=free
    if [[ ${FF_LOCK_HELD:-0} == 1 ]]; then gpu_lock=claimed-but-free; [[ $code == 0 ]] && code=12; fi
  elif [[ ${FF_LOCK_HELD:-0} == 1 ]]; then
    gpu_lock=held-by-caller
  else
    gpu_lock=held-by-other; lock=true; [[ $code == 0 ]] && code=11
  fi
  bat=$(timeout 30 powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" \
        2>/dev/null | tr -dc 0-9)
  if [[ $code == 0 && $bat != 2 ]]; then code=20; fi
  if [[ $code == 0 ]]; then
    for i in 1 2 3 4 5; do
      line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
      util+=("${line%%,*}"); mem+=("$(echo "${line##*,}" | tr -d ' ')")
      [[ $i -lt 5 ]] && sleep 2.5
    done
    local mu mm
    mu=$(echo "${util[*]}" | median); mm=$(echo "${mem[*]}" | median)
    if [[ $mu -gt 5 || $mm -gt 500 ]]; then code=10; fi
  fi
  IFS=', ' read -r temp csm cmem pw drv pst <<< "$smi"
  # kernel clock tick (nominal 10000 us): above it, CLOCK_MONOTONIC runs fast (NOTEBOOK.md, Phase 0.6)
  local tick
  tick=$(python3 -c 'import ctypes as c, ctypes.util as u
class T(c.Structure): _fields_ = [("m", c.c_uint)] + [(f"f{i}", c.c_long) for i in range(8)] + [("tv", c.c_long * 2), ("tick", c.c_long), ("pad", c.c_long * 20)]
t = T(); c.CDLL(u.find_library("c")).adjtimex(c.byref(t)); print(t.tick)' 2>/dev/null)
  printf '{"ts":"%s","mode":"%s","code":%d,"disk_free_gb":%s,"dirty":%s,"lock_held":%s,"battery_status":"%s","gpu_util":[%s],"mem_used_mib":[%s],"temp_c":"%s","clk_sm_mhz":"%s","clk_mem_mhz":"%s","power_w":"%s","driver":"%s","pstate":"%s","kernel_tick_us":"%s","gpu_lock":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$mode" "$code" "${disk:-0}" "$dirty" "$lock" "${bat:-}" \
    "$(IFS=,; echo "${util[*]:-}")" "$(IFS=,; echo "${mem[*]:-}")" \
    "${temp:-}" "${csm:-}" "${cmem:-}" "${pw:-}" "${drv:-}" "${pst:-}" "${tick:-}" "$gpu_lock"
  return "$code"
}

check_once; rc=$?
if [[ $wait == 1 ]]; then
  for _ in 1 2 3 4 5 6; do
    [[ $rc == 10 || $rc == 20 ]] || break
    sleep 600
    check_once; rc=$?
  done
fi
exit "$rc"
