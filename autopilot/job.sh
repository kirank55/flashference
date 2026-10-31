#!/usr/bin/env bash
# job.sh: run a long command detached, so it survives the agent's session (autoplan.md §4.3).
#
#   autopilot/job.sh <name> -- <command...>            run holding the GPU lock (waits for it)
#   autopilot/job.sh --no-lock <name> -- <command...>  run without the lock (downloads, CPU builds)
#   autopilot/job.sh --running                         exit 0 while any job is running
#   autopilot/job.sh --status [name]                   one line per job: name state exit start end
#   autopilot/job.sh --kill <name>                     kill a job's whole process group
#
# Each job writes results/_jobs/<name>/ (git-ignored): cmd, pid, start, end, exit, log.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$ROOT/results/_jobs"
LOCK=/tmp/flashference-gpu.lock

if [[ ${1:-} == --_run ]]; then          # internal: the detached wrapper
  dir=$2; use_lock=$3; shift 3
  echo $$ > "$dir/pid"
  date -u +%FT%TZ > "$dir/start"
  if [[ $use_lock == 1 ]]; then flock "$LOCK" "$@"; else "$@"; fi
  rc=$?
  date -u +%FT%TZ > "$dir/end"
  echo "$rc" > "$dir/exit"
  exit "$rc"
fi

state() {  # running | done | dead
  local d=$1
  if [[ -f $d/exit ]]; then echo done
  elif [[ -f $d/pid ]] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then echo running
  else echo dead; fi
}

case ${1:-} in
  --running)
    for d in "$JOBS"/*/; do [[ -d $d ]] && [[ $(state "$d") == running ]] && exit 0; done
    exit 1 ;;
  --status)
    for d in "$JOBS"/${2:-*}/; do
      [[ -d $d ]] || continue
      printf '%s %s exit=%s start=%s end=%s\n' "$(basename "$d")" "$(state "$d")" \
        "$(cat "$d/exit" 2>/dev/null || echo -)" "$(cat "$d/start" 2>/dev/null || echo -)" \
        "$(cat "$d/end" 2>/dev/null || echo -)"
    done
    exit 0 ;;
  --kill)
    d="$JOBS/${2:?name}"
    [[ $(state "$d") == running ]] || { echo "job ${2} is not running" >&2; exit 1; }
    kill -- "-$(cat "$d/pid")"       # setsid made the wrapper a process-group leader
    exit $? ;;
esac

use_lock=1
if [[ ${1:-} == --no-lock ]]; then use_lock=0; shift; fi
name=${1:?usage: job.sh [--no-lock] <name> -- <command...>}; shift
[[ ${1:-} == -- ]] || { echo "usage: job.sh [--no-lock] <name> -- <command...>" >&2; exit 2; }
shift
[[ $# -gt 0 ]] || { echo "job.sh: no command" >&2; exit 2; }
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "job.sh: bad name '$name'" >&2; exit 2; }

dir="$JOBS/$name"
if [[ -d $dir ]]; then
  [[ $(state "$dir") == running ]] && { echo "job $name is already running" >&2; exit 1; }
  mv "$dir" "$dir.prev-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$dir"
printf '%q ' "$@" > "$dir/cmd"; echo >> "$dir/cmd"
echo "lock=$use_lock cwd=$PWD" >> "$dir/cmd"
setsid nohup "${BASH_SOURCE[0]}" --_run "$dir" "$use_lock" "$@" > "$dir/log" 2>&1 < /dev/null &
for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -f $dir/pid ]] && break; sleep 0.2; done
echo "started job $name (pid $(cat "$dir/pid" 2>/dev/null || echo '?')), log: $dir/log"
