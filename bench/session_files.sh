#!/usr/bin/env bash
# session_files.sh: every file modified since a given UTC time under $HOME and /tmp (Phase 0
# review fix: a repeatable check that no unrecorded work exists outside the repo's history).
# Bulk trees are only counted: build products, installs, caches, agent-harness files, other
# applications' files and the reviewers' scratch. Everything else is listed one file per line;
# files inside the repo are tagged tracked / ignored / UNTRACKED by git.
#   bench/session_files.sh "2026-10-31 08:55:00" > results/0/session-files.txt
set -u
since=${1:?usage: session_files.sh "<UTC time>"}
H=$HOME
R=$H/code-wsl/flashference
groups=(
  "build/install:$R/.git $R/llama.cpp $R/.venv $R/ref $H/opt/cuda-12.8"
  "caches:$(echo "$H"/.cache/* | tr ' ' '\n' | paste -sd' ')"
  "agent harness (this executor's tool outputs and settings):/tmp/claude-1000 $H/.claude $H/.claude.json $H/.t3"
  "other applications on this machine:/tmp/node-compile-cache $H/.gemini $H/.npm $H/.codex $H/.opencode $H/.copilot $H/.config $H/.local $H/.vscode-server /tmp/initgoogle_syslog_dir.1000 $(echo /tmp/vscode-* /tmp/remote-wsl-loc.txt /tmp/.bun-* | tr ' ' '\n' | paste -sd' ')"
  "reviewers' scratch:/tmp/review-0 /tmp/review-0-r2 /tmp/review-0-r2b"
  "executor's scratch clone for testing check.py before committing it:/tmp/ff-selftest"
)
echo "# files modified since $since UTC under $H and /tmp, listed $(date -u +%FT%TZ)"
prune=()
for g in "${groups[@]}"; do
  echo; echo "## counted only: ${g%%:*}"
  for d in ${g#*:}; do
    [[ -e $d ]] || continue
    n=$(find "$d" -xdev -type f -newermt "$since" 2>/dev/null | wc -l)
    [[ $n -gt 0 ]] && printf '%7d  %s\n' "$n" "$d"
    prune+=(-path "$d" -prune -o)
  done
done
echo; echo "## listed individually (mtime UTC, bytes, path, git state for repo files)"
find "$H" /tmp -xdev "${prune[@]}" -type f -newermt "$since" -printf '%TY-%Tm-%TdT%TH:%TM:%.12TSZ  %s  %p\n' 2>/dev/null | sort |
while read -r t sz p; do
  tag=""
  if [[ $p == "$R"/* ]]; then
    rel=${p#"$R"/}
    if git -C "$R" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then tag=tracked
    elif git -C "$R" check-ignore -q -- "$rel"; then tag=ignored
    else tag=UNTRACKED; fi
  fi
  echo "$t  $sz  $p${tag:+  [$tag]}"
done
