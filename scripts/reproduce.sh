#!/usr/bin/env bash
# One-command reproduction of the slmspeed headline results on Linux (CUDA GPU expected).
# Builds llama.cpp b11884 (63412d873) + patches/*.patch, fetches the bf16 GGUF, runs plain vs code vs copy-heavy
# configs via scripts/reproduce_bench.py and prints ratios vs plain.
# NOTE: written alongside reproduce.ps1 but only the PowerShell version has been run end to end (on Windows 11 / RTX 3050 Laptop).
#
# Usage: scripts/reproduce.sh [--smoke] [--passes N] [--n-predict N] [--llama-dir DIR] [--model FILE]
#                             [--build] [--build-only] [--include-prose] [--require-idle] [--cuda-arch ARCH]
# Env:   CMAKE_ARGS="..." replaces the default CUDA cmake flags (e.g. other backends - untested).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG=b11884; COMMIT=63412d873
WORK="$ROOT/work"; SRC="$WORK/llama.cpp"; OUT="$WORK/llama-patched"; BIN="$OUT/bin"
MODEL_URL="https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-bf16.gguf"
SMOKE=""; PASSES=3; NPRED=256; LLAMA_DIR="${SLMSPEED_LLAMA:-}"; MODEL=""; BUILD=0; BUILD_ONLY=0; PROSE=""; IDLE=""; ARCH="native"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke";; --passes) PASSES="$2"; shift;; --n-predict) NPRED="$2"; shift;;
    --llama-dir) LLAMA_DIR="$2"; shift;; --model) MODEL="$2"; shift;; --build) BUILD=1;; --build-only) BUILD_ONLY=1;;
    --include-prose) PROSE="--include-prose";; --require-idle) IDLE="--require-idle";; --cuda-arch) ARCH="$2"; shift;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac; shift
done

build_llama() {
  mkdir -p "$WORK"
  [ -d "$SRC/.git" ] || git -c core.autocrlf=false clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
  ( cd "$SRC"
    head="$(git rev-parse HEAD)"; case "$head" in $COMMIT*) ;; *) echo "llama.cpp is $head, expected $COMMIT" >&2; exit 1;; esac
    git checkout -- .
    git apply "$ROOT/patches/b4-trim-draft-head.patch"
    git apply "$ROOT/patches/e2-e4-adaptive-n-dynamic-vocab.patch"
    FLAGS="${CMAKE_ARGS:--DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH}"
    cmake -S . -B build-slmspeed -DCMAKE_BUILD_TYPE=Release $FLAGS -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
    cmake --build build-slmspeed --target llama-server llama-tokenize llama-quantize llama-bench -j "$(nproc)" )
  mkdir -p "$BIN"; cp -a "$SRC/build-slmspeed/bin/." "$BIN/"
  echo "== patched llama.cpp ready: $BIN"
}

if [ "$BUILD" = 1 ] || { [ -z "$LLAMA_DIR" ] && [ ! -x "$BIN/llama-server" ]; }; then build_llama; LLAMA_DIR="$BIN"; fi
[ -z "$LLAMA_DIR" ] && LLAMA_DIR="$BIN"
[ -x "$LLAMA_DIR/llama-server" ] || { echo "llama-server not found in $LLAMA_DIR" >&2; exit 1; }
echo "== llama.cpp: $LLAMA_DIR"
[ "$BUILD_ONLY" = 1 ] && exit 0

[ -n "$MODEL" ] || MODEL="$ROOT/models/Qwen_Qwen3.5-0.8B-bf16.gguf"
if [ ! -f "$MODEL" ]; then
  echo "== downloading the bf16 GGUF (~1.5 GB) to $MODEL"; mkdir -p "$(dirname "$MODEL")"; curl -L -C - -o "$MODEL" "$MODEL_URL"
fi
echo "== model: $MODEL"
export PYTHONUTF8=1
exec python3 scripts/reproduce_bench.py --llama-bin "$LLAMA_DIR" --model "$MODEL" --passes "$PASSES" --n-predict "$NPRED" $SMOKE $PROSE $IDLE
