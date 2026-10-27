#!/usr/bin/env bash
# One-command reproduction of the slmspeed headline results on Linux / WSL (CUDA GPU expected).
# Builds llama.cpp b11884 (63412d873) + patches/*.patch, fetches the bf16 GGUF, runs plain vs code vs copy-heavy
# configs via scripts/reproduce_bench.py and prints ratios vs plain.
# With nvcc: native CUDA build. Without nvcc (typical WSL): hybrid build, same idea as reproduce.ps1 —
# patched host code (src/ and common/ only) plus the official b11884 Ubuntu CUDA libraries.
# This tag publishes Ubuntu CUDA 12.8 and 13.4, not 12.4; 12.8 is the hybrid default.
#
# Usage: scripts/reproduce.sh [--smoke] [--passes N] [--n-predict N] [--llama-dir DIR] [--model FILE]
#                             [--build] [--build-only] [--include-prose] [--require-idle] [--cuda-arch ARCH]
# Env:   CMAKE_ARGS="..." replaces the default cmake flags (e.g. other backends - untested).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG=b11884; COMMIT=63412d873
CUDA_REL="${SLMSPEED_CUDA_REL:-12.8}"
WORK="$ROOT/work"; SRC="$WORK/llama.cpp"; OUT="$WORK/llama-patched"; BIN="$OUT/bin"
MODEL_URL="https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-bf16.gguf"
SMOKE=""; PASSES=3; NPRED=256; LLAMA_DIR="${SLMSPEED_LLAMA:-}"; MODEL=""; BUILD=0; BUILD_ONLY=0; PROSE=""; IDLE=""; ARCH="native"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke";; --passes) PASSES="$2"; shift;; --n-predict) NPRED="$2"; shift;;
    --llama-dir) LLAMA_DIR="$2"; shift;; --model) MODEL="$2"; shift;; --build) BUILD=1;; --build-only) BUILD_ONLY=1;;
    --include-prose) PROSE="--include-prose";; --require-idle) IDLE="--require-idle";; --cuda-arch) ARCH="$2"; shift;;
    -h|--help) sed -n '2,11p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac; shift
done

fetch() {
  local url="$1" dest="$2"
  if [ ! -s "$dest" ]; then
    echo "== downloading $(basename "$dest")"
    curl -L --fail -C - -o "$dest.partial" "$url"
    mv "$dest.partial" "$dest"
  fi
}

# Official Ubuntu CUDA tree + cudart/cublas next to the binary (RPATH is $ORIGIN).
stage_cuda_release() {
  local base="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}"
  local bin_tgz="$WORK/llama-${TAG}-bin-ubuntu-cuda-${CUDA_REL}-x64.tar.gz"
  local rt_tgz="$WORK/cudart-llama-${TAG}-bin-ubuntu-cuda-${CUDA_REL}-x64.tar.gz"
  local rel="$WORK/cuda-release" srcdir
  fetch "$base/$(basename "$bin_tgz")" "$bin_tgz"
  fetch "$base/$(basename "$rt_tgz")" "$rt_tgz"
  rm -rf "$rel"; mkdir -p "$rel" "$BIN"
  tar -xzf "$bin_tgz" -C "$rel"
  tar -xzf "$rt_tgz" -C "$rel"
  srcdir="$(find "$rel" -type f -name llama-server -printf '%h\n' | head -1)"
  [ -n "$srcdir" ] || { echo "llama-server missing in $bin_tgz" >&2; exit 1; }
  cp -a "$srcdir/." "$BIN/"
  find "$rel" -type f \( -name 'libcudart.so*' -o -name 'libcublas.so*' -o -name 'libcublasLt.so*' \) -exec cp -a {} "$BIN/" \;
}

# Replace host binaries/libs only. libggml-cuda.so and the CUDA runtime stay from the official release.
overlay_patched_host() {
  local built="$SRC/build-slmspeed/bin" f name
  [ -d "$built" ] || { echo "build dir missing: $built" >&2; exit 1; }
  find "$built" -maxdepth 1 -type f \( \
      -name 'llama-server' -o -name 'llama-tokenize' -o -name 'llama-quantize' -o -name 'llama-bench' \
      -o -name 'libllama.so*' -o -name 'libllama-common.so*' -o -name 'libllama-server-impl.so*' \
      -o -name 'libllama-quantize-impl.so*' -o -name 'libllama-bench-impl.so*' \
    \) -print | while read -r f; do
    cp -a "$f" "$BIN/"
  done
  # relative symlinks (libllama.so -> libllama.so.0) are not -type f
  find "$built" -maxdepth 1 -type l \( -name 'libllama.so*' -o -name 'libllama-common.so*' \) -print | while read -r f; do
    name="$(basename "$f")"
    rm -f "$BIN/$name"
    cp -a "$f" "$BIN/$name"
  done
  # grep -q would SIGPIPE strings, and pipefail would treat that as failure.
  if ! strings "$BIN/libllama.so" | grep -F "LLAMA_MTP_TRIM_FILE" >/dev/null; then
    echo "patched libllama.so does not contain the trim hook" >&2
    exit 1
  fi
}

build_llama() {
  mkdir -p "$WORK"
  [ -d "$SRC/.git" ] || git -c core.autocrlf=false clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
  local hybrid=0
  if ! command -v nvcc >/dev/null 2>&1; then hybrid=1; fi
  if [ "$hybrid" = 1 ]; then
    echo "== No CUDA toolkit: hybrid build (patched host code + official b${TAG#b} Ubuntu CUDA ${CUDA_REL} libraries)"
    stage_cuda_release
  fi
  ( cd "$SRC"
    head="$(git rev-parse HEAD)"; case "$head" in $COMMIT*) ;; *) echo "llama.cpp is $head, expected $COMMIT" >&2; exit 1;; esac
    git checkout -- .
    git apply "$ROOT/patches/b4-trim-draft-head.patch"
    git apply "$ROOT/patches/e2-e4-adaptive-n-dynamic-vocab.patch"
    if [ "$hybrid" = 1 ]; then
      FLAGS="${CMAKE_ARGS:--DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF}"
    else
      echo "== CUDA toolkit found: native CUDA build (arch $ARCH)"
      FLAGS="${CMAKE_ARGS:--DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH}"
    fi
    # $ORIGIN so the staged bin/ loads libggml-cuda.so beside the executable, not the build tree.
    cmake -S . -B build-slmspeed -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON $FLAGS -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
    cmake --build build-slmspeed --target llama-server llama-tokenize llama-quantize llama-bench -j "$(nproc)" )
  mkdir -p "$BIN"
  if [ "$hybrid" = 1 ]; then
    overlay_patched_host
  else
    cp -a "$SRC/build-slmspeed/bin/." "$BIN/"
  fi
  date -u +"patched %Y-%m-%dT%H:%M:%SZ cuda_rel=${CUDA_REL} hybrid=${hybrid}" > "$BIN/.slmspeed-patched"
  echo "== patched llama.cpp ready: $BIN"
}

# .slmspeed-patched distinguishes a finished hybrid/native build from the unpatched official tree.
if [ "$BUILD" = 1 ] || { [ -z "$LLAMA_DIR" ] && [ ! -f "$BIN/.slmspeed-patched" ]; }; then build_llama; LLAMA_DIR="$BIN"; fi
[ -z "$LLAMA_DIR" ] && LLAMA_DIR="$BIN"
[ -x "$LLAMA_DIR/llama-server" ] || { echo "llama-server not found in $LLAMA_DIR" >&2; exit 1; }
echo "== llama.cpp: $LLAMA_DIR"
# Official libggml-cuda.so resolves cudart/cublas via $ORIGIN; keep the dir on the path anyway.
export LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ "$BUILD_ONLY" = 1 ] && exit 0

[ -n "$MODEL" ] || MODEL="$ROOT/models/Qwen_Qwen3.5-0.8B-bf16.gguf"
if [ ! -f "$MODEL" ]; then
  echo "== downloading the bf16 GGUF (~1.5 GB) to $MODEL"; mkdir -p "$(dirname "$MODEL")"; curl -L -C - -o "$MODEL" "$MODEL_URL"
fi
echo "== model: $MODEL"
export PYTHONUTF8=1
exec python3 scripts/reproduce_bench.py --llama-bin "$LLAMA_DIR" --model "$MODEL" --passes "$PASSES" --n-predict "$NPRED" $SMOKE $PROSE $IDLE
