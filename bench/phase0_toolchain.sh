#!/usr/bin/env bash
# Phase 0 / Gate 0 evidence: the toolchain as this run sees it, saved as a raw file.
#   bench/phase0_toolchain.sh > results/0/toolchain.txt
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/autopilot/env.sh"
section() { printf '\n## %s\n$ %s\n' "$1" "$2"; bash -c "$2" 2>&1; echo "(exit $?)"; }
echo "# toolchain snapshot $(date -u +%FT%TZ)"
section "nvcc" "nvcc --version"
section "nvidia-smi" "nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv"
section "torch" "python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
section "llama-server" "$ROOT/llama.cpp/build/bin/llama-server --version"
section "llama.cpp checkout" "git -C $ROOT/llama.cpp rev-parse HEAD; git -C $ROOT/llama.cpp status --porcelain | wc -l"
section "build flags" "grep -E '^(CMAKE_BUILD_TYPE|GGML_CUDA|GGML_CUDA_GRAPHS|GGML_CUDA_FA|CMAKE_CUDA_ARCHITECTURES|LLAMA_OPENSSL|GGML_NATIVE):' $ROOT/llama.cpp/build/CMakeCache.txt"
# the executables are thin launchers; the code lives in the shared libraries next to them
section "binaries and libraries" "cd $ROOT && find llama.cpp/build/bin -maxdepth 1 -type f \\( -name 'llama-*' ! -name 'llama-test*' -o -name '*.so*' \\) | sort | xargs sha256sum"
section "CUDA libraries llama-server loads" "ldd $ROOT/llama.cpp/build/bin/llama-server | grep -E 'cuda|cublas'"
section "gcc / cmake / ninja" "gcc --version | head -1; cmake --version | head -1; ninja --version"
