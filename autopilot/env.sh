# env.sh: source this before any work (autoplan.md §3, Phase 0.2/0.4).
#   source autopilot/env.sh
# CUDA toolkit lives in user space (no sudo). WSL's own lib dir comes first in
# LD_LIBRARY_PATH so the Windows driver's libcuda.so is the one that loads.
FF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FF_ROOT
export CUDA_HOME="$HOME/opt/cuda-12.8"
export VIRTUAL_ENV="$FF_ROOT/.venv"
case ":$PATH:" in
  *":$CUDA_HOME/bin:"*) ;;
  *) export PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:$PATH" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
  *":$CUDA_HOME/lib64:"*) ;;
  *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
# A Hub network check once hung a run for minutes. Download steps override this
# with HF_HUB_OFFLINE=0 on their own command line.
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export FF_MODEL="$FF_ROOT/models/Qwen_Qwen3.5-0.8B-bf16.gguf"
export FF_GPU_LOCK=/tmp/flashference-gpu.lock
