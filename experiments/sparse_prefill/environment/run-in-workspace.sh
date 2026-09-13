#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$task_root"
if ! grep -Fq '580.173.02' /proc/driver/nvidia/version; then
  printf 'Kernel driver changed; revalidate workspace driver libraries before running.\n' >&2
  exit 1
fi
export LD_LIBRARY_PATH="$task_root/vendor/driver-580.173.02/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$task_root/.venv/bin:$task_root/vendor/driver-580.173.02/usr/bin:$PATH"
export TMPDIR="$task_root/tmp"
export XDG_CACHE_HOME="$task_root/cache"
export XDG_CONFIG_HOME="$task_root/config"
export XDG_DATA_HOME="$task_root/data"
export PIP_CACHE_DIR="$task_root/cache/pip"
export UV_CACHE_DIR="$task_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$task_root/python"
export UV_PYTHON_BIN_DIR="$task_root/tools/bin"
export UV_TOOL_DIR="$task_root/tools/managed"
export UV_TOOL_BIN_DIR="$task_root/tools/bin"
export PRE_COMMIT_HOME="$task_root/cache/pre-commit"
export HF_HOME="$task_root/cache/huggingface"
export TORCH_HOME="$task_root/cache/torch"
export TORCH_EXTENSIONS_DIR="$task_root/cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$task_root/cache/torch-inductor"
export FLASHINFER_WORKSPACE_BASE="$task_root/cache/flashinfer-workspace"
export VLLM_CONFIG_ROOT="$task_root/config/vllm"
export VLLM_CACHE_ROOT="$task_root/cache/vllm"
export VLLM_NO_USAGE_STATS=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRITON_CACHE_DIR="$task_root/cache/triton"
export CUDA_CACHE_PATH="$task_root/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
umask 077
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
if [ "$#" -eq 0 ]; then
  printf 'Usage: ./run-in-workspace.sh COMMAND [ARGS...]\n' >&2
  exit 2
fi
exec "$@"
