#!/usr/bin/env bash
set -euo pipefail

# Optional env override: ANYLABELING_CONDA_ENV=myenv ./run_anylabeling_gpu.sh
ANYLABELING_CONDA_ENV="${ANYLABELING_CONDA_ENV:-base}"
APP_HOST="${ANYLABELING_WEB_HOST:-127.0.0.1}"
APP_PORT="${ANYLABELING_WEB_PORT:-8765}"
MODE="web"

if [[ "${1:-}" == "--desktop" ]]; then
  MODE="desktop"
  shift
elif [[ "${1:-}" == "--web" ]]; then
  MODE="web"
  shift
fi

# If conda is available, run inside the requested env (default: base).
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ANYLABELING_CONDA_ENV}"
fi

# Force ONNX Runtime CUDA provider preference for AnyLabeling.
export ANYLABELING_FORCE_CUDA="${ANYLABELING_FORCE_CUDA:-1}"
export ANYLABELING_ENABLE_TENSORRT="${ANYLABELING_ENABLE_TENSORRT:-0}"
export ANYLABELING_ONNX_PROVIDERS="${ANYLABELING_ONNX_PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[AnyLabeling GPU] Launching with:"
echo "  ANYLABELING_FORCE_CUDA=${ANYLABELING_FORCE_CUDA}"
echo "  ANYLABELING_ENABLE_TENSORRT=${ANYLABELING_ENABLE_TENSORRT}"
echo "  ANYLABELING_ONNX_PROVIDERS=${ANYLABELING_ONNX_PROVIDERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ "${MODE}" == "desktop" ]]; then
  echo "[AnyLabeling GPU] Mode: Desktop GTK"
  exec python -m anylabeling.app "$@"
fi

URL="http://${APP_HOST}:${APP_PORT}"
echo "[AnyLabeling GPU] Mode: Browser App (${URL})"

if command -v xdg-open >/dev/null 2>&1; then
  # Open browser non-blocking; suppress output noise.
  (sleep 1; xdg-open "${URL}" >/dev/null 2>&1 || true) &
fi

exec python -m uvicorn webapp.backend.app:app --host "${APP_HOST}" --port "${APP_PORT}" "$@"
