#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

HOST="${ANYLABELING_WEBAPP_HOST:-127.0.0.1}"
PORT="${ANYLABELING_WEBAPP_PORT:-8000}"

python -m uvicorn webapp.backend.app:app --host "$HOST" --port "$PORT"
