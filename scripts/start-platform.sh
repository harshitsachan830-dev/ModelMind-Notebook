#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PORT="${PORT:-8899}"
WORKSPACE="${WORKSPACE:-$ROOT_DIR}"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    key="${key%%[[:space:]]*}"
    value="${value%$'\r'}"
    [[ -z "$key" || "$key" == \#* ]] && continue
    case "$key" in
      ML_PLATFORM_CLOUD_FALLBACK|GEMINI_API_KEY|GEMINI_MODEL)
        if [[ ${#value} -ge 2 && ( "$value" == \"*\" || "$value" == \'*\' ) ]]; then
          value="${value:1:${#value}-2}"
        fi
        export "$key=$value"
        ;;
    esac
  done < "$ENV_FILE"
  echo "Loaded Gemini settings from .env (values hidden)."
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating local virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
JUPYTER_BIN="$VENV_DIR/bin/jupyter"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
"$PYTHON_BIN" -m pip install -e "$ROOT_DIR/extensions/error-assistant"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. The local AI route is optional; install it from https://ollama.com/download"
  echo "Model installation remains explicit and must be approved before any pull command."
fi

exec "$JUPYTER_BIN" lab --no-browser --port "$PORT" --ServerApp.root_dir "$WORKSPACE"
