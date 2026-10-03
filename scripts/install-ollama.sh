#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-qwen2.5-coder:7b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download" >&2
  exit 1
fi

if ! ollama list | grep -Fq "$MODEL_ID"; then
  echo "Pulling model: $MODEL_ID"
  ollama pull "$MODEL_ID"
else
  echo "Model already installed: $MODEL_ID"
fi

echo "Model ready: $MODEL_ID"
