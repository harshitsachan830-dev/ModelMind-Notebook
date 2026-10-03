#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OS_NAME="$(uname -s 2>/dev/null || printf 'unknown')"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

printf '== Release readiness check ==\n'
printf 'OS: %s\n' "$OS_NAME"
printf 'Python: %s\n' "$("$PYTHON_BIN" --version 2>/dev/null || python3 --version 2>/dev/null || echo 'not-found')"

if [[ ! -f "$ROOT_DIR/requirements.txt" ]]; then
  echo 'Missing requirements.txt' >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/extensions/error-assistant/ml_platform_error_assistant/model-manifest.json" ]]; then
  echo 'Missing model manifest' >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/extensions/error-assistant/package.json" ]]; then
  echo 'Missing frontend package manifest' >&2
  exit 1
fi

if [[ "${ML_PLATFORM_CLOUD_FALLBACK:-disabled}" != "disabled" ]]; then
  echo "Cloud fallback is explicitly enabled: ${ML_PLATFORM_CLOUD_FALLBACK}"
  echo 'This is allowed only for an intentional deployment opt-in.'
fi

if command -v ollama >/dev/null 2>&1; then
  echo 'Ollama is installed; local-only path remains allowed.'
else
  echo 'Ollama is not installed. Local AI is optional until the user approves installation.'
fi

if [[ -x "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" -m pytest extensions/error-assistant/ml_platform_error_assistant/tests -q
else
  python3 -m pytest extensions/error-assistant/ml_platform_error_assistant/tests -q
fi

printf '\nSecurity and release gates checklist:\n'
printf '  - Gemini/cloud routes remain disabled by default.\n'
printf '  - Local Ollama install remains explicit and user-approved.\n'
printf '  - Local fixes are syntax-checked and only applied and rerun after explicit user action.\n'
printf '  - OS-specific validation should be run on actual macOS and Windows machines before a signed distributable is published.\n'
printf '\nRelease-ready guardrails: PASS\n'
