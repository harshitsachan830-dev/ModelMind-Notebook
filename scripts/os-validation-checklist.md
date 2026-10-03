# OS validation checklist for release steps 12-14

This checklist is the operational gate for any supported OS validation before a desktop or signed distribution is published.

## Required checks

### macOS

- Start the project from the repo root using `./scripts/start-platform.sh`
- Confirm JupyterLab loads and the Error Assistant panel is visible
- Confirm the platform health endpoint responds with `cloud_fallback: disabled`
- Confirm a deterministic NameError diagnosis resolves locally without cloud access
- Confirm the local Ollama model status returns `available` or `missing` with explicit user approval required for install
- Confirm the release guard script runs successfully: `./scripts/release-check.sh`

### Windows

- Start the project from PowerShell using `./scripts/start-platform.ps1`
- Confirm JupyterLab loads and the Error Assistant panel is visible
- Confirm the platform health endpoint responds with `cloud_fallback: disabled`
- Confirm the local AI install path remains manual and explicit on Windows
- Confirm no silent model pull is triggered during startup
- Confirm the release guard script runs successfully: `./scripts/release-check.ps1`

## Security gates

- Gemini is disabled by default unless `ML_PLATFORM_CLOUD_FALLBACK=enabled` is explicitly set for a deployment
- Secrets are never embedded in frontend code or downloaded desktop clients
- Remote calls are blocked unless the environment policy and backend configuration allow them
- Notebook-execution patches still run only after sandbox validation and explicit user approval

## Release gate

A release candidate should not be published until these checks are executed on the actual target OS machine and the results are recorded.
