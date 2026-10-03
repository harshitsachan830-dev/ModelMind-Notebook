# ML Notebook Platform

A cross-platform ML notebook product built on JupyterLab and Jupyter Server. The notebook remains the standard Jupyter experience; product features are added through a small JupyterLab frontend extension and an authenticated Jupyter Server extension.

The notebook application is branded **ModelMind Notebook** with the supplied ModelMind logo. The Error Assistant opens in a polished side panel with clear error summaries, readable explanations, collapsible tracebacks, and separate controls for suggesting, copying, and applying fixes. JupyterLab's built-in Help links, update/news settings, and required third-party attribution remain enabled.

The target product adds an error assistant that diagnoses notebook failures, uses local Ollama first for complex explanations, optionally routes to Gemini through a secured backend, and only offers code changes after sandbox validation. The desktop distribution is the planned way to manage local runtime setup on macOS and Windows. A browser-only page cannot silently install Ollama on a user's computer.

## Status

**Local notebook and Ollama flows are implemented and tested on the developer Mac.** Gemini code-fix fallback requires two opt-ins: the server must set `ML_PLATFORM_CLOUD_FALLBACK=enabled` and a server-side `GEMINI_API_KEY`, and the user must check **Allow Gemini fallback** in the assistant. Ollama remains first; Gemini is contacted only after Ollama cannot provide a valid fix. The UI warns that code, traceback, and adjacent cell context can be sent to Google. No credential is sent to the browser. A live key/model check and one generation call succeeded with `gemini-3.8-flash`; a later call returned a transient 503, and an ambiguous NameError test produced a syntactically valid but semantically poor suggestion. Suggestions still require review; OS certification is pending.

See [context.md](context.md) for the report-derived phase checklist, implementation decisions, verified environment facts, and next action. Update it and this README whenever a phase, dependency pin, security decision, or supported-platform result changes.

## Requirements

- Python with `venv` and `pip`. The developer machine currently has Python 3.14.2; the project's minimum supported Python version has not yet been certified.
- Node.js and npm are needed to develop/build the frontend extension. The developer Mac has Node.js 24.11.1 and npm 11.6.2.
- macOS or Windows for the first desktop target. Linux packaging is a later, explicitly tested target.
- Ollama is required for the optional local explanation path. The manifest records a recommended model, and explicit user approval is required before the app triggers a local `ollama pull`.

## Start the platform locally

Use the project launcher scripts for a standard local start.

macOS / Linux:

```sh
./scripts/start-platform.sh
```

Windows PowerShell:

```powershell
.\scripts\start-platform.ps1
```

The launcher creates or updates the local virtual environment, installs the pinned project requirements and the JupyterLab extension, and starts JupyterLab with a local workspace. It does not download Ollama or the recommended model automatically. If the local AI path is desired, install Ollama and then use the explicit model installer flow or the manual `ollama pull` command after approval.

## Start the stock notebook baseline

Run these commands from the repository root.

macOS / Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e extensions/error-assistant
jupyter lab
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e extensions/error-assistant
jupyter lab
```

Open the local URL printed by JupyterLab. The Error Assistant command is in the View menu; it opens the right-side panel and checks authenticated platform, Ollama, recommended-model, and Gemini fallback status. Select a failed cell and choose **Explain Error** or **Suggest Fix**. Choose **Ollama (local)** or **Gemini (cloud)** from **Fix provider**; Gemini requires both server configuration and the explicit **Allow Gemini use** opt-in. With Ollama selected, that opt-in also permits Gemini fallback if Ollama cannot provide a valid suggestion. The selected provider requests a complete corrected cell and rejects responses that appear to contain only a partial patch. **Apply Fix** remains a separate action that replaces and reruns the cell, so review the full suggested code first. Choose **Code Help** or **Ask AI** to ask local Ollama about the selected cell. When Gemini is used, the cell code, traceback, and adjacent code are sent to Google.

To configure Gemini, enter your key in the root `.env` file. The startup scripts load it without printing its value, and `.env` is ignored by Git. Change the fallback setting to `enabled` only when you want to permit Gemini requests. The assistant still requires the user to check its per-session opt-in before sending code to Google. Keep the key out of notebooks and source files.

```dotenv
ML_PLATFORM_CLOUD_FALLBACK=enabled
GEMINI_API_KEY=paste-your-key-here
GEMINI_MODEL=gemini-3.8-flash
```

Restart JupyterLab after editing `.env`. Leaving the setting unset or set to `disabled` keeps all Gemini calls off.

The backend checks Ollama at `127.0.0.1:11434` with bounded timeouts and a 64 KiB response limit. The manifest is [model-manifest.json](extensions/error-assistant/ml_platform_error_assistant/model-manifest.json). This Mac is an M4 with 16 GB unified memory; Ollama reports version `0.35.0`, and the recommended model can be installed only through the explicit user-approved install flow or a manual `ollama pull` command. Nothing is downloaded silently.

Current checks: production build succeeds; the Python suite passes, the install-gate tests pass, and the local AI path is verified with the model installed. Browser verification continues to report nonfatal JupyterLab shared-package patch-version warnings; resolve or regression-test before a release candidate. See [context.md](context.md) for details.

## Build phases

1. **Baseline:** freeze JupyterLab/Jupyter Server versions; create an isolated environment and verify normal notebook behavior.
2. **Jupyter integration:** add the Error Assistant command/panel, authenticated server extension endpoints, and normalized notebook error capture.
3. **Diagnostics and local AI:** add deterministic Python/Pandas/NumPy/scikit-learn diagnostics, Ollama health/model checks, and structured local explanations.
4. **Safe fixes:** generate candidate patches, validate them in a disposable execution context, show a diff, and preserve the original cell unless the user explicitly applies a passing fix.
5. **Desktop setup and cloud fallback:** build the macOS/Windows launcher and Ollama setup flow; Gemini fix fallback is server-side, bounded, and disabled unless both the deployment and user opt in.
6. **Install and enable flow:** add explicit user approval before any model pull; document the local installer scripts and the route-based ensure flow.
7. **Release readiness:** test OS failure/recovery paths, notebook security boundaries, resource limits, attribution, and packaging before considering automatic updates.

The report's detailed sequence and acceptance criteria are tracked in [context.md](context.md). The explicit installer flow now guards model installation behind user approval; cloud routing remains disabled until later gates are completed.

## Local install & enable flow

The project ships with explicit installer helpers for the local model path:

- macOS/Linux: `scripts/install-ollama.sh`
- Windows PowerShell: `scripts/install-ollama.ps1`

These scripts do not auto-download anything. They check whether Ollama and the recommended model are present, and they only invoke `ollama pull` when the user runs the script intentionally. The backend API exposes the same policy through authenticated routes:

- `POST /api/local-ai/install`
- `POST /api/local-ai/model/ensure`

Both routes require explicit approval through the `allow_download` flag before a model pull is attempted.

## Non-negotiable product constraints

- Jupyter remains the notebook engine. Keep its editor, execution model, file browser, kernel status, output, terminal, and document behavior intact.
- The frontend never receives model credentials or calls model-specific APIs directly.
- Send only bounded, relevant error context to an AI service. Never automatically send secrets, environment variables, unrelated files, or all notebook outputs.
- Deterministic diagnoses should not trigger Gemini. Cloud use is disabled by default via `ML_PLATFORM_CLOUD_FALLBACK`; a remote fix also requires a server-side API key and explicit per-session UI consent.
- Never present an AI-generated patch as applicable until sandbox validation passes; show the diff and require user action.
- Untrusted notebook execution requires isolation, resource limits, restricted filesystem/network access, and secret isolation before multi-user deployment.
- Preserve Jupyter's required notices and licenses; review Ollama and model licenses before distribution.
