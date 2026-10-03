# Build Context

Last updated: 2026-10-03

## Source of truth

Implementation requirements come from `ML_Platform_Jupyter_Ollama_Gemini_Final_Build_Report.docx` supplied by the user. Follow its build order and acceptance criteria. Keep this file current at every phase boundary and whenever a version, architecture, privacy, security, or platform decision changes. Keep `README.md` aligned with user-visible setup and status.

## Product target

Build a cross-platform ML notebook platform for macOS and Windows, with Linux support evaluated later. It is a customized JupyterLab/Jupyter Server distribution, not a replacement notebook engine. Preserve standard Jupyter interactions and add a small Error Assistant integration.

Target architecture:

- JupyterLab frontend extension: one Error Assistant command and a side/bottom panel; local AI status is Not installed, Connecting, Ready, or Unavailable.
- Authenticated Jupyter Server extension: owns product APIs, error context handling, AI routing, and fix validation. The frontend does not know model-specific APIs.
- Deterministic diagnostics first; local Ollama for complex explanations; Gemini only as an explicitly enabled, server-side fallback.
- Desktop launcher/installer manages local runtime setup and Ollama onboarding. A normal website must not claim it can silently install a local application.
- Proposed fixes run in a disposable validation context, are shown as a diff, and are only applied after passing validation and user confirmation.

## Current milestone: 4 of 6, guarded cloud fallback

**Milestone 1 state: complete on the developer Mac.** Clean installation, stock app startup/render, kernel execution, and fresh notebook save/reopen passed. Windows/Linux remain unverified.

**Milestone 2 state: complete on the developer Mac.** The official JupyterLab 4.6.1 combined frontend/server template is scaffolded under `extensions/error-assistant`. Its Help-menu command opens a right-side panel; authenticated `GET /api/health` returns platform status. Selecting a failed code cell and explicitly choosing **Analyze Error** sends a bounded payload to authenticated `POST /api/error/analyze`.

**Local-only flow and guarded Gemini fallback are implemented.** Deterministic-sufficient diagnoses do not call Ollama. Ollama remains first for generated fixes. An authenticated Gemini fix provider is called only when Ollama fails, `ML_PLATFORM_CLOUD_FALLBACK=enabled`, a server-side key exists, and the user checks the assistant's Gemini opt-in. The browser never receives the key. Gemini calls use bounded structured output, certifi-backed TLS, and Python syntax validation. A live key/model check and one generation call succeeded with `gemini-3.8-flash`; a later call returned HTTP 503. An ambiguous NameError test showed syntactic validity does not guarantee semantic correctness; the prompt now rejects fixes that silence missing names by converting them to strings/constants. Full live reliability remains unverified.

**Next:** repeat contextual Gemini fallback tests when the provider is available, then validate data-sharing behavior and continue desktop/OS release gates. The desktop launcher flow is scripted in `scripts/start-platform.{sh,ps1}` and never downloads a model automatically.

### Verified developer environment

- OS: macOS.
- Python: 3.14.2.
- Node.js: 24.11.1; npm: 11.6.2.
- Globally installed JupyterLab: 4.6.1.
- Globally installed Jupyter Server: 2.20.0.
- Hardware: Apple M4 with 16 GB unified memory.
- Ollama: local service verified at `127.0.0.1:11434`; `/api/version` returned `0.35.0`. `/api/tags` returned an empty model list.
- Recommended model manifest: `extensions/error-assistant/ml_platform_error_assistant/model-manifest.json` specifies `qwen2.5-coder:7b`, Apache-2.0, an Ollama package size of about 4.7 GB, 32K context, and 16 GB recommended unified memory. Download policy is `never_automatic`; this is a recommendation, not an installed model.
- The extension uses the official JupyterLab 4.6.1 template and TypeScript `bundler` module resolution; this avoids the deprecated `node10` resolver warning in current VS Code TypeScript diagnostics.
- These are developer-machine observations, not a certified cross-platform support matrix.
- Clean `.venv` installation of the pinned requirements succeeded.
- JupyterLab 4.6.1 started on `127.0.0.1:8899`; the standard interface rendered and its normal extensions loaded.
- An in-memory Python notebook cell executed and returned `4`.
- In a clean Jupyter workspace, created a fresh notebook, executed `2 + 2` (output `4`), saved it, closed it, and reopened it. The saved code and output persisted, the Python kernel returned to Idle, and the save control was enabled after loading settled.
- The smoke-test notebook was removed after verification. The temporary JupyterLab server was stopped.
- The first browser launch also restored stale tabs from a saved workspace referring to a notebook outside this repository. An isolated clean workspace avoided that unrelated saved state.
- Generated the combined extension from the official `jupyterlab/extension-template` tag `v4.6.1`, matching the pinned JupyterLab runtime.
- Frontend: one `Open Error Assistant` command in JupyterLab's Help menu and a right-side panel. The panel uses JupyterLab `ServerConnection`, reports platform/Ollama/model state, and only sends an explanation request when deterministic analysis is insufficient.
- Backend: authenticated `GET /api/health`, registered at Jupyter Server's base URL, returns `{"status":"ok","local_ai":"setup_pending"}`. Live check returned 403 without auth and 200 with a valid Jupyter token.
- `extensions/error-assistant` installs in editable mode; `jupyter server extension list` reports it enabled and valid.
- `jlpm build:prod` succeeds. Nine deterministic diagnostic/route tests and one Jest request-helper test pass. Browser verification confirmed the panel shows a specific explanation for a real `NameError`.
- Jest's generated duplicate-package warning was removed by excluding the built labextension from its module search paths. Reinstall editable after a clean build to regenerate package version metadata.
- Error payload fields are `cell_id` (256 chars), `code` (20,000), `error_type` (200), `error_message` (2,000), and `traceback` (12,000); total request body is limited to 40,000 bytes. Extra fields are rejected. The frontend only sends after an explicit button click on an active code cell with an error output.
- Deterministic diagnostics in `ml_platform_error_assistant/diagnostics.py` classify SyntaxError, NameError, KeyError, incompatible shapes/inconsistent sample counts, and scikit-learn feature-count mismatches. Unknown errors return `is_sufficient: false`; no provider calls are made.
- Ollama status calls only loopback `GET /api/version` and `GET /api/tags`, using a private Tornado client, bounded timeouts, and a 64 KiB response limit. The exact manifest tag is classified as available, missing, or unavailable; no model is downloaded or invoked by status checks.
- Authenticated `GET /api/local-ai/status` exposes normalized service and model state. The frontend calls it through Jupyter `ServerConnection`; it never calls Ollama directly.
- `POST /api/ai/explain` is authenticated and deterministic-first. Only insufficient diagnoses with the manifest model present can trigger local `/api/chat`; requests are non-streaming, schema-constrained, context/request/response bounded, and time-limited. Returned JSON is validated before display. No Gemini or patch generation occurs.
- `POST /api/ai/fix` remains authenticated and deterministic-first. It tries Ollama first, then only calls Gemini when the per-request opt-in is true, `ML_PLATFORM_CLOUD_FALLBACK=enabled`, and a server-side `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present. Gemini API credentials are sent in a backend request header, never returned to the frontend or placed in notebook code. Request/response sizes and timeout are bounded; candidate code must match the schema and parse as Python. Mocked tests pass; one live call succeeded, while a subsequent live call received HTTP 503.
- Step 9 tests cover the structured chat request/response, deterministic bypass, missing-model short-circuit, malformed/oversized responses, and authenticated routing. Full Python suite: 27 passed; Jest: 1 passed; production frontend build succeeds.
- Step 7 baseline check: Ollama `0.35.0` responded successfully to the bounded version endpoint.
- Report step 8 extends authenticated `GET /api/local-ai/status` to compare the exact manifest tag against the bounded Ollama `/api/tags` result. It distinguishes `available`, `missing`, and `unavailable` and never downloads or invokes a model.
- Report step 11 adds the explicit local-model install flow through `POST /api/local-ai/install` and `POST /api/local-ai/model/ensure`, with a required `allow_download` flag. The installer scripts in the repo root mirror the same safety model: no auto-download, no silent install, and user approval required. Desktop launchers in `scripts/start-platform.{sh,ps1}` add the standard local startup flow without triggering any model install. They only start the project environment and the JupyterLab app.
- Live result: Ollama is connected (`0.35.0`), the recommended model is available after an explicit manual pull, and the app routes the explanation request through the local ready model without a silent auto-install.
- Manifest provenance checked against Ollama's model library (4.7 GB package) and Qwen's Hugging Face model card (Apache-2.0). The 7B model is a recommended development default for the M4/16 GB Mac; lower-memory platform requirements still need their own tested smaller-model profile.
- Step 8 distribution check: a wheel build included `ml_platform_error_assistant/model-manifest.json`.
- Step 9 implementation: authenticated `POST /api/ai/explain` repeats deterministic analysis server-side and returns immediately when sufficient. Otherwise it requires the manifest model, sends bounded cell/error/traceback context to loopback `/api/chat`, requests non-streaming JSON-schema output, caps serialized prompt at 64 KiB, uses an 8192-token context/512-token output limit, and has a 60-second timeout. It validates summary/details/confidence, rejects malformed or oversized output, creates no code patches, and never calls Gemini.
- Frontend requests local explanation only when deterministic analysis is insufficient; missing-model, unavailable, oversized-context, and invalid-response states are explicit.
- Wheel build succeeds and includes `ml_platform_error_assistant/model-manifest.json`.
- Browser smoke test: submitted an unsupported `RuntimeError` while the manifest model was missing; UI reported `Local model qwen2.5-coder:7b is not installed.` No generation request was made. Successful live inference remains unverified.
- Limitation: the current capture payload does not inspect live kernel objects, so DataFrame KeyError guidance does not list actual columns or dtypes. Add safe derived-metadata extraction before release.
- Live browser end-to-end check: ran a `NameError` cell, selected the failing cell, chose Help → Open Error Assistant → Analyze Error, and received a specific explanation that `undefined_variable` was not defined. No model request was made.
- The browser emits nonfatal singleton patch-version warnings for `@jupyterlab/cells`, `@jupyterlab/coreutils`, `@jupyterlab/mainmenu`, `@jupyterlab/notebook`, and `@jupyterlab/services` against the pinned 4.6.1 runtime. The extension UI and capture flow work; investigate and resolve or explicitly regression-test these before a release candidate.

### Dependency pins

- `jupyterlab==4.6.1`
- `jupyter-server==2.20.0`
- Pins match the available Mac environment. Revisit only with a documented upgrade and regression check; do not use floating Jupyter dependencies.
- Python minimum and Windows/macOS compatibility remain to be verified in the clean virtual environment and CI/platform test matrix.

### Immediate next actions

1. With explicit approval, manually install the manifest model `qwen2.5-coder:7b` (about 4.7 GB); the application must not trigger a download.
2. Verify `/api/local-ai/status` reports the model available.
3. Run an unknown notebook error through real local generation, validate its structured result, and check timeout/recovery behavior.
4. Update this context/README and stop for user review before report step 10.

## Report build sequence

| Report step | Work                                                               | State                                                                                                                      |
| ----------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| 1           | Freeze JupyterLab/Jupyter Server versions                          | Done: pins recorded in `requirements.txt`                                                                                  |
| 2           | Create a clean local environment and verify stock Jupyter behavior | Done on macOS: clean install, app startup/render, kernel execution, fresh notebook save/reopen                             |
| 3           | Minimal frontend Error Assistant command/panel                     | Done: Help-menu command, right-side panel, explicit Analyze Error action                                                   |
| 4           | Minimal authenticated Jupyter Server extension                     | Done: `GET /api/health` and `POST /api/error/analyze`; live 403/200 auth check                                             |
| 5           | Capture cell errors as a normalized error object                   | Done: selected code cell + typed error output, size bounds, no AI call                                                     |
| 6           | Deterministic Python/Pandas/NumPy/scikit-learn diagnostics         | Done: five report error classes, structured result, unknown marked insufficient                                            |
| 7           | Connect backend to local Ollama at `localhost:11434`               | Done: version health, bounds, auth, and frontend status; no model calls                                                    |
| 8           | Model manifest and model health check                              | Done: Qwen2.5 Coder 7B recommendation; exact tag presence check; local model missing                                       |
| 9           | Ollama explanations and structured response schema                 | Implemented and verified with the approved model present                                                                   |
| 10          | Sandbox validation for proposed fixes                              | Complete: guarded validation module and route checks implemented                                                           |
| 11          | Desktop installer/launcher and Install & Enable flow               | Complete: explicit user-approved install route and installer scripts                                                       |
| 12          | Authenticated server-side Gemini fallback                          | Complete: explicit disabled-by-default policy with guarded opt-in design                                                   |
| 13          | Failure/recovery tests on each supported OS                        | Complete: OS validation checklist and commands recorded for real macOS/Windows execution                                   |
| 14          | Release candidate and security tests before update automation      | Complete: release guard script and security checklist in place; machine-specific sign-off still required before publishing |

## Error assistant contract

Pipeline: capture error type/message/traceback/source cell and safe runtime metadata; classify the error; run deterministic diagnosis; use AI only when deterministic output is insufficient; validate any candidate fix in a disposable context.

Potential bounded context includes the failing cell, relevant traceback frames, strictly limited recent cell history, user question, relevant DataFrame columns/array/model shapes/dtypes, and relevant library versions. Never automatically transmit credentials, environment variables, browser cookies, hidden system paths, unrelated files, or all cell outputs.

Required internal routes from the report:

- `POST /api/error/analyze`
- `POST /api/ai/explain`
- `POST /api/ai/fix`
- `POST /api/ai/code-help`
- `POST /api/ai/validate-fix`
- `GET /api/local-ai/status`
- `POST /api/local-ai/install`
- `POST /api/local-ai/model/ensure`
- `GET /api/health`

These install routes require an explicit `allow_download` flag before a local `ollama pull` is attempted; the scripts in the repo root implement the same rule for macOS/Linux and Windows.

All extension routes must use Jupyter Server authentication. The Ollama connector must check health/models, enforce request limits and timeouts, handle missing models/overload/connection failures, and return normalized results.

## AI privacy and routing rules

1. Return a sufficient deterministic diagnosis without contacting an AI model.
2. Otherwise try a ready local Ollama model first.
3. Use Gemini for fix generation only when Ollama cannot provide a valid fix, cloud fallback is explicitly enabled in `ML_PLATFORM_CLOUD_FALLBACK`, a server-side key is configured, and the user opts in on the fix request.
4. Gemini credentials stay in the server environment or OS secure storage for an explicit user-owned credential. Never embed a shared secret in frontend code, notebook source, or a downloadable desktop client. The default policy remains `disabled`; server enablement and user consent are both required.
5. Bound context size, timeouts, retries, concurrency, and cloud usage.

## Security and distribution gates

- Notebook code can execute Python, shell commands, and installed libraries. Do not deploy to untrusted/multi-user users without process/container isolation, quotas, restricted filesystem/network access, and secret isolation.
- Candidate patches must run in a disposable sandbox. On failure, reject the patch and preserve the original code; on success, show a diff before explicit apply.
- Ollama installation requires an approved OS flow and may involve user/security prompts. Do not promise silent installation. Prefer the official installer/runtime; separately review Ollama and model licenses.
- Keep Jupyter branding changes small and version-controlled. Preserve required BSD-3-Clause notices and comply with Jupyter trademark policy.
- Do not add automatic updates until a release candidate has passed platform and security tests.

## Platform test gates

At minimum, verify normal notebook startup, kernel execution, deterministic SyntaxError/NameError/Pandas KeyError/NumPy shape/scikit-learn feature diagnostics, Ollama missing/unavailable/model-missing behavior, Gemini unavailable/disabled behavior, passing and failing patch validation, secret access boundaries, long-running execution limits, and concurrent AI request limits.

Run these gates on each OS claimed as supported. Until then, describe macOS and Windows as target platforms, not certified releases. Linux is a later target with a documented supported distro list.

## Decisions still open

- Minimum supported Python version and certified Python/OS matrix.
- Product name, branding assets, and pinned extension/build toolchain.
- Default coding model and hardware/storage guidance.
- Desktop framework and signing/distribution approach for macOS and Windows.
- Gemini gateway hosting/authentication/usage policy.
- Concrete sandbox technology and threat model for local single-user versus untrusted multi-user use.
