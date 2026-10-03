import json

import tornado

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

from .cloud_policy import (
    gemini_api_key_configured,
    gemini_fallback_enabled,
    health_policy_status,
)
from .diagnostics import analyze_error
from .gemini import suggest_fix_with_gemini
from .fix_validation import (
    normalize_fix_payload,
    suggest_rowwise_apply_fix,
    suggest_merge_suffix_fix,
    validate_candidate_fix,
)
from .ollama import (
    check_ollama_status,
    ensure_model_ready,
    explain_code_with_ollama,
    explain_with_ollama,
    suggest_fix_with_ollama,
)

MAX_CAPTURE_BYTES = 40_000
MAX_FIELD_LENGTHS = {
    "cell_id": 256,
    "code": 20_000,
    "error_type": 200,
    "error_message": 2_000,
    "traceback": 12_000,
}


def normalize_error_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("The request body must be a JSON object.")

    if set(payload) != set(MAX_FIELD_LENGTHS):
        raise ValueError("The request must contain exactly the supported error fields.")

    normalized = {}
    for field, max_length in MAX_FIELD_LENGTHS.items():
        value = payload[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string.")
        if len(value) > max_length:
            raise ValueError(f"{field} exceeds the {max_length}-character limit.")
        if field in {"cell_id", "error_type"} and not value:
            raise ValueError(f"{field} cannot be empty.")
        normalized[field] = value

    return normalized


class HealthHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        self.set_header("Content-Type", "application/json")
        health = {"status": "ok", "local_ai": "setup_pending"}
        health.update(health_policy_status())
        self.finish(json.dumps(health))


class AnalyzeErrorHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The error payload is too large."}))
            return

        try:
            payload = json.loads(self.request.body)
            normalized = normalize_error_payload(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.finish(
            json.dumps(
                {
                    "status": "analyzed",
                    "error": normalized,
                    "analysis": analyze_error(normalized),
                }
            )
        )


class ExplainErrorHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The error payload is too large."}))
            return

        try:
            normalized = normalize_error_payload(json.loads(self.request.body))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        deterministic = analyze_error(normalized)
        if deterministic["is_sufficient"]:
            self.finish(
                json.dumps(
                    {"status": "deterministic_sufficient", "analysis": deterministic}
                )
            )
            return

        result = await explain_with_ollama(normalized)
        self.finish(json.dumps(result))


class LocalAIStatusHandler(APIHandler):
    @tornado.web.authenticated
    async def get(self):
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(await check_ollama_status()))


class LocalAIInstallHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The install request is too large."}))
            return

        try:
            payload = json.loads(self.request.body)
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object.")
            model_id = payload.get("model_id")
            if model_id is not None and not isinstance(model_id, str):
                raise ValueError("The model_id must be a string when provided.")
            allow_download = bool(payload.get("allow_download", False))
            result = await ensure_model_ready(model_id=model_id, allow_download=allow_download)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.finish(json.dumps(result))


class LocalAIModelEnsureHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The ensure request is too large."}))
            return

        try:
            payload = json.loads(self.request.body)
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object.")
            model_id = payload.get("model_id")
            if model_id is not None and not isinstance(model_id, str):
                raise ValueError("The model_id must be a string when provided.")
            allow_download = bool(payload.get("allow_download", False))
            result = await ensure_model_ready(model_id=model_id, allow_download=allow_download)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.finish(json.dumps(result))


class SuggestFixHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The fix request is too large."}))
            return

        try:
            payload = json.loads(self.request.body)
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object.")
            error = payload.get("error")
            notebook_context = payload.get("notebook_context", "")
            provider = payload.get("provider", "ollama")
            allow_gemini_fallback = payload.get("allow_gemini_fallback", False)
            if provider not in {"ollama", "gemini"}:
                raise ValueError("provider must be either 'ollama' or 'gemini'.")
            if not isinstance(allow_gemini_fallback, bool):
                raise ValueError("allow_gemini_fallback must be a boolean.")
            if not isinstance(notebook_context, str) or len(notebook_context) > 12_000:
                raise ValueError("Notebook context must be a string of at most 12000 characters.")
            normalized = normalize_error_payload(error)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        if provider == "ollama":
            deterministic_fix = suggest_merge_suffix_fix(normalized, normalized["code"])
            if deterministic_fix is None:
                deterministic_fix = suggest_rowwise_apply_fix(
                    normalized, normalized["code"]
                )
            if deterministic_fix is not None:
                self.finish(json.dumps(deterministic_fix))
                return

        if provider == "gemini":
            if not allow_gemini_fallback:
                self.finish(
                    json.dumps(
                        {
                            "status": "gemini_opt_in_required",
                            "message": "Enable the Gemini opt-in before sending notebook code to Google.",
                        }
                    )
                )
                return
            if not gemini_fallback_enabled():
                self.finish(
                    json.dumps(
                        {
                            "status": "gemini_disabled",
                            "message": "Gemini is disabled by the server policy.",
                        }
                    )
                )
                return
            if not gemini_api_key_configured():
                self.finish(
                    json.dumps(
                        {
                            "status": "gemini_unconfigured",
                            "message": "Set GEMINI_API_KEY in the Jupyter server environment to enable Gemini.",
                        }
                    )
                )
                return
            gemini_result = await suggest_fix_with_gemini(
                normalized, notebook_context=notebook_context
            )
            gemini_result["fallback_used"] = False
            self.finish(json.dumps(gemini_result))
            return

        local_result = await suggest_fix_with_ollama(
            normalized, notebook_context=notebook_context
        )
        if local_result.get("status") == "suggested":
            local_result.setdefault("provider", "ollama")
            self.finish(json.dumps(local_result))
            return

        if not allow_gemini_fallback:
            local_result["fallback_used"] = False
            self.finish(json.dumps(local_result))
            return
        if not gemini_fallback_enabled():
            self.finish(
                json.dumps(
                    {
                        "status": "gemini_disabled",
                        "fallback_used": False,
                        "ollama_status": local_result.get("status"),
                        "message": "Gemini fallback is disabled by the server policy.",
                    }
                )
            )
            return
        if not gemini_api_key_configured():
            self.finish(
                json.dumps(
                    {
                        "status": "gemini_unconfigured",
                        "fallback_used": False,
                        "ollama_status": local_result.get("status"),
                        "message": "Set GEMINI_API_KEY in the Jupyter server environment to enable Gemini.",
                    }
                )
            )
            return

        gemini_result = await suggest_fix_with_gemini(
            normalized, notebook_context=notebook_context
        )
        gemini_result["ollama_status"] = local_result.get("status")
        gemini_result["fallback_used"] = gemini_result.get("status") == "suggested"
        self.finish(json.dumps(gemini_result))


class CodeHelpHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The code-help request is too large."}))
            return

        try:
            payload = json.loads(self.request.body)
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object.")
            code = payload.get("code")
            question = payload.get("question")
            notebook_context = payload.get("notebook_context", "")
            if not isinstance(code, str) or not code.strip() or len(code) > 20_000:
                raise ValueError("Code must be a non-empty string of at most 20000 characters.")
            if not isinstance(question, str) or not question.strip() or len(question) > 1_000:
                raise ValueError("Question must be a non-empty string of at most 1000 characters.")
            if not isinstance(notebook_context, str) or len(notebook_context) > 12_000:
                raise ValueError("Notebook context must be a string of at most 12000 characters.")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.finish(
            json.dumps(
                await explain_code_with_ollama(
                    code, question, notebook_context=notebook_context
                )
            )
        )


class ValidateFixHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        self.set_header("Content-Type", "application/json")
        if len(self.request.body) > MAX_CAPTURE_BYTES:
            self.set_status(413)
            self.finish(json.dumps({"error": "The validation request is too large."}))
            return

        try:
            payload = json.loads(self.request.body)
            normalized = normalize_fix_payload(payload)
            result = validate_candidate_fix(
                normalized["original_code"],
                normalized["candidate_code"],
                payload.get("error_type"),
                payload.get("error_message"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.finish(json.dumps(result))


def setup_route_handlers(web_app):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    health_route_pattern = url_path_join(base_url, "api", "health")
    analyze_error_route_pattern = url_path_join(base_url, "api", "error", "analyze")
    explain_error_route_pattern = url_path_join(base_url, "api", "ai", "explain")
    local_ai_status_route_pattern = url_path_join(base_url, "api", "local-ai", "status")
    local_ai_install_route_pattern = url_path_join(base_url, "api", "local-ai", "install")
    local_ai_model_ensure_route_pattern = url_path_join(base_url, "api", "local-ai", "model", "ensure")
    suggest_fix_route_pattern = url_path_join(base_url, "api", "ai", "fix")
    code_help_route_pattern = url_path_join(base_url, "api", "ai", "code-help")
    validate_fix_route_pattern = url_path_join(base_url, "api", "ai", "validate-fix")
    handlers = [
        (health_route_pattern, HealthHandler),
        (analyze_error_route_pattern, AnalyzeErrorHandler),
        (explain_error_route_pattern, ExplainErrorHandler),
        (local_ai_status_route_pattern, LocalAIStatusHandler),
        (local_ai_install_route_pattern, LocalAIInstallHandler),
        (local_ai_model_ensure_route_pattern, LocalAIModelEnsureHandler),
        (suggest_fix_route_pattern, SuggestFixHandler),
        (code_help_route_pattern, CodeHelpHandler),
        (validate_fix_route_pattern, ValidateFixHandler),
    ]

    web_app.add_handlers(host_pattern, handlers)
