import json
import os

import pytest
from tornado.httpclient import HTTPError

from ml_platform_error_assistant.ollama import (
    OLLAMA_BASE_URL,
    OLLAMA_EXPLAIN_MAX_BODY_BYTES,
    OLLAMA_EXPLAIN_TIMEOUT_SECONDS,
    OLLAMA_STATUS_MAX_BODY_BYTES,
    OLLAMA_STATUS_TIMEOUT_SECONDS,
    check_ollama_status,
    ensure_model_ready,
    explain_code_with_ollama,
    explain_with_ollama,
    load_model_manifest,
    suggest_fix_with_ollama,
)


class FakeResponse:
    def __init__(self, code, body):
        self.code = code
        self.body = body


class FakeClient:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.requests = []
        self.raise_error = None

    async def fetch(self, request, raise_error=True):
        self.requests.append(request)
        self.raise_error = raise_error
        if self.error:
            raise self.error
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_status_checks_ollama_version_with_bounded_request():
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, b'{"models":[]}'),
        ]
    )

    result = await check_ollama_status(client)

    assert result == {
        "status": "connected",
        "service_available": True,
        "version": "0.35.0",
        "model_status": "missing",
        "model_id": "qwen2.5-coder:7b",
        "model_available": False,
    }
    assert client.requests[0].url == f"{OLLAMA_BASE_URL}/api/version"
    assert client.requests[1].url == f"{OLLAMA_BASE_URL}/api/tags"
    assert all(
        request.request_timeout == OLLAMA_STATUS_TIMEOUT_SECONDS
        and request.connect_timeout == 1.0
        for request in client.requests
    )
    assert client.raise_error is False


@pytest.mark.asyncio
async def test_status_reports_recommended_model_when_present():
    model_id = load_model_manifest()["recommended_model"]["id"]
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
        ]
    )

    result = await check_ollama_status(client)

    assert result["status"] == "connected"
    assert result["model_status"] == "available"
    assert result["model_available"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client",
    [
        FakeClient([FakeResponse(503, b"unavailable")]),
        FakeClient([FakeResponse(200, b"not-json")]),
        FakeClient([FakeResponse(200, b"{}")]),
        FakeClient([FakeResponse(200, b"x" * (OLLAMA_STATUS_MAX_BODY_BYTES + 1))]),
        FakeClient(error=HTTPError(599, "connection timed out")),
    ],
)
async def test_unhealthy_or_invalid_ollama_responses_are_unavailable(client):
    assert await check_ollama_status(client) == {
        "status": "unavailable",
        "service_available": False,
        "version": None,
        "model_status": "not_checked",
        "model_id": "qwen2.5-coder:7b",
        "model_available": False,
    }


@pytest.mark.asyncio
async def test_model_list_failure_does_not_claim_model_is_missing():
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(503, b"unavailable"),
        ]
    )

    result = await check_ollama_status(client)

    assert result["status"] == "connected"
    assert result["model_status"] == "unavailable"
    assert result["model_available"] is False


@pytest.mark.asyncio
async def test_explanation_uses_manifest_model_and_validates_structured_response():
    model_id = load_model_manifest()["recommended_model"]["id"]
    explanation = {
        "summary": "The requested name is undefined.",
        "details": ["Define the variable before using it."],
        "confidence": "high",
    }
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
            FakeResponse(
                200,
                json_bytes({"message": {"content": json.dumps(explanation)}}),
            ),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "unknown_name",
        "error_type": "NameError",
        "error_message": "name 'unknown_name' is not defined",
        "traceback": "Traceback (most recent call last)",
    }

    result = await explain_with_ollama(error, client)

    assert result == {
        "status": "explained",
        "model_id": model_id,
        "explanation": explanation,
    }
    request = client.requests[2]
    payload = json.loads(request.body)
    assert request.url == f"{OLLAMA_BASE_URL}/api/chat"
    assert request.request_timeout == OLLAMA_EXPLAIN_TIMEOUT_SECONDS
    assert request.connect_timeout == 2.0
    assert payload["model"] == model_id
    assert payload["stream"] is False
    assert payload["format"] == {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 1_000},
            "details": {
                "type": "array",
                "items": {"type": "string", "maxLength": 500},
                "minItems": 1,
                "maxItems": 5,
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["summary", "details", "confidence"],
        "additionalProperties": False,
    }
    assert len(request.body) < OLLAMA_EXPLAIN_MAX_BODY_BYTES


@pytest.mark.asyncio
async def test_missing_manifest_model_prevents_chat_request():
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, b'{"models":[]}'),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "traceback",
    }

    result = await explain_with_ollama(error, client)

    assert result["status"] == "model_missing"
    assert len(client.requests) == 2


@pytest.mark.asyncio
async def test_oversized_serialized_context_is_not_sent_to_ollama():
    model_id = load_model_manifest()["recommended_model"]["id"]
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "é" * 20_000,
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "traceback",
    }

    result = await explain_with_ollama(error, client)

    assert result["status"] == "context_too_large"
    assert len(client.requests) == 2


@pytest.mark.asyncio
async def test_invalid_structured_model_response_is_rejected():
    model_id = load_model_manifest()["recommended_model"]["id"]
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
            FakeResponse(200, b'{"message":{"content":"not json"}}'),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "traceback",
    }

    result = await explain_with_ollama(error, client)

    assert result["status"] == "invalid_model_response"


@pytest.mark.asyncio
async def test_fix_suggestion_uses_local_model_and_returns_complete_valid_cell():
    model_id = load_model_manifest()["recommended_model"]["id"]
    fix = {
        "summary": "Use the correct pyplot module name.",
        "candidate_code": "import seaborn as sns\nimport matplotlib.pyplot as plt",
    }
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
            FakeResponse(200, json_bytes({"message": {"content": json.dumps(fix)}})),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "import seaborn as sns\\nimport matplotlib.pyplo",
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'matplotlib.pyplo'",
        "traceback": "Traceback (most recent call last)",
    }

    result = await suggest_fix_with_ollama(
        error, client, notebook_context="plt.figure(figsize=(8, 5))"
    )

    assert result == {"status": "suggested", "model_id": model_id, **fix}
    request = client.requests[2]
    payload = json.loads(request.body)
    assert request.url == f"{OLLAMA_BASE_URL}/api/chat"
    assert payload["model"] == model_id
    assert payload["messages"][0]["role"] == "system"
    assert "complete replacement" in payload["messages"][0]["content"]
    assert "never return only a changed line" in payload["messages"][0]["content"]
    model_context = json.loads(payload["messages"][1]["content"])
    assert model_context["adjacent_notebook_code"] == "plt.figure(figsize=(8, 5))"


@pytest.mark.asyncio
async def test_fix_suggestion_rejects_invalid_python_from_local_model():
    model_id = load_model_manifest()["recommended_model"]["id"]
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
            FakeResponse(
                200,
                json_bytes(
                    {
                        "message": {
                            "content": json.dumps(
                                {"summary": "Broken fix", "candidate_code": "import ="}
                            )
                        }
                    }
                ),
            ),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "print(x)",
        "error_type": "NameError",
        "error_message": "name 'x' is not defined",
        "traceback": "Traceback",
    }

    assert (await suggest_fix_with_ollama(error, client))["status"] == "invalid_model_response"


@pytest.mark.asyncio
async def test_fix_suggestion_rejects_partial_cell_from_local_model():
    model_id = load_model_manifest()["recommended_model"]["id"]
    code = "\n".join(f"value_{index} = {index}" for index in range(10))
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
            FakeResponse(
                200,
                json_bytes(
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Fix the value.",
                                    "candidate_code": "value_0 = 1",
                                }
                            )
                        }
                    }
                ),
            ),
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": code,
        "error_type": "NameError",
        "error_message": "name 'value' is not defined",
        "traceback": "Traceback",
    }

    result = await suggest_fix_with_ollama(error, client)

    assert result["status"] == "incomplete_model_response"
    payload = json.loads(client.requests[2].body)
    assert "every other clear bug" in payload["messages"][0]["content"]
    assert "never return only a changed line" in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_code_help_uses_local_model_and_forwards_user_question():
    model_id = load_model_manifest()["recommended_model"]["id"]
    explanation = {
        "summary": "The function normalizes experience and performance into a weighted score.",
        "details": ["Experience contributes 35%.", "Performance contributes 65% of the score."],
        "confidence": "high",
    }
    client = FakeClient(
        [
            FakeResponse(200, b'{"version":"0.35.0"}'),
            FakeResponse(200, json_bytes({"models": [{"name": model_id}]})),
            FakeResponse(
                200,
                json_bytes({"message": {"content": json.dumps(explanation)}}),
            ),
        ]
    )

    result = await explain_code_with_ollama(
        "def score(row): return row['Experience'] / 5",
        "What does this function calculate?",
        notebook_context="df.apply(score, axis=1)",
        client=client,
    )

    assert result == {
        "status": "explained",
        "model_id": model_id,
        "explanation": explanation,
    }
    payload = json.loads(client.requests[2].body)
    user_context = json.loads(payload["messages"][1]["content"])
    assert "What does this function calculate?" == user_context["question"]
    assert user_context["adjacent_notebook_code"] == "df.apply(score, axis=1)"


def test_explanation_validator_rejects_non_string_confidence():
    from ml_platform_error_assistant.ollama import validate_explanation

    with pytest.raises(ValueError, match="confidence"):
        validate_explanation(
            {
                "summary": "A summary.",
                "details": ["One detail."],
                "confidence": [],
            }
        )


@pytest.mark.asyncio
async def test_ensure_model_ready_requires_explicit_install_permission(monkeypatch):
    async def fake_check(client=None):
        return {
            "status": "connected",
            "service_available": True,
            "version": "0.35.0",
            "model_status": "missing",
            "model_id": load_model_manifest()["recommended_model"]["id"],
            "model_available": False,
        }

    monkeypatch.setattr("ml_platform_error_assistant.ollama.check_ollama_status", fake_check)

    result = await ensure_model_ready(allow_download=False)

    assert result["status"] == "manual_required"
    assert result["model_id"] == load_model_manifest()["recommended_model"]["id"]


@pytest.mark.asyncio
async def test_ensure_model_ready_uses_manual_pull_when_allowed(monkeypatch):
    called = {}
    model_id = load_model_manifest()["recommended_model"]["id"]
    checks = iter(
        [
            {
                "status": "connected",
                "service_available": True,
                "version": "0.35.0",
                "model_status": "missing",
                "model_id": model_id,
                "model_available": False,
            },
            {
                "status": "connected",
                "service_available": True,
                "version": "0.35.0",
                "model_status": "available",
                "model_id": model_id,
                "model_available": True,
            },
        ]
    )

    async def fake_check(client=None):
        return next(checks)

    def fake_run(command, capture_output, text, timeout, check=False):
        called["command"] = command
        return type("Result", (), {"returncode": 0, "stdout": "pull ok", "stderr": ""})()

    monkeypatch.setattr("ml_platform_error_assistant.ollama.check_ollama_status", fake_check)
    monkeypatch.setattr("ml_platform_error_assistant.ollama.subprocess.run", fake_run)

    result = await ensure_model_ready(allow_download=True)

    assert result["status"] == "installed"
    assert result["model_id"] == model_id
    assert os.path.basename(called["command"][0]) == "ollama"
    assert called["command"][1] == "pull"


def json_bytes(value):
    return json.dumps(value).encode()