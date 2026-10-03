import json

import pytest

from ml_platform_error_assistant.gemini import (
    GEMINI_API_BASE_URL,
    GEMINI_TIMEOUT_SECONDS,
    suggest_fix_with_gemini,
)


class FakeResponse:
    def __init__(self, code, body):
        self.code = code
        self.body = body


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def fetch(self, request, raise_error=True):
        self.requests.append(request)
        return self.responses.pop(0)


def json_bytes(value):
    return json.dumps(value).encode("utf-8")


@pytest.mark.asyncio
async def test_gemini_fix_keeps_api_key_in_header_and_validates_complete_cell(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "server-secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    fix = {"summary": "Add the expected field.", "candidate_code": "print(1)"}
    client = FakeClient(
        [
            FakeResponse(
                200,
                json_bytes(
                    {"candidates": [{"content": {"parts": [{"text": json.dumps(fix)}]}}]}
                ),
            )
        ]
    )
    error = {
        "cell_id": "private-cell-id",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    result = await suggest_fix_with_gemini(
        error, notebook_context="nearby()", client=client
    )

    assert result == {
        "status": "suggested",
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
        **fix,
    }
    request = client.requests[0]
    assert request.url == f"{GEMINI_API_BASE_URL}/models/gemini-2.5-flash:generateContent"
    assert request.request_timeout == GEMINI_TIMEOUT_SECONDS
    assert request.headers["x-goog-api-key"] == "server-secret"
    assert "server-secret" not in request.url
    request_payload = json.loads(request.body)
    system_instruction = request_payload["systemInstruction"]["parts"][0]["text"]
    assert "never convert a missing identifier to a string" in system_instruction
    prompt_context = json.loads(request_payload["contents"][0]["parts"][0]["text"])
    assert prompt_context["adjacent_notebook_code"] == "nearby()"
    assert "cell_id" not in prompt_context


@pytest.mark.asyncio
async def test_gemini_returns_unconfigured_without_sending_request(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = FakeClient([])
    error = {
        "cell_id": "cell-1",
        "code": "print(1)",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    result = await suggest_fix_with_gemini(error, client=client)

    assert result["status"] == "gemini_unconfigured"
    assert client.requests == []


@pytest.mark.asyncio
async def test_gemini_rejects_invalid_python_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "server-secret")
    client = FakeClient(
        [
            FakeResponse(
                200,
                json_bytes(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": json.dumps(
                                                {
                                                    "summary": "Invalid fix.",
                                                    "candidate_code": "def =",
                                                }
                                            )
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
            )
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": "print(1)",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    result = await suggest_fix_with_gemini(error, client=client)

    assert result["status"] == "invalid_gemini_response"


@pytest.mark.asyncio
async def test_gemini_rejects_partial_cell_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "server-secret")
    code = "\n".join(f"value_{index} = {index}" for index in range(10))
    fix = {"summary": "Fix the value.", "candidate_code": "value_0 = 1"}
    client = FakeClient(
        [
            FakeResponse(
                200,
                json_bytes(
                    {"candidates": [{"content": {"parts": [{"text": json.dumps(fix)}]}}]}
                ),
            )
        ]
    )
    error = {
        "cell_id": "cell-1",
        "code": code,
        "error_type": "NameError",
        "error_message": "name 'value' is not defined",
        "traceback": "Traceback",
    }

    result = await suggest_fix_with_gemini(error, client=client)

    assert result["status"] == "incomplete_gemini_response"
    request_payload = json.loads(client.requests[0].body)
    system_instruction = request_payload["systemInstruction"]["parts"][0]["text"]
    assert "every other clear bug" in system_instruction
    assert "never return only a changed line" in system_instruction