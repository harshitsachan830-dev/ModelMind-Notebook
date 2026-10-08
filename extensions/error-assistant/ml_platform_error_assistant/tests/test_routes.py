import json

from ml_platform_error_assistant import routes


async def test_health(jp_fetch):
    # When
    response = await jp_fetch("api", "health")

    # Then
    assert response.code == 200
    payload = json.loads(response.body)
    assert payload == {
        "status": "ok",
        "local_ai": "setup_pending",
        "cloud_fallback": "disabled",
        "gemini_fallback_available": False,
    }


async def test_dataset_target_analyzes_categorical_branch(jp_fetch, tmp_path):
    dataset_path = tmp_path / "students.csv"
    dataset_path.write_text(
        "branch\nCSE\nCSE\nAI-ML\nCSE-AIML\n",
        encoding="utf-8",
    )

    response = await jp_fetch(
        "api",
        "dataset",
        "target",
        method="POST",
        body=json.dumps({
            "file_path": str(dataset_path),
            "target_column": "branch",
        }),
    )

    assert response.code == 200
    payload = json.loads(response.body)
    assert payload["task"] == "classification"
    assert payload["task_type"] == "Multiclass Classification"
    assert payload["class_counts"] == {"CSE": 2, "AI-ML": 1, "CSE-AIML": 1}
    assert payload["class_percentages"] == {
        "CSE": 50.0,
        "AI-ML": 25.0,
        "CSE-AIML": 25.0,
    }


async def test_dataset_target_detects_numeric_binary_classification(
    jp_fetch, tmp_path
):
    dataset_path = tmp_path / "promotions.csv"
    dataset_path.write_text(
        "promoted\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n",
        encoding="utf-8",
    )

    response = await jp_fetch(
        "api",
        "dataset",
        "target",
        method="POST",
        body=json.dumps({
            "file_path": str(dataset_path),
            "target_column": "promoted",
        }),
    )

    assert response.code == 200
    payload = json.loads(response.body)
    assert payload["task"] == "classification"
    assert payload["task_type"] == "Binary Classification"
    assert payload["class_counts"] == {"0": 5, "1": 5}


def test_model_recommendation_encodes_categorical_target(tmp_path):
    from ml_platform_error_assistant.model_recommender import recommend_models

    dataset_path = tmp_path / "students.csv"
    dataset_path.write_text(
        "hours,branch\n1,CSE\n2,AI-ML\n3,CSE\n4,AI-ML\n"
        "5,CSE\n6,AI-ML\n7,CSE\n8,AI-ML\n"
        "9,CSE\n10,AI-ML\n11,CSE\n12,AI-ML\n"
        "13,CSE\n14,AI-ML\n15,CSE\n16,AI-ML\n"
        "17,CSE\n18,AI-ML\n19,CSE\n20,AI-ML\n",
        encoding="utf-8",
    )

    result = recommend_models(dataset_path, target_col="branch")

    assert result["handled"] is True
    assert result["target_is_categorical"] is True
    assert result["target_encoding"] == "LabelEncoder"
    assert result["models"]
    for model in result["models"]:
        code = model["code"]
        compile(code, f"generated:{model['id']}", "exec")
        assert "target_encoder.fit_transform(y)" in code
        assert "target_encoder.inverse_transform" in code
        assert "Target label encoding:" in code


async def test_error_capture_returns_normalized_payload(jp_fetch):
    error = {
        "cell_id": "cell-17",
        "code": "model.fit(X_train, y_train)",
        "error_type": "ValueError",
        "error_message": "inconsistent numbers of samples",
        "traceback": "Traceback (most recent call last): ...",
    }

    response = await jp_fetch(
        "api",
        "error",
        "analyze",
        method="POST",
        body=json.dumps(error),
    )

    assert response.code == 200
    payload = json.loads(response.body)
    assert payload["status"] == "analyzed"
    assert payload["error"] == error
    assert payload["analysis"]["category"] == "shape_mismatch"
    assert payload["analysis"]["is_sufficient"] is True


async def test_error_capture_rejects_oversized_code(jp_fetch):
    error = {
        "cell_id": "cell-17",
        "code": "x" * 20_001,
        "error_type": "ValueError",
        "error_message": "invalid",
        "traceback": "",
    }

    response = await jp_fetch(
        "api",
        "error",
        "analyze",
        method="POST",
        body=json.dumps(error),
        raise_error=False,
    )

    assert response.code == 400


async def test_local_ai_status_route_returns_service_status(jp_fetch, monkeypatch):
    async def connected_status():
        return {
            "status": "connected",
            "service_available": True,
            "version": "0.35.0",
            "model_status": "missing",
            "model_id": "qwen2.5-coder:7b",
            "model_available": False,
        }

    monkeypatch.setattr(routes, "check_ollama_status", connected_status)
    response = await jp_fetch("api", "local-ai", "status")

    assert response.code == 200
    assert json.loads(response.body)["status"] == "connected"


async def test_local_ai_status_route_requires_jupyter_auth(jp_fetch):
    response = await jp_fetch(
        "api",
        "local-ai",
        "status",
        headers={"Authorization": "token invalid"},
        raise_error=False,
    )

    assert response.code == 403


async def test_local_ai_install_route_requires_manual_approval(jp_fetch, monkeypatch):
    async def fake_status(client=None):
        return {
            "status": "connected",
            "service_available": True,
            "version": "0.35.0",
            "model_status": "missing",
            "model_id": "qwen2.5-coder:7b",
            "model_available": False,
        }

    monkeypatch.setattr("ml_platform_error_assistant.ollama.check_ollama_status", fake_status)
    response = await jp_fetch(
        "api",
        "local-ai",
        "install",
        method="POST",
        body=json.dumps({"allow_download": False}),
    )

    payload = json.loads(response.body)
    assert response.code == 200
    assert payload["status"] == "manual_required"


async def test_explain_route_keeps_sufficient_diagnosis_local(jp_fetch, monkeypatch):
    async def unexpected_ollama_call(_error):
        raise AssertionError("Ollama must not be called for a sufficient diagnosis.")

    monkeypatch.setattr(routes, "explain_with_ollama", unexpected_ollama_call)
    error = {
        "cell_id": "cell-1",
        "code": "unknown_name",
        "error_type": "NameError",
        "error_message": "name 'unknown_name' is not defined",
        "traceback": "Traceback...",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "explain",
        method="POST",
        body=json.dumps(error),
    )

    payload = json.loads(response.body)
    assert response.code == 200
    assert payload["status"] == "deterministic_sufficient"
    assert payload["analysis"]["category"] == "name_error"


async def test_explain_route_calls_local_provider_for_insufficient_diagnosis(
    jp_fetch, monkeypatch
):
    async def local_explanation(error):
        assert error["error_type"] == "CustomError"
        return {
            "status": "model_missing",
            "model_id": "qwen2.5-coder:7b",
        }

    monkeypatch.setattr(routes, "explain_with_ollama", local_explanation)
    error = {
        "cell_id": "cell-1",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback...",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "explain",
        method="POST",
        body=json.dumps(error),
    )

    assert response.code == 200
    assert json.loads(response.body) == {
        "status": "model_missing",
        "model_id": "qwen2.5-coder:7b",
    }


async def test_explain_route_requires_jupyter_auth(jp_fetch):
    response = await jp_fetch(
        "api",
        "ai",
        "explain",
        method="POST",
        body="{}",
        headers={"Authorization": "token invalid"},
        raise_error=False,
    )

    assert response.code == 403


async def test_fix_route_uses_local_ollama_suggestion(jp_fetch, monkeypatch):
    async def local_fix(error, notebook_context=""):
        assert error["error_type"] == "ModuleNotFoundError"
        assert "plt.figure" in notebook_context
        return {
            "status": "suggested",
            "model_id": "qwen2.5-coder:7b",
            "summary": "Correct the pyplot module name.",
            "candidate_code": "import matplotlib.pyplot as plt",
        }

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", local_fix)
    error = {
        "cell_id": "cell-1",
        "code": "import matplotlib.pyplo",
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'matplotlib.pyplo'",
        "traceback": "Traceback",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "fix",
        method="POST",
        body=json.dumps({"error": error, "notebook_context": "plt.figure()"}),
    )

    assert response.code == 200
    assert json.loads(response.body)["candidate_code"] == "import matplotlib.pyplot as plt"


async def test_fix_route_repairs_rowwise_apply_keyerror_before_model_call(
    jp_fetch, monkeypatch
):
    async def unexpected_model_call(*args, **kwargs):
        raise AssertionError("The exact row-wise apply repair should not need a model call.")

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", unexpected_model_call)
    error = {
        "cell_id": "cell-1",
        "code": (
            'def calculate_score(row):\n'
            '    return row["Experience"] / 5\n'
            'df["Score"] = df.apply(calculate_score)'
        ),
        "error_type": "KeyError",
        "error_message": "'Experience'",
        "traceback": "KeyError: 'Experience'",
    }

    response = await jp_fetch(
        "api", "ai", "fix", method="POST", body=json.dumps({"error": error})
    )

    payload = json.loads(response.body)
    assert response.code == 200
    assert payload["status"] == "suggested"
    assert 'df.apply(calculate_score, axis=1)' in payload["candidate_code"]


async def test_fix_route_repairs_merge_suffix_keyerror_before_model_call(
    jp_fetch, monkeypatch
):
    async def unexpected_model_call(*args, **kwargs):
        raise AssertionError("The merge suffix repair should not need a model call.")

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", unexpected_model_call)
    error = {
        "cell_id": "cell-2",
        "code": (
            'department_score = df.groupby("Department")["Score"].mean().reset_index()\n'
            'df = df.merge(department_score, on="Department", suffixes=("", "_Department"))\n'
            'df["Difference"] = df["Score"] - df["Department_Score"]'
        ),
        "error_type": "KeyError",
        "error_message": "'Department_Score'",
        "traceback": "KeyError: 'Department_Score'",
    }

    response = await jp_fetch(
        "api", "ai", "fix", method="POST", body=json.dumps({"error": error})
    )

    payload = json.loads(response.body)
    assert response.code == 200
    assert payload["status"] == "suggested"
    assert 'df["Score_Department"]' in payload["candidate_code"]


async def test_fix_route_does_not_use_gemini_without_user_opt_in(jp_fetch, monkeypatch):
    async def local_failure(error, notebook_context=""):
        return {"status": "model_missing", "model_id": "qwen2.5-coder:7b"}

    async def unexpected_gemini_call(*args, **kwargs):
        raise AssertionError("Gemini must not be called without request opt-in.")

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", local_failure)
    monkeypatch.setattr(routes, "suggest_fix_with_gemini", unexpected_gemini_call)
    monkeypatch.setattr(routes, "gemini_fallback_enabled", lambda: True)
    monkeypatch.setattr(routes, "gemini_api_key_configured", lambda: True)
    error = {
        "cell_id": "cell-3",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    response = await jp_fetch(
        "api", "ai", "fix", method="POST", body=json.dumps({"error": error})
    )

    result = json.loads(response.body)
    assert result["status"] == "model_missing"
    assert result["fallback_used"] is False


async def test_fix_route_requires_global_enablement_before_gemini(jp_fetch, monkeypatch):
    async def local_failure(error, notebook_context=""):
        return {"status": "ollama_unavailable", "model_id": "qwen2.5-coder:7b"}

    async def unexpected_gemini_call(*args, **kwargs):
        raise AssertionError("Gemini must remain disabled by default.")

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", local_failure)
    monkeypatch.setattr(routes, "suggest_fix_with_gemini", unexpected_gemini_call)
    monkeypatch.setattr(routes, "gemini_fallback_enabled", lambda: False)
    error = {
        "cell_id": "cell-4",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "fix",
        method="POST",
        body=json.dumps({"error": error, "allow_gemini_fallback": True}),
    )

    result = json.loads(response.body)
    assert result["status"] == "gemini_disabled"
    assert result["fallback_used"] is False


async def test_fix_route_uses_gemini_after_ollama_failure_with_both_opt_ins(
    jp_fetch, monkeypatch
):
    async def local_failure(error, notebook_context=""):
        return {"status": "invalid_model_response", "model_id": "qwen2.5-coder:7b"}

    async def gemini_fix(error, notebook_context=""):
        assert error["error_type"] == "CustomError"
        assert notebook_context == "nearby()"
        return {
            "status": "suggested",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "summary": "Fix the custom error.",
            "candidate_code": "print(1)",
        }

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", local_failure)
    monkeypatch.setattr(routes, "suggest_fix_with_gemini", gemini_fix)
    monkeypatch.setattr(routes, "gemini_fallback_enabled", lambda: True)
    monkeypatch.setattr(routes, "gemini_api_key_configured", lambda: True)
    error = {
        "cell_id": "cell-5",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "fix",
        method="POST",
        body=json.dumps(
            {
                "error": error,
                "notebook_context": "nearby()",
                "allow_gemini_fallback": True,
            }
        ),
    )

    result = json.loads(response.body)
    assert result["status"] == "suggested"
    assert result["provider"] == "gemini"
    assert result["ollama_status"] == "invalid_model_response"
    assert result["fallback_used"] is True


async def test_fix_route_uses_selected_gemini_without_calling_ollama(jp_fetch, monkeypatch):
    async def unexpected_local_call(*args, **kwargs):
        raise AssertionError("Selected Gemini should not call Ollama first.")

    async def gemini_fix(error, notebook_context=""):
        assert error["error_type"] == "CustomError"
        return {
            "status": "suggested",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "summary": "Fix the custom error.",
            "candidate_code": "print(1)",
        }

    monkeypatch.setattr(routes, "suggest_fix_with_ollama", unexpected_local_call)
    monkeypatch.setattr(routes, "suggest_fix_with_gemini", gemini_fix)
    monkeypatch.setattr(routes, "gemini_fallback_enabled", lambda: True)
    monkeypatch.setattr(routes, "gemini_api_key_configured", lambda: True)
    error = {
        "cell_id": "cell-direct-gemini",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "fix",
        method="POST",
        body=json.dumps(
            {
                "error": error,
                "provider": "gemini",
                "allow_gemini_fallback": True,
            }
        ),
    )

    result = json.loads(response.body)
    assert response.code == 200
    assert result["status"] == "suggested"
    assert result["provider"] == "gemini"
    assert result["fallback_used"] is False


async def test_fix_route_requires_user_opt_in_for_selected_gemini(jp_fetch, monkeypatch):
    async def unexpected_gemini_call(*args, **kwargs):
        raise AssertionError("Gemini must not be called without user opt-in.")

    monkeypatch.setattr(routes, "suggest_fix_with_gemini", unexpected_gemini_call)
    monkeypatch.setattr(routes, "gemini_fallback_enabled", lambda: True)
    monkeypatch.setattr(routes, "gemini_api_key_configured", lambda: True)
    error = {
        "cell_id": "cell-gemini-no-opt-in",
        "code": "custom()",
        "error_type": "CustomError",
        "error_message": "failed",
        "traceback": "Traceback",
    }

    response = await jp_fetch(
        "api",
        "ai",
        "fix",
        method="POST",
        body=json.dumps({"error": error, "provider": "gemini"}),
    )

    result = json.loads(response.body)
    assert response.code == 200
    assert result["status"] == "gemini_opt_in_required"


async def test_code_help_route_calls_local_provider(jp_fetch, monkeypatch):
    async def local_code_help(code, question, notebook_context=""):
        assert "calculate_score" in code
        assert question == "Explain the scoring formula"
        assert "df.apply" in notebook_context
        return {
            "status": "explained",
            "model_id": "qwen2.5-coder:7b",
            "explanation": {
                "summary": "It combines normalized metrics.",
                "details": ["The components are weighted."],
                "confidence": "high",
            },
        }

    monkeypatch.setattr(routes, "explain_code_with_ollama", local_code_help)
    response = await jp_fetch(
        "api",
        "ai",
        "code-help",
        method="POST",
        body=json.dumps(
            {
                "code": "def calculate_score(row): return row['Experience'] / 5",
                "question": "Explain the scoring formula",
                "notebook_context": "df.apply(calculate_score, axis=1)",
            }
        ),
    )

    payload = json.loads(response.body)
    assert response.code == 200
    assert payload["status"] == "explained"
    assert payload["explanation"]["summary"] == "It combines normalized metrics."


async def test_code_help_route_requires_jupyter_auth(jp_fetch):
    response = await jp_fetch(
        "api",
        "ai",
        "code-help",
        method="POST",
        body=json.dumps({"code": "print(1)", "question": "Explain"}),
        headers={"Authorization": "token invalid"},
        raise_error=False,
    )

    assert response.code == 403
