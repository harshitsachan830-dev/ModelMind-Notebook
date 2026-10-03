import pytest

from ml_platform_error_assistant.diagnostics import analyze_error


@pytest.mark.parametrize(
    ("error_type", "error_message", "category", "summary_part"),
    [
        ("SyntaxError", "invalid syntax", "syntax_error", "could not parse"),
        ("NameError", "name 'features' is not defined", "name_error", "features"),
        ("KeyError", "'missing_column'", "key_error", "missing_column"),
        (
            "ValueError",
            "operands could not be broadcast together with shapes (2, 3) (4, 3)",
            "shape_mismatch",
            "dimensions do not match",
        ),
        (
            "ValueError",
            "X has 8 features, but LinearRegression is expecting 10 features as input",
            "sklearn_feature_mismatch",
            "8 features",
        ),
    ],
)
def test_known_errors_have_deterministic_diagnoses(
    error_type, error_message, category, summary_part
):
    result = analyze_error(
        {
            "error_type": error_type,
            "error_message": error_message,
        }
    )

    assert result["category"] == category
    assert summary_part in result["summary"]
    assert result["details"]
    assert result["is_sufficient"] is True


def test_unknown_error_is_marked_insufficient():
    result = analyze_error({"error_type": "CustomError", "error_message": "failed"})

    assert result["category"] == "unknown"
    assert result["is_sufficient"] is False