from ml_platform_error_assistant.fix_validation import (
    is_complete_cell_fix,
    suggest_rowwise_apply_fix,
    suggest_merge_suffix_fix,
    validate_candidate_fix,
)


def test_complete_cell_fix_rejects_a_single_line_patch_for_a_large_cell():
    original = "\n".join(f"value_{index} = {index}" for index in range(10))

    assert not is_complete_cell_fix(original, "value_0 = 1")
    assert is_complete_cell_fix(original, original.replace("value_0 = 0", "value_0 = 1"))


def test_validate_candidate_fix_accepts_fixed_code():
    result = validate_candidate_fix(
        original_code="x = 1\nprint(y)",
        candidate_code="x = 1\nprint(x)",
        error_type="NameError",
        error_message="name 'y' is not defined",
    )

    assert result["status"] == "validated"
    assert result["passed"] is True
    assert result["error"] is None


def test_validate_candidate_fix_rejects_code_that_still_fails():
    result = validate_candidate_fix(
        original_code="print(x)",
        candidate_code="print(x)",
        error_type="NameError",
        error_message="name 'x' is not defined",
    )

    assert result["status"] == "failed"
    assert result["passed"] is False
    assert "error" in result


def test_rowwise_apply_keyerror_adds_axis_one_to_matching_call():
    code = '''def calculate_score(row):
    return row["Experience"] / 5

df["Score"] = df.apply(calculate_score)
'''

    result = suggest_rowwise_apply_fix(
        {"error_type": "KeyError", "error_message": "'Experience'"}, code
    )

    assert result["status"] == "suggested"
    assert 'df.apply(calculate_score, axis=1)' in result["candidate_code"]


def test_rowwise_apply_fix_does_not_match_other_errors_or_explicit_axis():
    code = '''def calculate_score(row):
    return row["Experience"] / 5

df["Score"] = df.apply(calculate_score, axis=1)
'''

    assert suggest_rowwise_apply_fix(
        {"error_type": "NameError", "error_message": "name 'df' is not defined"},
        code,
    ) is None
    assert suggest_rowwise_apply_fix(
        {"error_type": "KeyError", "error_message": "'Experience'"}, code
    ) is None


def test_merge_suffix_keyerror_uses_actual_right_hand_suffix_name():
    code = '''department_score = df.groupby("Department")["Score"].mean().reset_index()
df = df.merge(department_score, on="Department", suffixes=("", "_Department"))
df["Difference"] = df["Score"] - df["Department_Score"]
'''

    result = suggest_merge_suffix_fix(
        {"error_type": "KeyError", "error_message": "'Department_Score'"}, code
    )

    assert result["status"] == "suggested"
    assert 'df["Score_Department"]' in result["candidate_code"]


def test_merge_suffix_fix_ignores_non_keyerrors_and_unrelated_merge():
    code = '''df = df.merge(other, on="Department", suffixes=("", "_Department"))
print(df["Department_Score"])
'''

    assert suggest_merge_suffix_fix(
        {"error_type": "NameError", "error_message": "Department_Score"}, code
    ) is None
    assert suggest_merge_suffix_fix(
        {"error_type": "KeyError", "error_message": "Department_Score"}, code
    ) is None
