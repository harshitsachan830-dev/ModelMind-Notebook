import ast
import difflib
import re

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _parse_code(code, label):
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"{label} contains invalid Python: {exc.msg}.") from exc


def is_complete_cell_fix(original_code, candidate_code):
    try:
        original_statements = len(ast.parse(original_code).body)
    except SyntaxError:
        original_statements = sum(
            bool(line.strip()) and not line.lstrip().startswith("#")
            for line in original_code.splitlines()
        )
    try:
        candidate_statements = len(ast.parse(candidate_code).body)
    except SyntaxError:
        return False

    minimum_statements = (original_statements + 1) // 2
    return candidate_statements >= minimum_statements


def _safe_exec(code):
    _parse_code(code, "Candidate code")
    namespace = {"__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<sandbox>", "exec"), namespace, namespace)
    return namespace


def normalize_fix_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("The request body must be a JSON object.")

    original_code = (
        payload.get("original_code")
        or payload.get("code")
        or payload.get("cell_code")
        or payload.get("source_code")
    )
    candidate_code = (
        payload.get("candidate_code")
        or payload.get("patched_code")
        or payload.get("proposed_fix")
        or payload.get("fix")
    )

    if candidate_code is not None and isinstance(candidate_code, dict):
        candidate_code = candidate_code.get("code") or candidate_code.get("patched_code")
    if isinstance(candidate_code, str):
        candidate_code = candidate_code.strip()
    if isinstance(original_code, str):
        original_code = original_code.strip()

    if original_code is None or not isinstance(original_code, str) or not original_code.strip():
        raise ValueError("The original code is required for validation.")
    if candidate_code is None or not isinstance(candidate_code, str) or not candidate_code.strip():
        raise ValueError("A candidate fix is required for validation.")

    return {"original_code": original_code, "candidate_code": candidate_code}


def validate_candidate_fix(original_code, candidate_code, error_type=None, error_message=None):
    _parse_code(original_code, "Original code")
    _parse_code(candidate_code, "Candidate code")

    try:
        _safe_exec(original_code)
        original_error = None
    except Exception as exc:  # pragma: no cover - exercised via route and tests
        original_error = f"{type(exc).__name__}: {exc}"

    try:
        _safe_exec(candidate_code)
        candidate_error = None
    except Exception as exc:  # pragma: no cover - exercised via route and tests
        candidate_error = f"{type(exc).__name__}: {exc}"

    if candidate_error is None:
        return {
            "status": "validated",
            "passed": True,
            "message": "The candidate fix executes successfully in the sandbox.",
            "error": None,
        }

    return {
        "status": "failed",
        "passed": False,
        "message": "The candidate fix still raises an error in the sandbox.",
        "error": candidate_error,
        "original_error": original_error,
    }


def suggest_fix_for_error(error, code):
    if not isinstance(error, dict):
        raise ValueError("The error payload must be a dictionary.")

    error_type = str(error.get("error_type", "")).strip()
    error_message = str(error.get("error_message", "")).strip()
    code = str(code or "").strip()

    if error_type == "NameError":
        match = re.search(r"name ['\"](.+?)['\"] is not defined", error_message)
        missing_name = match.group(1) if match else "missing_name"
        if missing_name in code:
            patch = code
        else:
            patch = f"{missing_name} = None\n{code}"
        return {
            "status": "suggested",
            "summary": f"Define or correct the missing name '{missing_name}' before it is used.",
            "diff": "\n".join(difflib.unified_diff(code.splitlines(), patch.splitlines(), fromfile="before", tofile="after", lineterm="")),
            "candidate_code": patch,
        }

    if error_type in {"TypeError", "ValueError"}:
        return {
            "status": "suggested",
            "summary": "Check the types and conversions used in the failing expression before retrying.",
            "diff": "",
            "candidate_code": code,
        }

    return {
        "status": "suggested",
        "summary": "Review the failing line and validate the relevant inputs before applying a patch.",
        "diff": "",
        "candidate_code": code,
    }


def suggest_rowwise_apply_fix(error, code):
    if not isinstance(error, dict) or error.get("error_type") != "KeyError":
        return None

    missing_column = str(error.get("error_message", "")).strip().strip("'\"")
    if not missing_column or not isinstance(code, str) or not code.strip():
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    row_functions = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        row_parameters = {argument.arg for argument in node.args.args}
        if not row_parameters:
            continue
        if any(
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Name)
            and child.value.id in row_parameters
            and isinstance(child.slice, ast.Constant)
            and child.slice.value == missing_column
            for child in ast.walk(node)
        ):
            row_functions.add(node.name)

    matching_calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in row_functions
            and not any(keyword.arg == "axis" for keyword in node.keywords)
        ):
            continue
        matching_calls.append(node)

    if len(matching_calls) != 1:
        return None

    call = matching_calls[0]
    lines = code.splitlines(keepends=True)
    start = sum(len(line.encode("utf-8")) for line in lines[: call.lineno - 1]) + call.col_offset
    end = sum(len(line.encode("utf-8")) for line in lines[: call.end_lineno - 1]) + call.end_col_offset
    encoded = code.encode("utf-8")
    closing_paren = end - 1
    if encoded[closing_paren : closing_paren + 1] != b")":
        return None

    call_prefix = encoded[start:closing_paren]
    insertion = start + len(call_prefix.rstrip())
    separator = b" axis=1" if call_prefix.rstrip().endswith(b",") else b", axis=1"
    candidate = (encoded[:insertion] + separator + encoded[insertion:]).decode("utf-8")
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None

    return {
        "status": "suggested",
        "summary": "This function reads row fields, so apply it row-wise with axis=1.",
        "diff": "",
        "candidate_code": candidate,
    }

def suggest_merge_suffix_fix(error, code):
    if not isinstance(error, dict) or error.get("error_type") != "KeyError":
        return None

    missing_column = str(error.get("error_message", "")).strip().strip("'\"")
    if not missing_column or not isinstance(code, str) or not code.strip():
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    missing_column_is_accessed = any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == missing_column
        for node in ast.walk(tree)
    )
    if not missing_column_is_accessed:
        return None

    candidates = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "merge"
        ):
            continue
        suffixes = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "suffixes"),
            None,
        )
        if not isinstance(suffixes, (ast.Tuple, ast.List)) or len(suffixes.elts) != 2:
            continue
        suffix_values = [
            item.value if isinstance(item, ast.Constant) else None
            for item in suffixes.elts
        ]
        for suffix in suffix_values:
            if not isinstance(suffix, str) or not suffix.startswith("_"):
                continue
            reversed_prefix = suffix[1:] + "_"
            if missing_column.startswith(reversed_prefix):
                base_name = missing_column[len(reversed_prefix) :]
                if base_name and any(
                    isinstance(item, ast.Constant) and item.value == base_name
                    for item in ast.walk(tree)
                ):
                    candidates.add(base_name + suffix)

    if len(candidates) != 1:
        return None

    actual_column = candidates.pop()
    double_quoted_missing = f'"{missing_column}"'
    single_quoted_missing = f"'{missing_column}'"
    if double_quoted_missing in code:
        candidate_code = code.replace(
            double_quoted_missing, f'"{actual_column}"'
        )
    elif single_quoted_missing in code:
        candidate_code = code.replace(
            single_quoted_missing, f"'{actual_column}'"
        )
    else:
        return None

    try:
        ast.parse(candidate_code)
    except SyntaxError:
        return None

    return {
        "status": "suggested",
        "summary": f"The merge suffix creates the column '{actual_column}'.",
        "diff": "",
        "candidate_code": candidate_code,
    }
