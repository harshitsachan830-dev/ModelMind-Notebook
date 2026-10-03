import re


def analyze_error(error):
    error_type = error["error_type"]
    message = error["error_message"]
    normalized_message = message.lower()

    if error_type == "SyntaxError":
        return {
            "category": "syntax_error",
            "summary": "Python could not parse this cell.",
            "details": [
                "Check the reported line for a missing colon, unmatched bracket, or unclosed string."
            ],
            "is_sufficient": True,
        }

    if error_type == "NameError":
        match = re.search(r"name ['\"](.+?)['\"] is not defined", message)
        if match:
            summary = f"The name '{match.group(1)}' is not defined."
        else:
            summary = "Python tried to use a name that is not defined."
        return {
            "category": "name_error",
            "summary": summary,
            "details": [
                "Check the spelling and confirm the variable, function, or import was defined in an executed cell."
            ],
            "is_sufficient": True,
        }

    if error_type == "KeyError":
        key = message.strip().strip("'\"")
        summary = f"The requested key '{key}' was not found." if key else "The requested key was not found."
        return {
            "category": "key_error",
            "summary": summary,
            "details": [
                "For a DataFrame lookup, compare the requested label with the actual columns, including spaces and capitalization."
            ],
            "is_sufficient": True,
        }

    feature_match = re.search(
        r"\bX has\s+(\d+)\s+features?,.*?\bexpect(?:ing|s)?\s+(\d+)\s+features?\b",
        message,
        re.IGNORECASE,
    )
    if feature_match:
        observed, expected = feature_match.groups()
        return {
            "category": "sklearn_feature_mismatch",
            "summary": f"The input has {observed} features, but the model expects {expected}.",
            "details": [
                "Use the same feature count and column order for training and prediction."
            ],
            "is_sufficient": True,
        }

    shapes = re.findall(r"\(\s*\d+(?:\s*,\s*\d+)*\s*,?\s*\)", message)
    if len(shapes) >= 2 or "inconsistent numbers of samples" in normalized_message:
        shape_details = []
        if len(shapes) >= 2:
            shape_details.append(f"Reported shapes: {shapes[0]} and {shapes[1]}.")
        sample_match = re.search(
            r"inconsistent numbers of samples[^\d]*(\d+)[^\d]+(\d+)",
            message,
            re.IGNORECASE,
        )
        if sample_match:
            first, second = sample_match.groups()
            shape_details.append(f"Reported sample counts: {first} and {second}.")
        shape_details.append(
            "Check that the dimensions being combined or passed together are compatible."
        )
        return {
            "category": "shape_mismatch",
            "summary": "The array or dataset dimensions do not match.",
            "details": shape_details,
            "is_sufficient": True,
        }

    if error_type in {"ValueError", "IndexError", "TypeError", "ImportError", "ModuleNotFoundError"}:
        return {
            "category": "python_error",
            "summary": f"{error_type}: {message}" if message else f"A {error_type} occurred.",
            "details": ["Review the relevant traceback frame and the values passed to the failing operation."],
            "is_sufficient": False,
        }

    return {
        "category": "unknown",
        "summary": f"A {error_type or 'unknown'} error occurred.",
        "details": ["The deterministic analyzer does not yet have a specific diagnosis for this error."],
        "is_sufficient": False,
    }