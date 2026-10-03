"""
Deterministic error analyzer for the ML Platform Error Assistant.

Priority order:
1. Exact pattern matches (sklearn unseen features, shape mismatch, etc.)
2. Error-type specific handling with `is_sufficient=True`
3. Fallback to Ollama for anything not covered (`is_sufficient=False`)
"""
import re


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a sufficient result
# ──────────────────────────────────────────────────────────────────────────────
def _ok(category, summary, details):
    return {"category": category, "summary": summary, "details": details, "is_sufficient": True}


def _needs_ai(category, summary, details):
    return {"category": category, "summary": summary, "details": details, "is_sufficient": False}


# ──────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────────────
def analyze_error(error):
    error_type = error["error_type"]
    message = error["error_message"]
    code = error.get("code", "")
    norm = message.lower()

    # ── SyntaxError ─────────────────────────────────────────────────────────
    if error_type == "SyntaxError":
        details = [
            "Check the reported line for a missing colon, unmatched bracket, or unclosed string.",
        ]
        # Try to extract line number from traceback
        tb = error.get("traceback", "")
        line_match = re.search(r"line (\d+)", tb)
        if line_match:
            details.insert(0, f"Python reported a syntax problem near line {line_match.group(1)}.")
        return _ok("syntax_error", "Python could not parse this cell.", details)

    # ── NameError ───────────────────────────────────────────────────────────
    if error_type == "NameError":
        match = re.search(r"name ['\"](.+?)['\"] is not defined", message)
        if match:
            name = match.group(1)
            # Check whether the name actually appears in the cell code
            if name in code:
                details = [
                    f"'{name}' is referenced but may have been defined in a different kernel "
                    "session or a cell that has not yet been executed.",
                    "Run all preceding cells in order, or define the variable in this cell.",
                ]
            else:
                details = [
                    f"'{name}' is not defined anywhere in this cell.",
                    "Check spelling and confirm the variable, function, or import exists in an "
                    "already-executed cell.",
                ]
            return _ok("name_error", f"The name '{name}' is not defined.", details)
        return _ok(
            "name_error",
            "Python tried to use a name that is not defined.",
            [
                "Check spelling and confirm the variable, function, or import was defined in an "
                "executed cell."
            ],
        )

    # ── AttributeError ──────────────────────────────────────────────────────
    if error_type == "AttributeError":
        attr_match = re.search(r"has no attribute ['\"](.+?)['\"]", message)
        type_match = re.search(r"object has no attribute|'([A-Za-z0-9_]+)' object", message)
        if attr_match:
            attr = attr_match.group(1)
            return _ok(
                "attribute_error",
                f"The object does not have an attribute named '{attr}'.",
                [
                    f"Verify the object type before calling '.{attr}'.",
                    "Common causes: calling a DataFrame method on a Series, or accessing a "
                    "column that was not yet created.",
                ],
            )
        return _ok(
            "attribute_error",
            "Attribute access failed on the object.",
            [
                "Check the object's type and available methods.",
                "A common cause is chaining calls where an intermediate result is None or "
                "a different type than expected.",
            ],
        )

    # ── TypeError ───────────────────────────────────────────────────────────
    if error_type == "TypeError":
        # Unsupported operand
        op_match = re.search(
            r"unsupported operand type\(s\) for (.+?): '(.+?)' and '(.+?)'", message
        )
        if op_match:
            op, t1, t2 = op_match.groups()
            return _ok(
                "type_error_operand",
                f"Cannot apply operator '{op.strip()}' between '{t1}' and '{t2}'.",
                [
                    f"Convert the operands to compatible types before using '{op.strip()}'.",
                    "For example: int(x) + y, or float(x) * y.",
                ],
            )
        # Not callable
        if "is not callable" in norm:
            nc_match = re.search(r"'(.+?)' object is not callable", message)
            name = nc_match.group(1) if nc_match else "object"
            return _ok(
                "type_error_not_callable",
                f"'{name}' is not callable — it is a value, not a function.",
                [
                    "Remove the parentheses if you meant to reference the variable.",
                    "Alternatively, the variable name may shadow an intended function import.",
                ],
            )
        # Wrong number of arguments
        args_match = re.search(r"takes (\d+) positional argument", message)
        if args_match:
            return _ok(
                "type_error_args",
                "A function received the wrong number of arguments.",
                [
                    "Check the function signature for required and optional parameters.",
                    "Use keyword arguments to avoid positional confusion.",
                ],
            )
        return _needs_ai(
            "type_error",
            f"TypeError: {message}",
            ["Review the types of all values passed to the failing operation."],
        )

    # ── KeyError ────────────────────────────────────────────────────────────
    if error_type == "KeyError":
        key = message.strip().strip("'\"")
        details = [
            "For a DataFrame, compare the key with df.columns — watch for leading/trailing "
            "spaces and case differences.",
            "For a dict, use .get(key, default) to avoid KeyError when the key may be absent.",
        ]
        if key:
            # Detect pandas column access pattern
            if f'["{key}"]' in code or f"['{key}']" in code:
                details.insert(0, f"The column or key '{key}' is referenced in this cell but "
                               "was not found in the DataFrame / dict.")
            return _ok("key_error", f"The key '{key}' was not found.", details)
        return _ok("key_error", "A required key was not found.", details)

    # ── IndexError ──────────────────────────────────────────────────────────
    if error_type == "IndexError":
        idx_match = re.search(r"index (\d+) is out of bounds for axis \d+ with size (\d+)", message)
        if idx_match:
            idx, size = idx_match.groups()
            return _ok(
                "index_error",
                f"Index {idx} is out of range (size = {size}).",
                [
                    f"The valid range is 0 to {int(size) - 1}.",
                    "Ensure the index comes from the actual length of the sequence.",
                ],
            )
        if "list index out of range" in norm:
            return _ok(
                "index_error",
                "List index out of range.",
                [
                    "Check the list length with len() before indexing.",
                    "Negative indices count from the end: -1 is the last element.",
                ],
            )
        return _ok(
            "index_error",
            "A sequence index is out of bounds.",
            ["Check the length of the sequence and ensure the index is within range."],
        )

    # ── scikit-learn: unseen feature names (modern pandas-backed scikit-learn) ─
    # e.g. "The feature names should match those that were passed during fit.
    #        Feature names unseen at fit time:\n- salary_per_experience"
    if error_type == "ValueError" and "feature names unseen at fit time" in norm:
        unseen_match = re.findall(r"^-\s*(.+)$", message, re.MULTILINE)
        unseen = [f.strip() for f in unseen_match if f.strip()]
        seen_match = re.findall(r"Feature names seen at fit time, yet now missing:\s*\n((?:-\s*.+\n?)+)", message, re.IGNORECASE)
        missing = []
        if seen_match:
            missing = [f.strip().lstrip("- ") for f in seen_match[0].splitlines() if f.strip()]

        details = []
        if unseen:
            details.append(
                f"Column(s) not seen during training: {', '.join(unseen)}. "
                "Drop them before calling .transform() or .predict()."
            )
        if missing:
            details.append(
                f"Column(s) expected from training but now missing: {', '.join(missing)}. "
                "Ensure the same features are present for prediction."
            )
        if not details:
            details = [
                "The prediction input contains columns that were not present during training.",
                "Use the exact same column names and count as when .fit() was called.",
            ]
        details.append(
            "Fix: pass only the columns used during training to .transform() / .predict(), "
            "e.g. new_df[feature_cols] where feature_cols = X_train.columns.tolist()."
        )
        return _ok(
            "sklearn_unseen_features",
            "Scikit-learn received feature names that were not seen during training.",
            details,
        )

    # ── scikit-learn: feature count mismatch (legacy / NumPy arrays) ────────
    feature_match = re.search(
        r"\bX has\s+(\d+)\s+features?,.*?\bexpect(?:ing|s)?\s+(\d+)\s+features?\b",
        message,
        re.IGNORECASE,
    )
    if feature_match:
        observed, expected = feature_match.groups()
        return _ok(
            "sklearn_feature_mismatch",
            f"The input has {observed} features, but the model expects {expected}.",
            [
                "Use the same feature count and column order for training and prediction.",
                "Store the column list when fitting: feature_cols = X_train.columns.tolist(), "
                "then select them at inference: X_new[feature_cols].",
            ],
        )

    # ── Shape / sample count mismatch ───────────────────────────────────────
    shapes = re.findall(r"\(\s*\d+(?:\s*,\s*\d+)*\s*,?\s*\)", message)
    if len(shapes) >= 2 or "inconsistent numbers of samples" in norm:
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
            shape_details.append(f"Sample counts: {first} vs {second}.")
        shape_details.append(
            "Ensure X and y (or any arrays combined/stacked together) have matching first dimensions."
        )
        return _ok("shape_mismatch", "Array or dataset dimensions do not match.", shape_details)

    # ── ValueError: cannot convert float NaN ────────────────────────────────
    if error_type == "ValueError" and "nan" in norm and "input contains nan" not in norm:
        return _ok(
            "nan_value_error",
            "A NaN (Not a Number) value was encountered where a finite value is required.",
            [
                "Use df.isnull().sum() to locate NaN values in your DataFrame.",
                "Handle them with .dropna(), .fillna(value), or imputation before fitting.",
            ],
        )

    # ── ValueError: could not convert string to float ───────────────────────
    if error_type == "ValueError" and "could not convert string to float" in norm:
        str_match = re.search(r"could not convert string to float: '(.+?)'", message)
        bad_val = f" ('{str_match.group(1)}')" if str_match else ""
        return _ok(
            "string_to_float_error",
            f"A non-numeric string{bad_val} was found where a number is required.",
            [
                "Encode categorical columns before fitting: pd.get_dummies() or LabelEncoder.",
                "Check for stray string values in numeric columns with df.dtypes and "
                "df[col].unique().",
            ],
        )

    # ── ValueError: Input contains NaN ──────────────────────────────────────
    if error_type == "ValueError" and "input contains nan" in norm:
        return _ok(
            "nan_input_error",
            "The model or transformer received NaN values in its input.",
            [
                "Impute or drop NaN values before passing data to fit() or transform().",
                "Use SimpleImputer from sklearn.impute or df.fillna().",
            ],
        )

    # ── ValueError: setting an array element with a sequence ────────────────
    if error_type == "ValueError" and "setting an array element with a sequence" in norm:
        return _ok(
            "ragged_array_error",
            "NumPy received arrays of inconsistent lengths (ragged/jagged array).",
            [
                "Ensure all rows/lists have the same number of elements.",
                "Use dtype=object if you intentionally need variable-length sequences.",
            ],
        )

    # ── ZeroDivisionError ───────────────────────────────────────────────────
    if error_type == "ZeroDivisionError":
        return _ok(
            "zero_division_error",
            "Division by zero occurred.",
            [
                "Guard the division: result = a / b if b != 0 else 0.",
                "Check that the denominator variable is not 0 or an empty array slice.",
            ],
        )

    # ── ImportError / ModuleNotFoundError ───────────────────────────────────
    if error_type in {"ImportError", "ModuleNotFoundError"}:
        mod_match = re.search(r"No module named '(.+?)'", message)
        if mod_match:
            mod = mod_match.group(1).split(".")[0]
            return _ok(
                "missing_module",
                f"The module '{mod}' is not installed.",
                [
                    f"Install it by running: %pip install {mod}",
                    "After installing, restart the kernel and re-run this cell.",
                ],
            )
        return _ok(
            "import_error",
            "An import failed.",
            [
                "Check the module name for typos.",
                "Install it with %pip install <module-name> and restart the kernel.",
            ],
        )

    # ── MemoryError ─────────────────────────────────────────────────────────
    if error_type == "MemoryError":
        return _ok(
            "memory_error",
            "The operation ran out of memory.",
            [
                "Reduce data size: sample with df.sample(frac=0.1) or process in batches.",
                "Free unused variables with del var and gc.collect().",
                "Use chunked or sparse representations for large datasets.",
            ],
        )

    # ── RecursionError ──────────────────────────────────────────────────────
    if error_type == "RecursionError":
        return _ok(
            "recursion_error",
            "Maximum recursion depth exceeded.",
            [
                "Check for a missing base case in a recursive function.",
                "Refactor deep recursion into an iterative loop.",
                "Temporarily increase the limit: import sys; sys.setrecursionlimit(5000).",
            ],
        )

    # ── TimeoutError ────────────────────────────────────────────────────────
    if error_type in {"TimeoutError", "concurrent.futures.TimeoutError"}:
        return _ok(
            "timeout_error",
            "The operation timed out.",
            [
                "Check network connectivity for remote operations.",
                "Increase the timeout parameter, or switch to a faster/local alternative.",
            ],
        )

    # ── OSError / FileNotFoundError / PermissionError ────────────────────────
    if error_type in {"FileNotFoundError", "OSError", "PermissionError"}:
        path_match = re.search(r"No such file or directory: '(.+?)'", message)
        if path_match:
            path = path_match.group(1)
            return _ok(
                "file_not_found",
                f"The file or directory '{path}' was not found.",
                [
                    "Verify the path with os.getcwd() and os.listdir().",
                    "Use a raw string (r'path\\to\\file') or forward slashes on Windows.",
                ],
            )
        if error_type == "PermissionError":
            return _ok(
                "permission_error",
                "Permission denied when accessing a file or resource.",
                [
                    "Check that you have read/write access to the file.",
                    "Avoid writing to system-protected paths.",
                ],
            )
        return _ok(
            "os_error",
            f"OS Error: {message}",
            ["Check that the file path is correct, the file exists, and you have access."],
        )

    # ── RuntimeError ────────────────────────────────────────────────────────
    if error_type == "RuntimeError":
        if "cuda" in norm or "gpu" in norm:
            return _ok(
                "cuda_runtime_error",
                "A CUDA/GPU runtime error occurred.",
                [
                    "Check GPU memory with torch.cuda.memory_summary() or nvidia-smi.",
                    "Move tensors to CPU for debugging: tensor.cpu().",
                    "Set device='cpu' to rule out GPU issues.",
                ],
            )
        return _needs_ai(
            "runtime_error",
            f"RuntimeError: {message}",
            ["Review the traceback for the operation that triggered the runtime failure."],
        )

    # ── General ValueError not caught above ─────────────────────────────────
    if error_type == "ValueError":
        return _needs_ai(
            "value_error",
            f"ValueError: {message}",
            ["Review the values and types passed to the failing function."],
        )

    # ── StopIteration / GeneratorExit ───────────────────────────────────────
    if error_type in {"StopIteration", "GeneratorExit"}:
        return _ok(
            "iterator_exhausted",
            "The iterator was exhausted or closed prematurely.",
            [
                "Convert to a list before iterating multiple times: items = list(gen).",
                "Check that the generator or iterator is not consumed before the loop.",
            ],
        )

    # ── Overflow / FloatingPoint errors ─────────────────────────────────────
    if error_type in {"OverflowError", "FloatingPointError"}:
        return _ok(
            "overflow_error",
            f"{error_type}: a numeric value exceeded representable bounds.",
            [
                "Scale/normalize input values before computations.",
                "Use np.float64 or Python Decimal for high-precision arithmetic.",
            ],
        )

    # ── Catch-all: send to Ollama ────────────────────────────────────────────
    return _needs_ai(
        "unknown",
        f"A {error_type or 'unknown'} error occurred.",
        ["The deterministic analyzer does not have a specific diagnosis for this error type."],
    )