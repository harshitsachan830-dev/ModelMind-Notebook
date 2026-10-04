"""
Dataset Intelligence Backend
Provides: Dataset X-Ray, Smart Preprocessing Adviser 2.0
Ported and adapted from the ModelMind standalone app.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# =========================================================
# SAFE IMPORT – pandas is optional (warn if missing)
# =========================================================

try:
    import pandas as pd  # type: ignore

    _PANDAS_OK = True
except ImportError:  # pragma: no cover
    _PANDAS_OK = False


# =========================================================
# LOADER
# =========================================================

def _require_pandas() -> None:
    if not _PANDAS_OK:
        raise RuntimeError(
            "pandas is not installed in this notebook environment. "
            "Run: pip install pandas openpyxl"
        )


def _safe_json(value: Any) -> Any:
    """Recursively convert non-JSON-serialisable values."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


def load_dataframe(path: Path) -> "pd.DataFrame":
    _require_pandas()
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"Dataset X-Ray supports .csv, .xlsx, .xls, and .json files. "
        f"Received: {path.name}"
    )


# =========================================================
# DATASET X-RAY  (general analysis + preview)
# =========================================================

def analyze_dataset(path: Path) -> dict:
    """Return a rich profile of the uploaded dataset."""
    try:
        df = load_dataframe(path)
    except ValueError as exc:
        return {"supported": False, "message": str(exc)}
    except Exception as exc:
        return {"supported": False, "message": f"Could not load dataset: {exc}"}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    missing_values: dict[str, int] = {
        str(col): int(count)
        for col, count in df.isnull().sum().items()
        if count > 0
    }

    # Preview — first 10 rows, NaN → None
    preview_df = df.head(10).astype(object).where(pd.notnull(df.head(10)), None)
    preview = [
        {str(k): (None if v != v else v) for k, v in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]

    likely_id_cols = [
        str(col)
        for col in df.columns
        if (
            str(col).lower() == "id"
            or str(col).lower().endswith("_id")
            or str(col).lower().startswith("id_")
        )
    ]

    # Column-level statistics
    column_stats: list[dict] = []
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isnull().sum())
        total = len(series)
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))

        stat: dict = {
            "name": str(col),
            "dtype": str(series.dtype),
            "is_numeric": is_numeric,
            "missing_count": missing_count,
            "missing_pct": round(missing_count / total * 100, 2) if total else 0,
            "unique_count": unique_count,
        }

        if is_numeric and len(non_null) > 0:
            num = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(num) > 0:
                stat["min"] = float(num.min())
                stat["max"] = float(num.max())
                stat["mean"] = round(float(num.mean()), 4)
                stat["median"] = float(num.median())
                stat["std"] = round(float(num.std()), 4) if len(num) > 1 else 0.0
                stat["skewness"] = round(float(num.skew()), 4)

        column_stats.append(stat)

    result = {
        "supported": True,
        "filename": path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "numeric_columns": [str(c) for c in numeric_cols],
        "categorical_columns": [str(c) for c in categorical_cols],
        "missing_values": missing_values,
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "likely_id_columns": likely_id_cols,
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "preview": preview,
        "column_stats": column_stats,
    }
    return _safe_json(result)


# =========================================================
# SMART PREPROCESSING ADVISER 2.0
# =========================================================

def _missing_severity(pct: float) -> str:
    if pct == 0:
        return "none"
    if pct < 5:
        return "low"
    if pct < 20:
        return "moderate"
    if pct < 50:
        return "high"
    return "very_high"


def _is_likely_id(col: str, unique_count: int, non_null_count: int) -> bool:
    name = str(col).lower()
    name_based = name == "id" or name.endswith("_id") or name.startswith("id_")
    unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0
    return bool(name_based or (non_null_count >= 20 and unique_ratio >= 0.98))


def _numeric_rec(col: str, series: "pd.Series", level: str) -> dict:
    total = len(series)
    non_null = series.dropna()
    non_null_count = len(non_null)
    missing_count = int(series.isnull().sum())
    missing_pct = missing_count / total * 100 if total else 0

    num = pd.to_numeric(non_null, errors="coerce").dropna()
    skewness = float(num.skew()) if len(num) > 2 else 0.0

    q1 = float(num.quantile(0.25)) if len(num) else 0
    q3 = float(num.quantile(0.75)) if len(num) else 0
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((num < lower) | (num > upper)).sum()) if len(num) else 0
    outlier_pct = outlier_count / len(num) * 100 if len(num) else 0.0
    unique_count = int(non_null.nunique())

    warnings: list[str] = []
    if unique_count <= 1:
        warnings.append("This feature is constant — it may provide no predictive value.")
    if _is_likely_id(col, unique_count, non_null_count):
        warnings.append("This column looks like an ID. Review before using as a model feature.")
    if missing_pct >= 50:
        warnings.append("More than half of values are missing. Reconsider keeping this feature.")

    # Imputation strategy
    if missing_count == 0:
        imputation = "none"
    elif abs(skewness) > 0.5 or outlier_count > 0:
        imputation = "median"
    else:
        imputation = "mean"

    # Scaling strategy
    scaling = "standard_scaler"
    if abs(skewness) > 1 or outlier_pct > 5:
        scaling = "robust_scaler"

    # Build level-specific advice
    if level == "basic":
        if imputation == "mean":
            action = "Fill missing values with the mean."
            code = f'df["{col}"] = df["{col}"].fillna(df["{col}"].mean())'
        elif imputation == "median":
            action = "Fill missing values with the median (skewed/outliers detected)."
            code = f'df["{col}"] = df["{col}"].fillna(df["{col}"].median())'
        else:
            action = "No missing value handling needed."
            code = f"# {col}: no missing values"
        explanation = "ModelMind chose this strategy based on the feature's shape and outliers."

    elif level == "medium":
        if imputation == "mean":
            action = "Mean imputation — feature is roughly symmetric with few outliers."
            code = f'mean_val = df["{col}"].mean()\ndf["{col}"] = df["{col}"].fillna(mean_val)'
        elif imputation == "median":
            action = f"Median imputation — skewness={round(skewness,3)}, IQR outliers={outlier_count}."
            code = f'median_val = df["{col}"].median()\ndf["{col}"] = df["{col}"].fillna(median_val)'
        else:
            action = "No missing value handling needed."
            code = f"# {col}: no missing values"
        explanation = (
            f"Skewness={round(skewness,3)}, outliers={outlier_count} ({round(outlier_pct,1)}%). "
            "Median is preferred for asymmetric distributions."
        )

    else:  # advanced
        strat = "median" if imputation == "median" else "mean"
        if imputation == "none":
            action = "No imputer needed."
            code = f"# {col}: no missing values"
        else:
            action = (
                f'Use SimpleImputer(strategy="{strat}") inside a Pipeline. '
                "Fit only on training data."
            )
            code = (
                "from sklearn.impute import SimpleImputer\n\n"
                f'imputer = SimpleImputer(strategy="{strat}")\n'
                f'X_train[["{col}"]] = imputer.fit_transform(X_train[["{col}"]])\n'
                f'X_test[["{col}"]] = imputer.transform(X_test[["{col}"]])'
            )
        explanation = (
            f"Skewness={round(skewness,3)}, outlier_rate={round(outlier_pct,1)}%. "
            "For production ML, fit preprocessing on training data only."
        )

    return _safe_json({
        "feature": col,
        "feature_type": "numerical",
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 2),
        "missing_severity": _missing_severity(missing_pct),
        "unique_count": unique_count,
        "skewness": round(skewness, 4),
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_pct, 2),
        "imputation_strategy": imputation,
        "scaling_strategy": scaling,
        "recommended_action": action,
        "explanation": explanation,
        "example_code": code,
        "warnings": warnings,
    })


def _categorical_rec(col: str, series: "pd.Series", level: str) -> dict:
    total = len(series)
    non_null = series.dropna()
    non_null_count = len(non_null)
    missing_count = int(series.isnull().sum())
    missing_pct = missing_count / total * 100 if total else 0
    unique_count = int(non_null.nunique())
    unique_ratio = unique_count / non_null_count if non_null_count else 0

    warnings: list[str] = []
    if _is_likely_id(col, unique_count, non_null_count):
        warnings.append("This looks like an identifier — encoding it may mislead the model.")
    if unique_count <= 1:
        warnings.append("Only 1 or 0 distinct categories — likely not useful as a feature.")
    if unique_count >= 50 or (unique_count >= 20 and unique_ratio >= 0.5):
        warnings.append("High cardinality — one-hot encoding will create many columns.")
    if missing_pct >= 50:
        warnings.append("More than half of values are missing.")

    if level == "basic":
        if missing_count > 0:
            action = "Fill missing categories with the most frequent value."
            code = f'df["{col}"] = df["{col}"].fillna(df["{col}"].mode()[0])'
        else:
            action = "No missing value handling needed."
            code = f"# {col}: no missing values"
        explanation = "Categorical features need complete values before most models can use them."

    elif level == "medium":
        if missing_count > 0:
            action = (
                "Compare mode-imputation vs. an explicit 'Unknown' category. "
                "Choose based on what missingness means in this dataset."
            )
            code = (
                f'mode_val = df["{col}"].mode()[0]\n'
                f'df["{col}"] = df["{col}"].fillna(mode_val)'
            )
        else:
            action = "No missing value handling needed."
            code = f"# {col}: no missing values"
        explanation = (
            f"{unique_count} categories, {round(missing_pct,1)}% missing. "
            "Cardinality and the meaning of missingness should guide preprocessing."
        )

    else:  # advanced
        if missing_count > 0:
            action = "Use SimpleImputer + OneHotEncoder inside a ColumnTransformer Pipeline."
            code = (
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.preprocessing import OneHotEncoder\n"
                "from sklearn.pipeline import Pipeline\n\n"
                "cat_pipeline = Pipeline([\n"
                '    ("imputer", SimpleImputer(strategy="most_frequent")),\n'
                '    ("encoder", OneHotEncoder(handle_unknown="ignore")),\n'
                "])"
            )
        else:
            action = "No imputation needed. Use OneHotEncoder inside the training pipeline."
            code = (
                "from sklearn.preprocessing import OneHotEncoder\n\n"
                'encoder = OneHotEncoder(handle_unknown="ignore")'
            )
        explanation = (
            f"Cardinality={unique_count}, unique_ratio={round(unique_ratio,3)}. "
            "Keep imputation and encoding inside a Pipeline for train/test integrity."
        )

    return _safe_json({
        "feature": col,
        "feature_type": "categorical",
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 2),
        "missing_severity": _missing_severity(missing_pct),
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "imputation_strategy": "most_frequent" if missing_count > 0 else "none",
        "recommended_action": action,
        "explanation": explanation,
        "example_code": code,
        "warnings": warnings,
    })


def analyze_preprocessing(path: Path, level: str = "basic") -> dict:
    """Smart Preprocessing Adviser 2.0 — full dataset recommendations."""
    level = level.strip().lower()
    if level not in {"basic", "medium", "advanced"}:
        level = "basic"

    try:
        df = load_dataframe(path)
    except Exception as exc:
        return {"handled": False, "error": str(exc)}

    rows, cols = int(df.shape[0]), int(df.shape[1])
    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())

    recommendations: list[dict] = []
    num_count = cat_count = warn_count = 0

    for col in df.columns:
        series = df[col]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        if is_numeric:
            num_count += 1
            rec = _numeric_rec(str(col), series, level)
        else:
            cat_count += 1
            rec = _categorical_rec(str(col), series, level)
        warn_count += len(rec.get("warnings", []))
        recommendations.append(rec)

    dataset_warnings: list[str] = []
    if duplicate_rows > 0:
        dataset_warnings.append(
            f"{duplicate_rows} duplicate rows found. Review before removing."
        )
    if rows == 0:
        dataset_warnings.append("Dataset contains no rows.")

    return _safe_json({
        "handled": True,
        "source": "modelmind-local",
        "engine": "smart-preprocessing-adviser-2.0",
        "level": level,
        "filename": path.name,
        "rows": rows,
        "columns": cols,
        "numerical_features": num_count,
        "categorical_features": cat_count,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "feature_warning_count": warn_count,
        "dataset_warnings": dataset_warnings,
        "recommendations": recommendations,
        "safety": {
            "dataset_modified": False,
            "automatic_apply": False,
            "message": "ModelMind generated recommendations only. Dataset was NOT modified.",
        },
    })


# =========================================================
# AI DEBUGGER TUTOR (deterministic rule engine)
# =========================================================

_TUTOR_RULES: dict[str, dict] = {

    # ── 1. NameError ─────────────────────────────────────────
    "NameError": {
        "title": "Variable not found",
        "icon": "🔎",
        "basic": {
            "what": "Python tried to use a variable that hasn't been defined yet.",
            "why": "You may have a typo in the variable name, or you haven't run the cell that creates it.",
            "steps": [
                "Check the spelling of the variable name — Python is case-sensitive.",
                "Make sure you ran all cells above this one in order (Kernel → Restart & Run All).",
                "If you deleted the variable with del, redefine it.",
            ],
            "code_tip": "print(dir())  # lists all names currently in scope",
        },
        "medium": {
            "what": "NameError: the identifier is not in the current LEGB scope.",
            "why": "The name was never assigned in this scope, or a prior cell was skipped.",
            "steps": [
                "Run all preceding cells: Kernel → Restart & Run All.",
                "Verify spelling including underscores and capitalisation.",
                "If defined inside a function, it won't be visible outside.",
            ],
            "code_tip": "import sys; print([k for k in dir() if 'your_var' in k.lower()])",
        },
        "advanced": {
            "what": "LEGB scope resolution failed to find the identifier.",
            "why": "Cell execution ordering in non-linear notebook, function scope leakage, or name deleted after kernel restart.",
            "steps": [
                "Restart kernel and run cells sequentially.",
                "Check for conditional assignments that don't execute on every run.",
                "Inspect globals() to confirm scope contents.",
            ],
            "code_tip": "print({k: type(v).__name__ for k, v in globals().items()})",
        },
    },

    # ── 2. KeyError ──────────────────────────────────────────
    "KeyError": {
        "title": "Column or key not found",
        "icon": "🗝️",
        "basic": {
            "what": "Python looked for a column name (or dict key) that doesn't exist.",
            "why": "The column name may be misspelled, have extra spaces, or may have changed.",
            "steps": [
                "Print df.columns to see the real column names.",
                "Check for typos, extra spaces, or wrong capitalisation.",
                "Strip whitespace from all column names with df.columns.str.strip().",
            ],
            "code_tip": "print(df.columns.tolist())  # copy the exact name from here",
        },
        "medium": {
            "what": "KeyError: the requested key is absent from the dict/DataFrame.",
            "why": "Column name mismatch — trailing whitespace, capitalisation, or renamed column.",
            "steps": [
                "Strip whitespace: df.columns = df.columns.str.strip()",
                "Lowercase all columns: df.columns = df.columns.str.lower()",
                "Use df.get(col, default) for optional keys.",
            ],
            "code_tip": "df.columns = df.columns.str.strip().str.lower()\nprint(df.columns.tolist())",
        },
        "advanced": {
            "what": "KeyError during label-based lookup — schema drift.",
            "why": "A preprocessing step dropped or renamed columns, breaking downstream code.",
            "steps": [
                "Validate schema at every pipeline boundary.",
                "Use a ColumnTransformer to track column names explicitly.",
                "Assert required columns before transformation.",
            ],
            "code_tip": "required = ['age', 'salary']\nmissing = [c for c in required if c not in df.columns]\nassert not missing, f'Missing columns: {missing}'",
        },
    },

    # ── 3. ValueError (general + sub-rules handled in analyze_error_for_tutor) ──
    "ValueError": {
        "title": "Wrong value or shape",
        "icon": "⚠️",
        "basic": {
            "what": "A function received a value it can't work with — often NaN, wrong shape, or mixed types.",
            "why": "Common causes: missing values (NaN), mismatched number of rows, or text in a numeric column.",
            "steps": [
                "Check for NaN: df.isnull().sum() — fix with df.fillna(df.mean())",
                "Check shapes: print(X.shape, y.shape) — rows must match.",
                "Check for text in number columns: df.dtypes",
            ],
            "code_tip": "# Fix missing values (Basic)\ndf.fillna(df.mean(numeric_only=True), inplace=True)\nprint(df.isnull().sum())",
        },
        "medium": {
            "what": "ValueError: value violates the constraint expected by the function.",
            "why": "NaN in features, X/y length mismatch, string column passed to sklearn, or wrong label encoding.",
            "steps": [
                "Handle NaN with SimpleImputer (median is more robust than mean).",
                "Encode categorical columns before fitting.",
                "Ensure X and y have the same number of rows.",
            ],
            "code_tip": "from sklearn.impute import SimpleImputer\nimport numpy as np\n\n# M02 — median imputation (robust to outliers)\nimp = SimpleImputer(strategy='median')\nX_imputed = imp.fit_transform(X)\nprint('NaN remaining:', np.isnan(X_imputed).sum())",
        },
        "advanced": {
            "what": "ValueError — input validation failed at the sklearn/numpy API boundary.",
            "why": "Dtype mismatch, NaN/Inf values, or shape constraint violation in the estimator.",
            "steps": [
                "Build a preprocessing Pipeline so transformations are reproducible.",
                "Use ColumnTransformer for mixed numeric/categorical data.",
                "Validate inputs with check_array before fitting.",
            ],
            "code_tip": "from sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\n\nnumeric_pipe = Pipeline([\n    ('imputer', SimpleImputer(strategy='median')),\n    ('scaler', StandardScaler())\n])\ncategorical_pipe = Pipeline([\n    ('imputer', SimpleImputer(strategy='most_frequent')),\n    ('encoder', OneHotEncoder(handle_unknown='ignore'))\n])\npreprocessor = ColumnTransformer([\n    ('num', numeric_pipe, numeric_cols),\n    ('cat', categorical_pipe, categorical_cols)\n])",
        },
    },

    # ── 4. TypeError ─────────────────────────────────────────
    "TypeError": {
        "title": "Wrong type",
        "icon": "🔧",
        "basic": {
            "what": "Python tried to use a value in a way that doesn't match its type.",
            "why": "Common in ML: a string column was passed to a numeric function (like mean/fit).",
            "steps": [
                "Check column types: df.dtypes",
                "Drop non-numeric columns: df.select_dtypes(include='number')",
                "Convert numeric strings: pd.to_numeric(df['col'], errors='coerce')",
            ],
            "code_tip": "# Basic fix — keep only numeric columns\nX = df.select_dtypes(include='number')\nprint(X.dtypes)",
        },
        "medium": {
            "what": "TypeError: type mismatch at a function call boundary.",
            "why": "Object/string columns mixed with numeric, or wrong numpy array type passed to sklearn.",
            "steps": [
                "Convert object columns: pd.to_numeric(df['col'], errors='coerce')",
                "Use LabelEncoder for categorical target columns.",
                "Use .values or .to_numpy() when passing DataFrame to sklearn.",
            ],
            "code_tip": "import pandas as pd\nfrom sklearn.preprocessing import LabelEncoder\n\n# Fix string target column\nle = LabelEncoder()\ny = le.fit_transform(df['target'])\n\n# Fix string feature columns\nfor col in df.select_dtypes('object').columns:\n    df[col] = pd.to_numeric(df[col], errors='coerce')",
        },
        "advanced": {
            "what": "TypeError due to dtype protocol mismatch at an API boundary.",
            "why": "Implicit coercion failed; numpy/pandas passed where a specific protocol was expected.",
            "steps": [
                "Use ColumnTransformer to handle mixed types in a single pipeline.",
                "Cast arrays explicitly at the pipeline boundary.",
                "Use FunctionTransformer for custom type handling.",
            ],
            "code_tip": "from sklearn.preprocessing import FunctionTransformer\nimport numpy as np\n\n# Force float dtype at pipeline entry\ncaster = FunctionTransformer(lambda X: X.astype(np.float64))\n# Use inside a Pipeline:\n# Pipeline([('cast', caster), ('scaler', StandardScaler()), ('model', model)])",
        },
    },

    # ── 5. IndexError ────────────────────────────────────────
    "IndexError": {
        "title": "Index out of range",
        "icon": "📏",
        "basic": {
            "what": "You tried to access a position that doesn't exist in a list or array.",
            "why": "Python lists start at index 0, not 1. The item you're asking for may not exist.",
            "steps": [
                "Check the length: print(len(my_list))",
                "Remember Python starts at 0 — last item is index len-1.",
                "Use negative indices: -1 is the last element.",
            ],
            "code_tip": "print(len(my_list))       # check length\nprint(my_list[:5])         # preview first 5\nprint(my_list[-1])         # safe last element",
        },
        "medium": {
            "what": "IndexError: sequence index out of bounds after data transformation.",
            "why": "Array shape changed after filtering/splitting, or a hardcoded index is now wrong.",
            "steps": [
                "Print array shape before the failing line.",
                "Use .shape[0] instead of hardcoded integers.",
                "Check if a filter/dropna emptied the array.",
            ],
            "code_tip": "print('Shape:', arr.shape)\nprint('Length:', len(arr))\nassert len(arr) > 0, 'Array is empty!'",
        },
        "advanced": {
            "what": "IndexError during numpy/pandas indexing after reshape or boolean masking.",
            "why": "Shape mismatch after train/test split, or a reshape that doesn't match data size.",
            "steps": [
                "Validate shapes at every transformation step.",
                "Avoid integer indices that depend on external state.",
                "Use iloc/loc for DataFrame indexing instead of positional int.",
            ],
            "code_tip": "# Safe reshape\nn = arr.shape[0]\narr_2d = arr.reshape(n, -1)  # -1 infers the second dim\nassert arr_2d.shape[0] == n",
        },
    },

    # ── 6. AttributeError ────────────────────────────────────
    "AttributeError": {
        "title": "Method or attribute not found",
        "icon": "🔩",
        "basic": {
            "what": "You tried to use a method or property that the object doesn't have.",
            "why": "Wrong variable type, typo in method name, or the model hasn't been fit yet.",
            "steps": [
                "Print type(my_object) to see what it actually is.",
                "Make sure you called model.fit() before model.predict().",
                "Check the correct method name in the docs.",
            ],
            "code_tip": "print(type(model))          # check the object type\nprint(hasattr(model, 'predict'))  # check if method exists",
        },
        "medium": {
            "what": "AttributeError: object has no attribute — model not fitted or wrong type.",
            "why": "The object is None, a list, or an unfitted estimator.",
            "steps": [
                "Call model.fit(X_train, y_train) before predict.",
                "Check for None: if model is not None: ...",
                "Verify library version: import sklearn; print(sklearn.__version__)",
            ],
            "code_tip": "from sklearn.utils.validation import check_is_fitted\ntry:\n    check_is_fitted(model)\nexcept Exception as e:\n    print('Model not fitted:', e)\n    model.fit(X_train, y_train)",
        },
        "advanced": {
            "what": "AttributeError: API mismatch between library versions.",
            "why": "Method renamed or removed in a newer/older version of sklearn, pandas, or numpy.",
            "steps": [
                "Pin library versions in requirements.txt.",
                "Use hasattr() checks before calling optional methods.",
                "Read the CHANGELOG for the relevant library version.",
            ],
            "code_tip": "import sklearn, pandas, numpy\nprint(f'sklearn={sklearn.__version__}, pandas={pandas.__version__}, numpy={numpy.__version__}')",
        },
    },

    # ── 7. ModuleNotFoundError ───────────────────────────────
    "ModuleNotFoundError": {
        "title": "Package not installed",
        "icon": "📦",
        "basic": {
            "what": "Python couldn't find the library you tried to import.",
            "why": "The package isn't installed, or the import name is different from the install name.",
            "steps": [
                "Install it in a new cell: !pip install package-name",
                "Note: scikit-learn installs as 'sklearn' (import sklearn, install scikit-learn).",
                "Restart the kernel after installing.",
            ],
            "code_tip": "# Common ML installs\n!pip install scikit-learn pandas numpy matplotlib seaborn xgboost",
        },
        "medium": {
            "what": "ModuleNotFoundError: package not on sys.path.",
            "why": "Wrong Python environment — package installed globally but not in this kernel.",
            "steps": [
                "Check which Python is running: import sys; print(sys.executable)",
                "Install in the correct environment: !{sys.executable} -m pip install package-name",
                "Restart the kernel after installing.",
            ],
            "code_tip": "import sys\nprint(sys.executable)\n!{sys.executable} -m pip install scikit-learn",
        },
        "advanced": {
            "what": "Module resolution failed — package absent or namespace collision.",
            "why": "Virtual environment isolation, wrong kernel, or a local file shadowing the library.",
            "steps": [
                "Verify: !pip show package-name",
                "Confirm kernel matches environment: import sys; sys.executable",
                "Check for local files named the same as the package (e.g., pandas.py).",
            ],
            "code_tip": "import subprocess, sys\nresult = subprocess.run([sys.executable, '-m', 'pip', 'show', 'scikit-learn'], capture_output=True, text=True)\nprint(result.stdout)",
        },
    },

    # ── 8. FileNotFoundError ─────────────────────────────────
    "FileNotFoundError": {
        "title": "File not found",
        "icon": "📂",
        "basic": {
            "what": "Python couldn't find the file at the path you specified.",
            "why": "The file doesn't exist, the name is misspelled, or it's in a different folder.",
            "steps": [
                "List files in current folder: import os; os.listdir('.')",
                "Check the exact filename and extension (case-sensitive on Mac/Linux).",
                "Upload the file to the notebook if it's missing.",
            ],
            "code_tip": "import os\nprint('Working folder:', os.getcwd())\nprint('Files:', os.listdir('.'))",
        },
        "medium": {
            "what": "FileNotFoundError: path doesn't resolve to an existing file.",
            "why": "Relative path resolves to notebook's working directory, not the script's location.",
            "steps": [
                "Use pathlib: from pathlib import Path; list(Path('.').glob('*.csv'))",
                "Use an absolute path or build it from the notebook path.",
                "Confirm file was uploaded successfully.",
            ],
            "code_tip": "from pathlib import Path\ncsv_files = list(Path('.').glob('*.csv'))\nprint('CSV files found:', csv_files)",
        },
        "advanced": {
            "what": "OS-level file open failed — path encoding or cwd mismatch.",
            "why": "Working directory differs from expected (subprocess/pytest changes cwd), or encoding issues.",
            "steps": [
                "Use pathlib for all path manipulation.",
                "Resolve the path: Path(filename).resolve()",
                "Use __file__ to get the script's directory for relative paths.",
            ],
            "code_tip": "from pathlib import Path\np = Path('data.csv').resolve()\nprint('Looking for:', p)\nprint('Exists:', p.exists())",
        },
    },

    # ── 9. ZeroDivisionError ─────────────────────────────────
    "ZeroDivisionError": {
        "title": "Division by zero",
        "icon": "➗",
        "basic": {
            "what": "You tried to divide a number by zero.",
            "why": "The denominator in your calculation ended up being zero.",
            "steps": [
                "Add a guard: if denominator != 0: result = a / denominator",
                "Print the denominator value to understand why it's zero.",
                "Check if your data/dataset is empty.",
            ],
            "code_tip": "# Safe division\nresult = a / b if b != 0 else 0",
        },
        "medium": {
            "what": "ZeroDivisionError: denominator evaluated to zero at runtime.",
            "why": "Empty dataset after filtering, or a metric (like std) with only one sample.",
            "steps": [
                "Check dataset size: len(df) > 0 before computing statistics.",
                "Use np.divide with where= for safe array division.",
                "Handle class imbalance that may cause empty subsets.",
            ],
            "code_tip": "import numpy as np\nnp.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)",
        },
        "advanced": {
            "what": "ZeroDivisionError in a numerical computation pipeline.",
            "why": "Edge case: all samples of a class, zero variance, empty train split.",
            "steps": [
                "Add domain-specific validation before computation.",
                "Use np.errstate to detect and handle float division by zero.",
                "Use StratifiedKFold to prevent empty class splits.",
            ],
            "code_tip": "with np.errstate(divide='ignore', invalid='ignore'):\n    result = np.true_divide(a, b)\n    result[~np.isfinite(result)] = 0  # replace inf/nan with 0",
        },
    },

    # ── 10. SyntaxError ─────────────────────────────────────
    "SyntaxError": {
        "title": "Syntax error",
        "icon": "✏️",
        "basic": {
            "what": "Python couldn't understand your code due to a punctuation or spelling mistake.",
            "why": "A bracket, quote, colon, or comma is missing or in the wrong place.",
            "steps": [
                "Look at the line number in the error message.",
                "Check the line above it too — the error is often one line earlier.",
                "Count opening and closing brackets (, [, { — they must match.",
            ],
            "code_tip": "# Common mistakes:\n# Missing colon:  if x > 0    ← needs ':'\n# Unclosed string: print('hello)  ← missing closing quote",
        },
        "medium": {
            "what": "SyntaxError: Python's parser couldn't tokenise the source.",
            "why": "Unmatched delimiter, wrong indentation, or invalid statement.",
            "steps": [
                "Check reported line AND the line above.",
                "Ensure all brackets/quotes are balanced.",
                "Watch for f-strings with nested quotes of the same type.",
            ],
            "code_tip": "import ast\ntry:\n    ast.parse(your_code_string)\nexcept SyntaxError as e:\n    print(f'Syntax error at line {e.lineno}: {e.msg}')",
        },
        "advanced": {
            "what": "SyntaxError at parse time — grammar invalid for this Python version.",
            "why": "Python version mismatch (e.g., walrus operator := needs Python 3.8+), or smart quotes from web.",
            "steps": [
                "Check Python version: import sys; sys.version",
                "Inspect AST with ast.parse() for the exact node.",
                "Avoid copying code from web pages that use smart quotes (' ' \" \").",
            ],
            "code_tip": "import sys\nprint(sys.version)  # check Python version",
        },
    },

    # ── 11. MemoryError ─────────────────────────────────────
    "MemoryError": {
        "title": "Not enough memory",
        "icon": "💾",
        "basic": {
            "what": "Your computer ran out of RAM while processing the data.",
            "why": "The dataset or model is too large to fit in memory.",
            "steps": [
                "Load only the columns you need: pd.read_csv(file, usecols=['col1','col2'])",
                "Load data in chunks: pd.read_csv(file, chunksize=10000)",
                "Delete unused variables: del large_df; import gc; gc.collect()",
            ],
            "code_tip": "# Load only needed columns\ndf = pd.read_csv('data.csv', usecols=['feature1', 'feature2', 'target'])\nprint(df.shape, df.memory_usage().sum() / 1e6, 'MB')",
        },
        "medium": {
            "what": "MemoryError: system RAM exhausted during data processing.",
            "why": "Large matrix operations, one-hot encoding with high-cardinality columns, or duplicated data.",
            "steps": [
                "Downcast dtypes to save memory: df = df.astype({'int_col': 'int32', 'float_col': 'float32'})",
                "Use sparse matrices for one-hot encoded data.",
                "Use chunked processing instead of loading all at once.",
            ],
            "code_tip": "# Reduce memory usage\nfor col in df.select_dtypes('float64').columns:\n    df[col] = df[col].astype('float32')\nfor col in df.select_dtypes('int64').columns:\n    df[col] = df[col].astype('int32')\nprint('Memory:', df.memory_usage().sum() / 1e6, 'MB')",
        },
        "advanced": {
            "what": "MemoryError: heap exhausted — data doesn't fit in RAM.",
            "why": "Full matrix materialisation, OHE explosion, or no streaming support.",
            "steps": [
                "Use Dask for out-of-core processing.",
                "Use sparse_output=True in OneHotEncoder.",
                "Use memory-mapped arrays (np.memmap) for large datasets.",
            ],
            "code_tip": "from sklearn.preprocessing import OneHotEncoder\nimport scipy.sparse as sp\n\n# Sparse output to avoid memory explosion\nenc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')\nX_encoded = enc.fit_transform(df[categorical_cols])  # sparse matrix",
        },
    },

    # ── 12. RecursionError ───────────────────────────────────
    "RecursionError": {
        "title": "Infinite recursion",
        "icon": "🔄",
        "basic": {
            "what": "A function called itself too many times without stopping.",
            "why": "There's a missing or unreachable base case in a recursive function.",
            "steps": [
                "Make sure your recursive function has a stopping condition (base case).",
                "Add a print statement to trace the recursion depth.",
                "Consider using a loop instead of recursion.",
            ],
            "code_tip": "import sys\nsys.setrecursionlimit(100)  # lower limit to catch bugs faster\n# Better: convert recursion to a while loop",
        },
        "medium": {
            "what": "RecursionError: maximum recursion depth exceeded.",
            "why": "Missing base case or an accidental circular reference.",
            "steps": [
                "Trace the call stack: use a counter to detect infinite loops.",
                "Refactor to iterative using a stack data structure.",
                "Check for circular object references.",
            ],
            "code_tip": "def safe_recurse(n, depth=0, max_depth=100):\n    if depth > max_depth:\n        raise ValueError(f'Recursion too deep at n={n}')\n    if n <= 0:\n        return 0  # base case\n    return n + safe_recurse(n - 1, depth + 1)",
        },
        "advanced": {
            "what": "RecursionError: CPython's call stack limit (default 1000) exceeded.",
            "why": "Deep recursive algorithms or accidental circular references in data structures.",
            "steps": [
                "Refactor to iterative using an explicit stack.",
                "Use tail-call optimisation pattern or trampolining.",
                "Increase limit only as a last resort: sys.setrecursionlimit(5000).",
            ],
            "code_tip": "# Iterative DFS using explicit stack (replaces recursive DFS)\nstack = [root_node]\nwhile stack:\n    node = stack.pop()\n    # process node\n    stack.extend(node.children)",
        },
    },

    # ── 13. StopIteration / GeneratorExit ───────────────────
    "StopIteration": {
        "title": "Iterator exhausted",
        "icon": "🔚",
        "basic": {
            "what": "You tried to get the next item from an empty iterator or generator.",
            "why": "The loop or iterator has already gone through all its items.",
            "steps": [
                "Convert the iterator to a list first: items = list(my_iterator)",
                "Check if the iterator is empty before calling next().",
                "Use a for loop instead of manually calling next().",
            ],
            "code_tip": "# Safe next() with default\nvalue = next(my_iterator, None)  # returns None if exhausted\nif value is not None:\n    print(value)",
        },
        "medium": {
            "what": "StopIteration raised outside of a for-loop context.",
            "why": "Calling next() on an exhausted generator, or a generator function raising StopIteration inside.",
            "steps": [
                "Use next(it, default) instead of bare next(it).",
                "Recreate the iterator/generator before reusing.",
                "Convert to list for multiple passes: data = list(gen)",
            ],
            "code_tip": "# Correct: use default\nfor item in iterator:\n    process(item)\n\n# Or: materialise for reuse\ndata = list(generator_function())",
        },
        "advanced": {
            "what": "StopIteration propagating through a generator (PEP 479 change in Python 3.7+).",
            "why": "In Python 3.7+, StopIteration inside a generator raises RuntimeError instead of stopping silently.",
            "steps": [
                "Replace bare 'raise StopIteration' with 'return' in generators.",
                "Use explicit return statements to end generators.",
                "Check Python version compatibility for generator code.",
            ],
            "code_tip": "def my_gen():\n    for item in source:\n        yield item\n    return  # correct way to end a generator (not raise StopIteration)",
        },
    },
}

_GENERIC_TUTOR = {
    "title": "Python runtime error",
    "icon": "🐛",
    "basic": {
        "what": "Python stopped because it encountered an error it couldn't recover from.",
        "why": "Check the error type and message in the traceback for a specific clue.",
        "steps": [
            "Read the last line of the error message — it's the most specific.",
            "Look at the highlighted line number.",
            "Try printing the values on the failing line to inspect them.",
        ],
        "code_tip": "# Inspect values before the error\nprint(type(x), x)",
    },
    "medium": {
        "what": "An unhandled exception was raised during execution.",
        "why": "The specific error type wasn't matched by the local tutor engine.",
        "steps": [
            "Read the full traceback from top to bottom.",
            "Focus on the innermost frame — that's the actual failure point.",
            "Isolate the failing operation in a separate cell.",
        ],
        "code_tip": "import traceback\ntraceback.print_exc()  # print full traceback from an except block",
    },
    "advanced": {
        "what": "The error wasn't matched by the local tutor rule set.",
        "why": "This may be an uncommon exception or a library-specific error.",
        "steps": [
            "Read the traceback frames carefully for the library/module involved.",
            "Check the library's GitHub issues for similar reports.",
            "Use the Ollama or Gemini assistant for AI-powered analysis.",
        ],
        "code_tip": "# Wrap the failing code to capture details\ntry:\n    failing_operation()\nexcept Exception as e:\n    print(type(e).__name__, e)\n    import traceback; traceback.print_exc()",
    },
}


def analyze_error_for_tutor(
    error_type: str,
    error_message: str,
    traceback: str,
    code: str,
    level: str = "basic",
) -> dict:
    """
    Deterministic AI Debugger Tutor — rule-based, works 100% offline.
    BOUND RESTRICTION: Only suggests code fixes when confidence is >= 70%.
    Otherwise code suggestions are withheld to prevent recommending incorrect fixes.
    """
    level = level.strip().lower()
    if level not in {"basic", "medium", "advanced"}:
        level = "basic"

    import re as _re

    msg_lower = (error_message or "").lower()
    tb_lower = (traceback or "").lower()
    code_lower = (code or "").lower()

    # Extract key failing line from traceback
    failing_line = ""
    for line in reversed((traceback or "").splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("Traceback") and not stripped.startswith("File "):
            failing_line = stripped
            break

    # Default rule lookup
    rule = _TUTOR_RULES.get(error_type, _GENERIC_TUTOR)
    level_content = rule.get(level, rule.get("basic", {}))

    title = rule.get("title", "Runtime Error")
    icon = rule.get("icon", "🐛")
    what = level_content.get("what", "")
    why = level_content.get("why", "")
    steps = list(level_content.get("steps", []))
    code_tip = level_content.get("code_tip", "")
    hints: list[str] = []

    # Confidence calculation (0 - 100)
    confidence = 50
    can_suggest_code = False

    # ── 0. SyntaxError / _IncompleteInputError / IndentationError (Basic syntax) ──
    is_syntax = (
        error_type in {"_IncompleteInputError", "SyntaxError", "IndentationError", "TabError"}
        or "incomplete input" in msg_lower
        or "syntaxerror" in error_type.lower()
        or "was never closed" in msg_lower
        or "unclosed" in msg_lower
        or "invalid syntax" in msg_lower
        or "expected an indented block" in msg_lower
        or "unindent does not match" in msg_lower
    )
    if is_syntax:
        confidence = 96
        can_suggest_code = True

        if error_type in {"IndentationError", "TabError"} or "indent" in msg_lower or "tab" in msg_lower:
            title = "Indentation Error"
            icon = "📏"
            what = "Python relies strictly on consistent indentation (usually 4 spaces) to define code blocks."
            why = "A statement inside a function, loop, condition, or class was not indented, or mixed tabs and spaces."
            hints.append("⚡ Ensure 4-space indentation under def, if, for, while, and try statements.")
            steps = [
                f"Inspect line: {failing_line or 'Check cell'}",
                "Indent code inside blocks by exactly 4 spaces.",
                "Ensure you are not mixing Tab characters with space characters.",
            ]
            code_tip = (
                "# Example proper 4-space indentation:\n"
                "if True:\n"
                "    print('Properly indented line')  # 4 spaces\n"
            )
        else:
            # Unclosed parenthesis, brackets, quotes or missing colon
            title = "Unclosed Parenthesis / Syntax Error"
            icon = "✏️"
            what = "Python encountered incomplete code or a syntax error — an opening bracket, quote, or parenthesis was never closed."
            why = "Every opening '(', '[', '{' or string quote must have a matching closing pair."
            
            # Auto-calculate missing delimiters
            open_p = code.count('(') - code.count(')')
            open_b = code.count('[') - code.count(']')
            open_c = code.count('{') - code.count('}')
            missing_parts = []
            fixed_code = code.rstrip()
            if open_p > 0:
                missing_parts.append(f"{open_p} closing ')'")
                fixed_code += ')' * open_p
            if open_b > 0:
                missing_parts.append(f"{open_b} closing ']'")
                fixed_code += ']' * open_b
            if open_c > 0:
                missing_parts.append(f"{open_c} closing '}}'")
                fixed_code += '}' * open_c

            if missing_parts:
                hints.append(f"⚡ Missing {', '.join(missing_parts)} — close it to complete the statement.")
                steps = [
                    f"Add missing {', '.join(missing_parts)} at the end of the statement.",
                    f"Check line: {failing_line or (code.splitlines()[-1] if code else 'End of cell')}",
                    "Ensure parentheses around function calls (e.g. print(...)) are completely closed.",
                ]
                code_tip = (
                    f"# Corrected code with closing delimiters:\n"
                    f"{fixed_code}"
                )
            elif code.rstrip().endswith(("def", "if", "for", "while", "else", "elif", "try", "except", "finally", "class")):
                hints.append("⚡ Missing colon ':' at the end of compound statement.")
                steps = [
                    "Add ':' at the end of the statement.",
                    "Indent the block underneath by 4 spaces.",
                ]
                code_tip = f"{code.rstrip()}:\n    pass"
            else:
                hints.append("⚡ Check matching pairs of '()', '[]', '{}', and quotes \"\" ''.")
                steps = [
                    f"Inspect the failing line: {failing_line or 'Check cell'}",
                    "Look for unclosed quotes, brackets, or unexpected symbols.",
                    "Ensure statement is syntactically complete.",
                ]
                code_tip = (
                    "# Ensure all syntax is complete and closed:\n"
                    f"{code.rstrip()}"
                )

    # ── 1. ValueError sub-specialization & confidence ──────────────────
    elif error_type == "ValueError":
        # Sub-case A: Matrix multiplication / dimension mismatch (from numpy/pytorch/scipy)
        is_matmul = (
            any(p in msg_lower for p in ["matmul", "core dimension", "gufunc signature", "not aligned"])
            or ("shapes (" in msg_lower and "not aligned" in msg_lower)
            or ("dimension" in msg_lower and any(op in code_lower for op in ["@", "dot", "matmul"]))
        )
        if is_matmul:
            confidence = 94
            can_suggest_code = True
            title = "Matrix Multiplication Shape Mismatch"
            icon = "📐"
            what = (
                "Matrix multiplication failed due to incompatible inner dimensions. "
                "For A @ B, the number of columns in operand 0 must match the number of rows in operand 1."
            )
            why = "Inner dimension mismatch: (N, K) @ (K, M). Currently dimension K does not match between operands."
            hints.append("⚡ Matrix multiplication requires (N, K) @ (K, M) — check A.shape and B.shape.")
            if level == "basic":
                steps = [
                    "Print shapes: print('A.shape:', getattr(A, 'shape', None), 'B.shape:', getattr(B, 'shape', None))",
                    "If operand B dimensions are reversed, transpose it: result = A @ B.T",
                    "If multiplying with a vector, reshape to 2D column vector: B = B.reshape(-1, 1)",
                ]
                code_tip = (
                    "# 1. Check array shapes\n"
                    "print('Operand 0 shape:', getattr(A, 'shape', None))\n"
                    "print('Operand 1 shape:', getattr(B, 'shape', None))\n\n"
                    "# 2. Transpose B if inner dimensions are reversed:\n"
                    "# result = A @ B.T"
                )
            elif level == "medium":
                steps = [
                    "Assert inner dimension equality before performing matrix multiplication.",
                    "Use np.matmul or @ with validated shapes: A.shape[-1] == B.shape[0].",
                    "Dynamically transpose or squeeze dimensions where appropriate.",
                ]
                code_tip = (
                    "import numpy as np\n\n"
                    "# Validate inner dimensions\n"
                    "print(f'A: {A.shape}, B: {B.shape}')\n"
                    "if A.shape[-1] != B.shape[0] and A.shape[-1] == B.shape[-1]:\n"
                    "    result = np.matmul(A, B.T)\n"
                    "else:\n"
                    "    result = np.matmul(A, B)\n"
                    "print('Result shape:', result.shape)"
                )
            else:
                steps = [
                    "Use np.einsum for explicit tensor dimension contraction.",
                    "Guarantees index alignment ('ik,kj->ij') and prevents silent broadcast bugs.",
                    "Assert tensor rank and dimension contracts at model boundaries.",
                ]
                code_tip = (
                    "import numpy as np\n\n"
                    "# Explicit Einstein summation: contracts dimension k\n"
                    "assert A.shape[-1] == B.shape[0], f'Dimension mismatch: {A.shape} vs {B.shape}'\n"
                    "result = np.einsum('ik,kj->ij', A, B)"
                )

        # Sub-case B: Missing / NaN values in features
        elif any(k in msg_lower or k in tb_lower for k in ["nan", "contains nan", "missing value", "infinity or a value too large"]):
            confidence = 92
            can_suggest_code = True
            title = "Missing Values (NaN) in Features"
            icon = "🩹"
            what = "Estimator received feature matrix containing NaN or infinite values."
            why = "Standard scikit-learn models require clean, finite numeric data before fitting."
            hints.append("⚡ NaN values found — fill missing values before fitting.")
            if level == "basic":
                steps = [
                    "Check missing counts: df.isnull().sum()",
                    "Fill numeric missing values with column mean: df.fillna(df.mean(numeric_only=True), inplace=True)",
                    "Or drop rows with missing values: df.dropna(inplace=True)",
                ]
                code_tip = (
                    "# Fix missing values (Basic)\n"
                    "df.fillna(df.mean(numeric_only=True), inplace=True)\n"
                    "print('NaN remaining:', df.isnull().sum().sum())"
                )
            elif level == "medium":
                steps = [
                    "Impute missing values using SimpleImputer (median is outlier-robust).",
                    "Ensure categorical features are encoded separately.",
                ]
                code_tip = (
                    "from sklearn.impute import SimpleImputer\n"
                    "import numpy as np\n\n"
                    "# M02 — median imputation\n"
                    "imp = SimpleImputer(strategy='median')\n"
                    "X_imputed = imp.fit_transform(X)\n"
                    "print('NaN remaining:', np.isnan(X_imputed).sum())"
                )
            else:
                steps = [
                    "Build a reproducible scikit-learn Pipeline with ColumnTransformer.",
                    "Prevents data leakage across train and test splits.",
                ]
                code_tip = (
                    "from sklearn.pipeline import Pipeline\n"
                    "from sklearn.impute import SimpleImputer\n"
                    "from sklearn.preprocessing import StandardScaler\n\n"
                    "num_pipe = Pipeline([\n"
                    "    ('imputer', SimpleImputer(strategy='median')),\n"
                    "    ('scaler', StandardScaler())\n"
                    "])\n"
                    "X_clean = num_pipe.fit_transform(X)"
                )

        # Sub-case C: String to float conversion failure
        elif "could not convert string to float" in msg_lower or "string dtype" in msg_lower:
            confidence = 91
            can_suggest_code = True
            title = "Unencoded String in Numeric Model"
            icon = "🔤"
            what = "A text/string column was passed to a numeric function or model."
            why = "Machine learning algorithms require numeric matrices. Text features must be encoded."
            hints.append("⚡ String column detected — encode with OneHotEncoder or drop before fitting.")
            if level == "basic":
                steps = [
                    "Inspect data types: print(df.dtypes)",
                    "Select only numeric columns for modeling: X = df.select_dtypes(include='number')",
                ]
                code_tip = (
                    "# Select only numeric features\n"
                    "X_numeric = df.select_dtypes(include='number')\n"
                    "print('Numeric columns ready:', list(X_numeric.columns))"
                )
            elif level == "medium":
                steps = [
                    "Encode categorical text columns using LabelEncoder or pd.get_dummies().",
                ]
                code_tip = (
                    "from sklearn.preprocessing import LabelEncoder\n\n"
                    "# Encode categorical columns\n"
                    "for col in df.select_dtypes(include='object').columns:\n"
                    "    df[col] = LabelEncoder().fit_transform(df[col].astype(str))"
                )
            else:
                steps = [
                    "Use ColumnTransformer with OneHotEncoder(handle_unknown='ignore').",
                ]
                code_tip = (
                    "from sklearn.compose import ColumnTransformer\n"
                    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n\n"
                    "preprocessor = ColumnTransformer([\n"
                    "    ('num', StandardScaler(), num_cols),\n"
                    "    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)\n"
                    "])"
                )

        # Sub-case D: Inconsistent sample count
        elif "inconsistent number of samples" in msg_lower:
            confidence = 90
            can_suggest_code = True
            title = "Sample Count Mismatch (X vs y)"
            icon = "⚖️"
            what = "X (features) and y (labels) have different row counts."
            why = "Every row in feature matrix X must correspond to a target label in y."
            hints.append("⚡ Row counts differ — check len(X) vs len(y).")
            steps = [
                "Print row counts: print('X rows:', len(X), 'y rows:', len(y))",
                "Ensure any dropna() or filtering was applied to both X and y simultaneously.",
            ]
            code_tip = (
                "print(f'X samples: {len(X)}, y samples: {len(y)}')\n"
                "assert len(X) == len(y), 'X and y row counts must match!'"
            )

        else:
            # Sub-case E: Unverified / generic ValueError (< 70% threshold!)
            confidence = 42
            can_suggest_code = False
            title = "Unverified ValueError"
            icon = "❓"
            what = (
                "A function received an invalid value or parameter. "
                "The specific root cause could not be verified with high certainty offline."
            )
            why = "ValueErrors can arise from boundary conditions, negative values, or unsupported formats."
            steps = [
                f"Inspect the failing line: {failing_line or 'Check traceback'}",
                "Print input arguments and verify types with print(type(arg), repr(arg)).",
                "Check function documentation for valid ranges and parameters.",
                "Use the AI Assistant (Explain Error or Suggest Fix) for deep contextual diagnosis.",
            ]
            code_tip = ""
            hints.append("⚡ Confidence is below 70% — automatic code fix is withheld to prevent incorrect advice.")

    # ── 2. TypeError confidence ─────────────────────────────────────────
    elif error_type == "TypeError":
        if "reduction 'mean' with string dtype" in msg_lower or ("string" in msg_lower and "mean" in msg_lower):
            confidence = 90
            can_suggest_code = True
            title = "Mean Attempted on String Column"
            hints.append("⚡ Cannot compute mean on strings — use df.select_dtypes(include='number').")
        elif "not supported between" in msg_lower:
            confidence = 88
            can_suggest_code = True
            title = "Type Comparison Error"
            hints.append("⚡ Type comparison mismatch — ensure compared variables share the same type.")
        elif "takes" in msg_lower and "positional argument" in msg_lower:
            confidence = 82
            can_suggest_code = True
            title = "Wrong Function Arguments"
            hints.append("⚡ Check the function signature with help(function_name).")
        else:
            confidence = 45
            can_suggest_code = False
            title = "Unverified TypeError"
            steps = [
                f"Inspect the failing line: {failing_line or 'Check traceback'}",
                "Verify variable types: print([type(x) for x in args])",
                "Code suggestion withheld: confidence is below 70%.",
            ]
            code_tip = ""

    # ── 3. High-certainty standard Python errors (>= 70%) ──────────────
    elif error_type == "KeyError":
        confidence = 88
        can_suggest_code = True
        hints.append("⚡ Run df.columns.tolist() to see exact column names.")

    elif error_type == "NameError":
        confidence = 90
        can_suggest_code = True
        m = _re.search(r"name \'([^\']+)\' is not defined", error_message or "")
        var_name = m.group(1) if m else ""
        if var_name:
            hints.append(f"⚡ Variable '{var_name}' not found — check spelling or run previous cells.")

    elif error_type == "ZeroDivisionError":
        confidence = 95
        can_suggest_code = True
        hints.append("⚡ Zero denominator — add small epsilon: denominator + 1e-8.")

    elif error_type == "ModuleNotFoundError":
        confidence = 92
        can_suggest_code = True
        m = _re.search(r"No module named \'([^\']+)\'", error_message or "")
        pkg = m.group(1).split(".")[0] if m else ""
        if pkg:
            install_map = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "Pillow", "skimage": "scikit-image"}
            install_name = install_map.get(pkg, pkg)
            hints.append(f"⚡ Install: !pip install {install_name} then restart kernel.")

    elif error_type == "FileNotFoundError":
        confidence = 90
        can_suggest_code = True
        hints.append("⚡ Check current working directory: import os; print(os.listdir('.'))")

    elif error_type == "IndexError":
        confidence = 82
        can_suggest_code = True
        hints.append("⚡ Index out of range — check len(sequence) and boundary limits.")

    elif error_type == "AttributeError":
        if "predict" in msg_lower or "fit" in tb_lower or "fitted" in msg_lower:
            confidence = 86
            can_suggest_code = True
            title = "Model Not Fitted Before Prediction"
            hints.append("⚡ Call model.fit(X_train, y_train) before calling model.predict().")
        else:
            confidence = 45
            can_suggest_code = False
            title = "Unverified AttributeError"
            code_tip = ""
            steps = [
                f"Inspect the failing line: {failing_line or 'Check traceback'}",
                "Check available attributes: print(dir(object_name))",
                "Code suggestion withheld: confidence is below 70%.",
            ]

    elif error_type == "MemoryError":
        confidence = 85
        can_suggest_code = True
        hints.append("⚡ Free memory with del df; import gc; gc.collect()")

    else:
        # Generic unhandled
        confidence = 35
        can_suggest_code = False
        code_tip = ""
        steps = [
            f"Inspect the failing line: {failing_line or 'Check traceback'}",
            "Verify inputs and runtime state.",
            "Use the AI Assistant (Explain Error) for deeper diagnosis.",
        ]

    # Enforce strict 70% bound restriction
    if confidence < 70:
        can_suggest_code = False
        code_tip = ""
        confidence_reason = (
            f"Confidence score ({confidence}%) is below the 70% threshold required to recommend code. "
            "Automated code generation is withheld to protect your notebook from incorrect fixes."
        )
    else:
        confidence_reason = f"Confidence score is {confidence}% (High certainty pattern match)."

    return {
        "handled": True,
        "source": "modelmind-tutor-local",
        "error_type": error_type,
        "error_message": error_message,
        "failing_line": failing_line,
        "level": level,
        "title": title,
        "icon": icon,
        "what": what,
        "why": why,
        "steps": steps,
        "code_tip": code_tip,
        "smart_hints": hints,
        "coverage": "80%",
        "confidence": confidence,
        "can_suggest_code": can_suggest_code,
        "confidence_reason": confidence_reason,
        "note": (
            "Confidence is >= 70%: Code fix provided based on high-certainty offline pattern match."
            if can_suggest_code
            else "Confidence is below 70%: Code suggestion withheld to prevent incorrect advice. Follow diagnostic steps or use AI Assistant."
        ),
    }

