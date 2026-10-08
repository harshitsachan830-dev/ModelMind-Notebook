from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import (
    analyze_feature,
    load_dataset,
)


VALID_LEVELS = {
    "basic": "basic",
    "medium": "medium",
    "advanced": "advanced",
}


def normalize_level(level: str) -> str:
    return VALID_LEVELS.get(
        str(level).strip().lower(),
        "basic",
    )


def _round(value: float) -> float:
    return round(float(value), 4)


def _is_likely_id(
    column: str,
    unique_count: int,
    non_null_count: int,
) -> bool:
    name = str(column).lower()

    name_based = (
        name == "id"
        or name.endswith("_id")
        or name.startswith("id_")
    )

    unique_ratio = (
        unique_count / non_null_count
        if non_null_count > 0
        else 0
    )

    return bool(
        name_based
        or (
            non_null_count >= 20
            and unique_ratio >= 0.98
        )
    )


def _missing_severity(
    percentage: float,
) -> str:
    if percentage == 0:
        return "none"

    if percentage < 5:
        return "low"

    if percentage < 20:
        return "moderate"

    if percentage < 50:
        return "high"

    return "very_high"


# ─────────────────────────────────────────────────────────────────────────────
#  NUMERIC RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_recommendation(
    analysis: dict[str, Any],
    level: str,
) -> dict[str, Any]:

    feature = str(analysis["feature"])
    missing_count = int(analysis.get("missing_count", 0))
    missing_percentage = float(analysis.get("missing_percentage", 0))
    skewness = float(analysis.get("skewness", 0))
    outlier_count = int(analysis.get("outlier_count", 0))
    outlier_percentage = float(analysis.get("outlier_percentage", 0))
    unique_count = int(analysis.get("unique_count", 0))
    non_null_count = int(analysis.get("non_null_count", 0))
    imputation = str(analysis.get("imputation_strategy", "none"))
    scaling = str(analysis.get("scaling_strategy", "model_dependent"))
    transformation = str(analysis.get("transformation_strategy", "none"))

    warnings: list[str] = []

    if unique_count <= 1:
        warnings.append(
            "This feature is constant or nearly empty "
            "and may provide little predictive information."
        )

    if _is_likely_id(feature, unique_count, non_null_count):
        warnings.append(
            "This column behaves like an identifier. "
            "Review it before using it as a model feature."
        )

    if missing_percentage >= 50:
        warnings.append(
            "More than half of this feature is missing. "
            "Consider whether the feature should be kept "
            "before choosing an imputation strategy."
        )

    # ── BASIC LEVEL ─────────────────────────────────────────────────────────
    # Focus: Fill missing values only — simple one-liner pandas code
    if level == "basic":
        if imputation == "mean":
            missing_action = (
                "Fill missing values with the column mean."
            )
            missing_code = (
                f'# Step 1: Fill missing values in "{feature}" with the mean\n'
                f'df["{feature}"] = df["{feature}"].fillna(df["{feature}"].mean())\n'
                f'\n'
                f'# Verify no missing values remain\n'
                f'print(f"Missing values remaining: {{df[\"{feature}\"].isnull().sum()}}")'
            )
            explanation = (
                f'"{feature}" has {missing_count} missing value(s) '
                f"({_round(missing_percentage)}%). "
                "At the Basic level, we simply fill each gap with the average "
                "value of all non-missing rows. This is the easiest way to "
                "make your dataset ready for most ML models."
            )
            why = (
                "The mean is the arithmetic average of all values. "
                "It is a safe, quick default when data is roughly symmetric "
                "and you are just getting started."
            )
            risks = (
                "Mean imputation can underestimate variability and may be "
                "pulled by extreme outliers. If your data has many outliers, "
                "consider switching to Median level."
            )

        elif imputation == "median":
            missing_action = (
                "Fill missing values with the column median."
            )
            missing_code = (
                f'# Step 1: Fill missing values in "{feature}" with the median\n'
                f'df["{feature}"] = df["{feature}"].fillna(df["{feature}"].median())\n'
                f'\n'
                f'# Verify no missing values remain\n'
                f'print(f"Missing values remaining: {{df[\"{feature}\"].isnull().sum()}}")'
            )
            explanation = (
                f'"{feature}" has {missing_count} missing value(s) '
                f"({_round(missing_percentage)}%). "
                "At the Basic level, we fill each gap with the middle value "
                "(median) of the column, which is more robust to extreme numbers."
            )
            why = (
                "The median is the middle value when data is sorted. "
                "It is less affected by very large or very small numbers "
                "than the mean."
            )
            risks = (
                "Median imputation still reduces the true spread of data. "
                "For production models, consider a proper imputer fitted "
                "only on training data."
            )

        else:
            missing_action = (
                "No missing values found — no filling is needed."
            )
            missing_code = (
                f'# "{feature}" has no missing values — nothing to fill!\n'
                f'print(f"Missing: {{df[\"{feature}\"].isnull().sum()}}")  # Should print 0'
            )
            explanation = (
                f'"{feature}" is complete with no missing values. '
                "At the Basic level, no action is required for this column."
            )
            why = "The column is already complete."
            risks = "No risks — the column needs no imputation."

    # ── MEDIUM LEVEL ─────────────────────────────────────────────────────────
    # Focus: Fill NA + outlier detection + scaling suggestion
    elif level == "medium":
        strat_label = "mean" if imputation == "mean" else "median"

        if imputation in ("mean", "median"):
            val_expr = (
                f'df["{feature}"].mean()'
                if imputation == "mean"
                else f'df["{feature}"].median()'
            )
            missing_action = (
                f"Fill missing values with the {strat_label}, then "
                "detect and cap outliers using the IQR method."
            )
            missing_code = (
                f'import numpy as np\n'
                f'\n'
                f'# ── Step 1: Impute missing values with the {strat_label} ──\n'
                f'{strat_label}_val = {val_expr}\n'
                f'df["{feature}"] = df["{feature}"].fillna({strat_label}_val)\n'
                f'\n'
                f'# ── Step 2: Detect outliers using IQR method ──\n'
                f'Q1 = df["{feature}"].quantile(0.25)\n'
                f'Q3 = df["{feature}"].quantile(0.75)\n'
                f'IQR = Q3 - Q1\n'
                f'lower_bound = Q1 - 1.5 * IQR\n'
                f'upper_bound = Q3 + 1.5 * IQR\n'
                f'outliers = df["{feature}"][(df["{feature}"] < lower_bound) | (df["{feature}"] > upper_bound)]\n'
                f'print(f"Outliers detected: {{len(outliers)}}")\n'
                f'\n'
                f'# ── Step 3: Cap outliers (Winsorization) ──\n'
                f'df["{feature}"] = df["{feature}"].clip(lower=lower_bound, upper=upper_bound)\n'
                f'print(f"Range after capping: {{df[\"{feature}\"].min():.2f}} to {{df[\"{feature}\"].max():.2f}}")'
            )
            explanation = (
                f"Skewness = {_round(skewness)} and "
                f"IQR outliers = {outlier_count} "
                f"({_round(outlier_percentage)}%). "
                "These statistics guided the choice of imputation strategy. "
                "After filling missing values, the IQR method caps extreme "
                "values so they don't distort model training."
            )
            why = (
                f"Skewness of {_round(skewness)} indicates the distribution is "
                f"{'symmetric enough for mean' if imputation == 'mean' else 'skewed, making median more robust'}. "
                f"There are {outlier_count} outliers "
                f"({_round(outlier_percentage)}% of values) detected via IQR. "
                "Capping them prevents extreme values from dominating model weights."
            )
            risks = (
                "Capping changes the actual data values. "
                "Always apply the same bounds (Q1, Q3 from training data) "
                "to your test set — never recompute bounds on test data."
            )

        else:
            missing_action = (
                "No missing values to fill. Check for outliers and "
                "consider applying feature scaling for ML readiness."
            )
            missing_code = (
                f'# ── Step 1: No missing values in "{feature}" ──\n'
                f'print(f"Missing: {{df[\"{feature}\"].isnull().sum()}}")  # Should print 0\n'
                f'\n'
                f'# ── Step 2: Check for outliers using IQR ──\n'
                f'Q1 = df["{feature}"].quantile(0.25)\n'
                f'Q3 = df["{feature}"].quantile(0.75)\n'
                f'IQR = Q3 - Q1\n'
                f'lower_bound = Q1 - 1.5 * IQR\n'
                f'upper_bound = Q3 + 1.5 * IQR\n'
                f'outliers = df["{feature}"][(df["{feature}"] < lower_bound) | (df["{feature}"] > upper_bound)]\n'
                f'print(f"Outliers in {feature}: {{len(outliers)}}")\n'
                f'\n'
                f'# ── Step 3: Descriptive statistics ──\n'
                f'print(df["{feature}"].describe())'
            )
            explanation = (
                f'"{feature}" has no missing values. '
                f"Skewness = {_round(skewness)}, outliers = {outlier_count} "
                f"({_round(outlier_percentage)}%). "
                "At Medium level, we check for outliers even on complete columns "
                "because outliers can degrade model accuracy."
            )
            why = "A complete column still benefits from outlier review."
            risks = (
                "Ignoring outliers can distort distance-based models "
                "(e.g., KNN, SVM) and linear regression coefficients."
            )

    # ── ADVANCED LEVEL ───────────────────────────────────────────────────────
    # Focus: Full sklearn Pipeline with ColumnTransformer, StandardScaler, imputer
    else:
        strategy = "mean" if imputation == "mean" else "median"

        if imputation == "none":
            missing_action = (
                "No imputation needed. Add this feature to a numeric "
                "ColumnTransformer pipeline with StandardScaler."
            )
            missing_code = (
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.compose import ColumnTransformer\n"
                "\n"
                f'# "{feature}" has no missing values — only scaling is needed\n'
                "numeric_pipeline = Pipeline(steps=[\n"
                '    ("scaler", StandardScaler()),\n'
                "])\n"
                "\n"
                "preprocessor = ColumnTransformer(transformers=[\n"
                f'    ("num", numeric_pipeline, ["{feature}"]),\n'
                "], remainder='passthrough')\n"
                "\n"
                "# Fit on training data only!\n"
                "X_train_transformed = preprocessor.fit_transform(X_train)\n"
                "X_test_transformed  = preprocessor.transform(X_test)"
            )
        else:
            missing_action = (
                f"Use SimpleImputer(strategy='{strategy}') + StandardScaler "
                "inside a ColumnTransformer pipeline. Fit on training data only."
            )
            missing_code = (
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.preprocessing import StandardScaler, RobustScaler\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.compose import ColumnTransformer\n"
                "\n"
                f'# Advanced pipeline for numerical feature: "{feature}"\n'
                "numeric_pipeline = Pipeline(steps=[\n"
                f'    ("imputer", SimpleImputer(strategy="{strategy}")),\n'
                '    ("scaler", StandardScaler()),      # Use RobustScaler() if outliers are present\n'
                "])\n"
                "\n"
                "# Combine with other columns in a ColumnTransformer\n"
                "preprocessor = ColumnTransformer(transformers=[\n"
                f'    ("num", numeric_pipeline, ["{feature}"]),\n'
                "    # Add more transformers for categorical columns here\n"
                "], remainder='passthrough')\n"
                "\n"
                "# ── Fit ONLY on training data ──\n"
                "X_train_transformed = preprocessor.fit_transform(X_train)\n"
                "X_test_transformed  = preprocessor.transform(X_test)\n"
                "\n"
                "# ── Full ML Pipeline (impute → scale → model) ──\n"
                "from sklearn.linear_model import LinearRegression\n"
                "full_pipeline = Pipeline(steps=[\n"
                '    ("preprocessor", preprocessor),\n'
                '    ("model", LinearRegression()),\n'
                "])\n"
                "full_pipeline.fit(X_train, y_train)\n"
                "score = full_pipeline.score(X_test, y_test)\n"
                'print(f"R² score: {score:.4f}")'
            )

        explanation = (
            f"Skewness = {_round(skewness)}, "
            f"outlier rate = {_round(outlier_percentage)}%. "
            "Advanced level wraps all preprocessing in a scikit-learn Pipeline "
            "inside a ColumnTransformer. This guarantees the imputer and scaler "
            "are fitted on training data only and reused consistently for "
            "validation and test data — preventing data leakage."
        )
        why = (
            "A Pipeline prevents data leakage by ensuring that statistics "
            "(mean, std, etc.) are computed only from training data. "
            "StandardScaler normalises features to zero mean and unit variance, "
            "which benefits gradient-based and distance-based models. "
            "RobustScaler is preferred when outliers are significant "
            f"(current outlier rate: {_round(outlier_percentage)}%)."
        )
        risks = (
            "Fitting the pipeline on the full dataset (instead of only "
            "training data) causes data leakage, inflating evaluation scores. "
            "Always use fit_transform() on X_train and transform() on X_test."
        )

    return {
        "feature": feature,
        "feature_type": "numerical",
        "dtype": analysis.get("dtype"),
        "missing_count": missing_count,
        "missing_pct": _round(missing_percentage),
        "missing_percentage": _round(missing_percentage),
        "missing_severity": _missing_severity(missing_percentage),
        "unique_count": unique_count,
        "skewness": _round(skewness),
        "distribution_shape": analysis.get("distribution_shape"),
        "outlier_count": outlier_count,
        "outlier_percentage": _round(outlier_percentage),
        "outlier_pct": _round(outlier_percentage),
        "imputation_strategy": imputation,
        "imputation_reason": analysis.get("imputation_reason"),
        "scaling_strategy": scaling,
        "scaling_reason": analysis.get("scaling_reason"),
        "transformation_strategy": transformation,
        "transformation_reason": analysis.get("transformation_reason"),
        "recommended_action": missing_action,
        "why": why,
        "risks": risks,
        "explanation": explanation,
        "example_code": missing_code,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CATEGORICAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _categorical_recommendation(
    analysis: dict[str, Any],
    level: str,
) -> dict[str, Any]:

    feature = str(analysis["feature"])
    missing_count = int(analysis.get("missing_count", 0))
    missing_percentage = float(analysis.get("missing_percentage", 0))
    unique_count = int(analysis.get("unique_count", 0))
    non_null_count = int(analysis.get("non_null_count", 0))

    unique_ratio = (
        unique_count / non_null_count
        if non_null_count > 0
        else 0
    )

    warnings: list[str] = []

    likely_id = _is_likely_id(feature, unique_count, non_null_count)

    if likely_id:
        warnings.append(
            "This categorical feature behaves like an "
            "identifier. Encoding identifiers can create "
            "misleading patterns."
        )

    if unique_count <= 1:
        warnings.append(
            "This feature has one or fewer observed "
            "categories and may not provide useful "
            "predictive information."
        )

    if unique_count >= 50 or (
        unique_count >= 20
        and unique_ratio >= 0.5
    ):
        warnings.append(
            "This feature has high cardinality. "
            "One-hot encoding may create many columns. "
            "Consider ordinal encoding or target encoding."
        )

    if missing_percentage >= 50:
        warnings.append(
            "More than half of this feature is missing. "
            "Review whether the feature is useful before "
            "automatically filling the missing values."
        )

    # Determine encoding hint
    is_high_cardinality = unique_count >= 10
    use_ordinal_hint = is_high_cardinality

    # ── BASIC LEVEL ─────────────────────────────────────────────────────────
    # Focus: Fill missing values only — simple fillna with mode
    if level == "basic":
        if missing_count > 0:
            action = (
                "Fill missing categories with the most frequent value (mode)."
            )
            code = (
                f'# Step 1: Find the most common category in "{feature}"\n'
                f'most_common = df["{feature}"].mode()[0]\n'
                f'print(f"Most common value: {{most_common}}")\n'
                f'\n'
                f'# Step 2: Fill missing values with the most common category\n'
                f'df["{feature}"] = df["{feature}"].fillna(most_common)\n'
                f'\n'
                f'# Step 3: Verify no missing values remain\n'
                f'print(f"Missing remaining: {{df[\"{feature}\"].isnull().sum()}}")'
            )
            explanation = (
                f'"{feature}" has {missing_count} missing value(s) '
                f"({_round(missing_percentage)}%). "
                "At the Basic level, we simply replace each empty cell with "
                "the category that appears most often — this is the safest "
                "beginner-friendly approach."
            )
            why = (
                "The mode (most frequent category) is the simplest and most "
                "intuitive way to fill missing text/category values. "
                "It keeps the feature usable without introducing new categories."
            )
            risks = (
                "Using the mode may over-represent one category and reduce "
                "the diversity of your data. If the missing values have a "
                "special meaning, consider using 'Unknown' instead."
            )

        else:
            action = "No missing values found — no filling is needed."
            code = (
                f'# "{feature}" has no missing values — nothing to fill!\n'
                f'print(f"Unique categories: {{df[\"{feature}\"].nunique()}}")\n'
                f'print(df["{feature}"].value_counts())'
            )
            explanation = (
                f'"{feature}" is complete with {unique_count} unique categories. '
                "At the Basic level, no filling is required."
            )
            why = "The column is already complete."
            risks = "No risks — the column needs no imputation at this level."

    # ── MEDIUM LEVEL ─────────────────────────────────────────────────────────
    # Focus: Fill NA + label encoding / one-hot encoding with pandas
    elif level == "medium":
        if missing_count > 0:
            action = (
                "Fill missing values with mode or 'Unknown', then apply "
                "one-hot encoding (low cardinality) or label encoding (high cardinality)."
            )
            if not use_ordinal_hint:
                code = (
                    f'# ── Step 1: Fill missing values ──\n'
                    f'mode_value = df["{feature}"].mode()[0]\n'
                    f'df["{feature}"] = df["{feature}"].fillna(mode_value)\n'
                    f'# Alternative: use "Unknown" if missingness has meaning\n'
                    f'# df["{feature}"] = df["{feature}"].fillna("Unknown")\n'
                    f'\n'
                    f'# ── Step 2: One-Hot Encoding (for low cardinality) ──\n'
                    f'# Creates one binary column per category\n'
                    f'df = pd.get_dummies(df, columns=["{feature}"], drop_first=True)\n'
                    f'print(f"Columns after encoding: {{list(df.columns)}}")'
                )
            else:
                code = (
                    f'# ── Step 1: Fill missing values ──\n'
                    f'mode_value = df["{feature}"].mode()[0]\n'
                    f'df["{feature}"] = df["{feature}"].fillna(mode_value)\n'
                    f'\n'
                    f'# ── Step 2: Label Encoding (for high cardinality = {unique_count} categories) ──\n'
                    f'# Converts each category to an integer\n'
                    f'df["{feature}_encoded"] = df["{feature}"].astype("category").cat.codes\n'
                    f'print(df[["{feature}", "{feature}_encoded"]].drop_duplicates().head(10))'
                )
            explanation = (
                f'"{feature}" has {unique_count} unique categories and '
                f"{_round(missing_percentage)}% missing values. "
                "At Medium level, we fill the gaps then encode categories "
                "into numbers so ML algorithms can process them. "
                f"{'Low cardinality → one-hot encoding.' if not use_ordinal_hint else f'High cardinality ({unique_count} categories) → label encoding.'}"
            )
            why = (
                "ML models require numeric inputs. "
                f"{'One-hot encoding creates a binary column per category — great for low-cardinality features.' if not use_ordinal_hint else 'Label encoding is efficient for high-cardinality features but implies an ordinal relationship.'}"
            )
            risks = (
                "One-hot encoding on high-cardinality features creates too many columns (curse of dimensionality). "
                "Label encoding implies ordinal order which may not exist. "
                "For production code, fit encoders only on training data."
            )

        else:
            action = (
                "No missing values. Apply one-hot or label encoding "
                "to convert categories to numbers."
            )
            if not use_ordinal_hint:
                code = (
                    f'# No missing values — apply One-Hot Encoding directly\n'
                    f'df = pd.get_dummies(df, columns=["{feature}"], drop_first=True)\n'
                    f'print(f"Columns after encoding: {{list(df.columns)}}")'
                )
            else:
                code = (
                    f'# No missing values — apply Label Encoding (high cardinality: {unique_count} categories)\n'
                    f'df["{feature}_encoded"] = df["{feature}"].astype("category").cat.codes\n'
                    f'print(df[["{feature}", "{feature}_encoded"]].drop_duplicates().head(10))'
                )
            explanation = (
                f'"{feature}" has {unique_count} categories and no missing values. '
                "At Medium level, we encode categories as numbers for ML compatibility."
            )
            why = "ML models need numeric inputs. Encoding transforms categories to numbers."
            risks = (
                "Encoding applied to the full dataset (not just training data) "
                "can cause data leakage. Use sklearn encoders in a pipeline for production."
            )

    # ── ADVANCED LEVEL ───────────────────────────────────────────────────────
    # Focus: sklearn Pipeline + ColumnTransformer + OneHotEncoder/OrdinalEncoder
    else:
        if unique_count <= 15 or not use_ordinal_hint:
            # Low cardinality → OneHotEncoder
            encoder_name = "OneHotEncoder"
            encoder_args = 'handle_unknown="ignore", sparse_output=False'
            encoder_note = "OneHotEncoder is best for low-cardinality features (≤15 categories)."
        else:
            # High cardinality → OrdinalEncoder
            encoder_name = "OrdinalEncoder"
            encoder_args = 'handle_unknown="use_encoded_value", unknown_value=-1'
            encoder_note = f"OrdinalEncoder is used here due to high cardinality ({unique_count} categories)."

        if missing_count > 0:
            action = (
                f"Use SimpleImputer + {encoder_name} inside a ColumnTransformer pipeline. "
                "Fit preprocessing on training data only."
            )
            code = (
                "from sklearn.impute import SimpleImputer\n"
                f"from sklearn.preprocessing import {encoder_name}\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.compose import ColumnTransformer\n"
                "\n"
                f'# Advanced categorical pipeline for: "{feature}"\n'
                f"# {encoder_note}\n"
                "categorical_pipeline = Pipeline(steps=[\n"
                '    ("imputer", SimpleImputer(strategy="most_frequent")),\n'
                f'    ("encoder", {encoder_name}({encoder_args})),\n'
                "])\n"
                "\n"
                "# Combine with numerical features in a ColumnTransformer\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "\n"
                "preprocessor = ColumnTransformer(transformers=[\n"
                f'    ("cat", categorical_pipeline, ["{feature}"]),\n'
                "    # Add numerical pipeline here:\n"
                "    # ('num', numeric_pipeline, numerical_cols),\n"
                "], remainder='passthrough')\n"
                "\n"
                "# ── Fit ONLY on training data ──\n"
                "X_train_transformed = preprocessor.fit_transform(X_train)\n"
                "X_test_transformed  = preprocessor.transform(X_test)\n"
                "\n"
                "# ── Full Pipeline: impute → encode → train model ──\n"
                "from sklearn.ensemble import RandomForestClassifier\n"
                "full_pipeline = Pipeline(steps=[\n"
                '    ("preprocessor", preprocessor),\n'
                '    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),\n'
                "])\n"
                "full_pipeline.fit(X_train, y_train)\n"
                'print(f"Accuracy: {full_pipeline.score(X_test, y_test):.4f}")'
            )

        else:
            action = (
                f"No imputation needed. Use {encoder_name} inside a "
                "ColumnTransformer for production-safe encoding."
            )
            code = (
                f"from sklearn.preprocessing import {encoder_name}, StandardScaler\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.compose import ColumnTransformer\n"
                "\n"
                f'# Advanced encoding for: "{feature}" (no missing values)\n'
                f"# {encoder_note}\n"
                "\n"
                "# ── Categorical sub-pipeline ──\n"
                "categorical_pipeline = Pipeline(steps=[\n"
                f'    ("encoder", {encoder_name}({encoder_args})),\n'
                "])\n"
                "\n"
                "# ── Numerical sub-pipeline (adjust column list as needed) ──\n"
                "numerical_pipeline = Pipeline(steps=[\n"
                '    ("scaler", StandardScaler()),\n'
                "])\n"
                "\n"
                "# ── ColumnTransformer combines both ──\n"
                "preprocessor = ColumnTransformer(transformers=[\n"
                f'    ("cat", categorical_pipeline, ["{feature}"]),\n'
                "    # ('num', numerical_pipeline, ['col1', 'col2', ...]),\n"
                "], remainder='passthrough')\n"
                "\n"
                "# ── Fit ONLY on training data ──\n"
                "X_train_transformed = preprocessor.fit_transform(X_train)\n"
                "X_test_transformed  = preprocessor.transform(X_test)"
            )

        explanation = (
            f"Cardinality = {unique_count} categories; unique ratio = "
            f"{_round(unique_ratio)}. {encoder_note} "
            "Advanced level wraps all preprocessing in a scikit-learn Pipeline "
            "inside a ColumnTransformer, guaranteeing the encoder is fitted "
            "only on training data and applied consistently to test data. "
            "This is the production-standard approach."
        )
        why = (
            f"{encoder_name} is chosen based on cardinality ({unique_count} categories). "
            "Wrapping it in a Pipeline with a ColumnTransformer ensures no data "
            "leakage: statistics are learned from training data only and "
            "reapplied to unseen data without re-fitting."
        )
        risks = (
            "Fitting encoders on the full dataset before the train/test split "
            "causes data leakage. Always split first, then fit the pipeline "
            "on X_train only. Use transform() (not fit_transform()) on X_test."
        )

    return {
        "feature": feature,
        "feature_type": "categorical",
        "dtype": analysis.get("dtype"),
        "missing_count": missing_count,
        "missing_pct": _round(missing_percentage),
        "missing_percentage": _round(missing_percentage),
        "missing_severity": _missing_severity(missing_percentage),
        "unique_count": unique_count,
        "unique_ratio": _round(unique_ratio),
        "imputation_strategy": (
            "none" if missing_count == 0 else "most_frequent_or_unknown"
        ),
        "recommended_action": action,
        "why": why,
        "risks": risks,
        "explanation": explanation,
        "example_code": code,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FULL-DATASET ADVANCED PIPELINE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _build_advanced_pipeline_summary(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> str:
    """Generate a suggested full ColumnTransformer pipeline for all columns."""
    num_list = repr(numeric_cols)
    cat_list = repr(categorical_cols)

    return (
        "# ═══════════════════════════════════════════════════════\n"
        "# 🚀 Suggested Full Pipeline (ColumnTransformer)\n"
        "# ═══════════════════════════════════════════════════════\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        f"numerical_cols   = {num_list}\n"
        f"categorical_cols = {cat_list}\n"
        "\n"
        "# ── Numerical sub-pipeline ──\n"
        "numeric_pipeline = Pipeline(steps=[\n"
        '    ("imputer", SimpleImputer(strategy="median")),\n'
        '    ("scaler",  StandardScaler()),\n'
        "])\n"
        "\n"
        "# ── Categorical sub-pipeline ──\n"
        "categorical_pipeline = Pipeline(steps=[\n"
        '    ("imputer", SimpleImputer(strategy="most_frequent")),\n'
        '    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),\n'
        "])\n"
        "\n"
        "# ── ColumnTransformer ──\n"
        "preprocessor = ColumnTransformer(transformers=[\n"
        '    ("num", numeric_pipeline,   numerical_cols),\n'
        '    ("cat", categorical_pipeline, categorical_cols),\n'
        "], remainder='drop')\n"
        "\n"
        "# ── Train / Test Split ──\n"
        "X = df.drop(columns=['target'])   # replace 'target' with your label column\n"
        "y = df['target']\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42\n"
        ")\n"
        "\n"
        "# ── Fit preprocessor on training data only ──\n"
        "X_train_transformed = preprocessor.fit_transform(X_train)\n"
        "X_test_transformed  = preprocessor.transform(X_test)\n"
        'print(f"Train shape: {X_train_transformed.shape}")\n'
        'print(f"Test shape:  {X_test_transformed.shape}")'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_preprocessing(
    path: Path,
    level: str = "Basic",
) -> dict[str, Any]:

    normalized_level = normalize_level(level)
    df = load_dataset(path)

    rows = int(df.shape[0])
    columns = int(df.shape[1])
    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())

    recommendations: list[dict[str, Any]] = []
    numerical_count = 0
    categorical_count = 0
    warning_count = 0
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for column in df.columns:
        feature_analysis = analyze_feature(path, str(column))

        if feature_analysis.get("is_numeric"):
            numerical_count += 1
            numeric_cols.append(str(column))
            recommendation = _numeric_recommendation(feature_analysis, normalized_level)
        else:
            categorical_count += 1
            categorical_cols.append(str(column))
            recommendation = _categorical_recommendation(feature_analysis, normalized_level)

        warning_count += len(recommendation["warnings"])
        recommendations.append(recommendation)

    dataset_warnings: list[str] = []

    if duplicate_rows > 0:
        dataset_warnings.append(
            f"{duplicate_rows} duplicate rows were "
            "detected. Review whether they represent "
            "real repeated observations before removing them."
        )

    if rows == 0:
        dataset_warnings.append("The dataset contains no rows.")

    if columns == 0:
        dataset_warnings.append("The dataset contains no columns.")

    # For advanced level, attach a full pipeline suggestion
    pipeline_code: str | None = None
    if normalized_level == "advanced":
        pipeline_code = _build_advanced_pipeline_summary(
            df, numeric_cols, categorical_cols
        )

    return {
        "handled": True,
        "source": "modelmind-local",
        "engine": "preprocessing-advisor",
        "level": normalized_level,
        "filename": path.name,
        "rows": rows,
        "columns": columns,
        "numerical_features": numerical_count,
        "categorical_features": categorical_count,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "feature_warning_count": warning_count,
        "dataset_warnings": dataset_warnings,
        "recommendations": recommendations,
        "pipeline_code": pipeline_code,
        "safety": {
            "dataset_modified": False,
            "automatic_apply": False,
            "message": (
                "ModelMind generated recommendations only. "
                "The dataset was not modified."
            ),
        },
    }