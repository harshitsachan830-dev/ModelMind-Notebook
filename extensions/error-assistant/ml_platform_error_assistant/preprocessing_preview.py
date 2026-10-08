from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import (
    analyze_feature,
    load_dataset,
)


SUPPORTED_NUMERICAL_STRATEGIES = {
    "mean",
    "median",
}

SUPPORTED_CATEGORICAL_STRATEGIES = {
    "most_frequent",
    "unknown",
}


def _round(value: Any) -> float | None:
    """
    Safely round a numerical value.
    """
    if value is None:
        return None

    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _safe_value(value: Any) -> Any:
    """
    Convert pandas / numpy values into JSON-safe values.
    """
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _sample_values(
    series: pd.Series,
    limit: int = 8,
) -> list[Any]:
    """
    Return a small JSON-safe sample.
    """
    values = series.head(limit).tolist()

    return [
        _safe_value(value)
        for value in values
    ]


def _numeric_stats(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate before/after statistics for a numerical feature.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    non_null = numeric.dropna()

    if non_null.empty:
        return {
            "missing_count": int(
                numeric.isna().sum()
            ),
            "missing_percentage": (
                round(
                    float(
                        numeric.isna().mean()
                        * 100
                    ),
                    4,
                )
                if len(numeric) > 0
                else 0.0
            ),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "missing_count": int(
            numeric.isna().sum()
        ),
        "missing_percentage": round(
            float(
                numeric.isna().mean()
                * 100
            ),
            4,
        ),
        "mean": _round(
            non_null.mean()
        ),
        "median": _round(
            non_null.median()
        ),
        "std": _round(
            non_null.std()
        ),
        "min": _round(
            non_null.min()
        ),
        "max": _round(
            non_null.max()
        ),
    }


def _categorical_stats(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate before/after statistics for a categorical feature.
    """
    missing_count = int(
        series.isna().sum()
    )

    missing_percentage = (
        round(
            float(
                series.isna().mean()
                * 100
            ),
            4,
        )
        if len(series) > 0
        else 0.0
    )

    non_null = series.dropna()

    mode_value = None

    if not non_null.empty:
        modes = non_null.mode()

        if not modes.empty:
            mode_value = _safe_value(
                modes.iloc[0]
            )

    return {
        "missing_count":
            missing_count,
        "missing_percentage":
            missing_percentage,
        "unique_count": int(
            non_null.nunique()
        ),
        "most_frequent_value":
            mode_value,
    }


def _preview_mean(
    series: pd.Series,
) -> tuple[pd.Series, Any]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    value = numeric.mean()

    if pd.isna(value):
        raise ValueError(
            "Mean imputation cannot be previewed "
            "because the feature has no usable "
            "numerical values."
        )

    return (
        numeric.fillna(value),
        _safe_value(value),
    )


def _preview_median(
    series: pd.Series,
) -> tuple[pd.Series, Any]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    value = numeric.median()

    if pd.isna(value):
        raise ValueError(
            "Median imputation cannot be previewed "
            "because the feature has no usable "
            "numerical values."
        )

    return (
        numeric.fillna(value),
        _safe_value(value),
    )


def _preview_most_frequent(
    series: pd.Series,
) -> tuple[pd.Series, Any]:
    modes = series.dropna().mode()

    if modes.empty:
        raise ValueError(
            "Most-frequent imputation cannot be "
            "previewed because the feature has "
            "no observed values."
        )

    value = modes.iloc[0]

    return (
        series.fillna(value),
        _safe_value(value),
    )


def _preview_unknown(
    series: pd.Series,
) -> tuple[pd.Series, str]:
    value = "Unknown"

    return (
        series.fillna(value),
        value,
    )


def preview_preprocessing(
    path: Path,
    feature: str,
    strategy: str,
) -> dict[str, Any]:
    """
    Preview a preprocessing operation.

    IMPORTANT:
    This function NEVER writes the transformed
    dataframe back to disk.

    The returned "after" values exist only in memory.
    """

    df = load_dataset(path)

    if feature not in df.columns:
        raise ValueError(
            f'Feature "{feature}" was not found '
            "in the dataset."
        )

    normalized_strategy = (
        str(strategy)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    analysis = analyze_feature(
        path,
        feature,
    )

    is_numeric = bool(
        analysis.get(
            "is_numeric",
            False,
        )
    )

    original_series = (
        df[feature].copy()
    )

    before_sample = _sample_values(
        original_series
    )

    if is_numeric:
        if (
            normalized_strategy
            not in
            SUPPORTED_NUMERICAL_STRATEGIES
        ):
            raise ValueError(
                "Unsupported numerical preview "
                f'strategy: "{strategy}". '
                "Supported strategies are mean "
                "and median."
            )

        before_stats = _numeric_stats(
            original_series
        )

        if normalized_strategy == "mean":
            transformed_series, fill_value = (
                _preview_mean(
                    original_series
                )
            )

            explanation = (
                "Missing numerical values are "
                "temporarily replaced with the "
                "feature mean for this preview."
            )

        else:
            transformed_series, fill_value = (
                _preview_median(
                    original_series
                )
            )

            explanation = (
                "Missing numerical values are "
                "temporarily replaced with the "
                "feature median for this preview."
            )

        after_stats = _numeric_stats(
            transformed_series
        )

    else:
        if (
            normalized_strategy
            not in
            SUPPORTED_CATEGORICAL_STRATEGIES
        ):
            raise ValueError(
                "Unsupported categorical preview "
                f'strategy: "{strategy}". '
                "Supported strategies are "
                "most_frequent and unknown."
            )

        before_stats = (
            _categorical_stats(
                original_series
            )
        )

        if (
            normalized_strategy
            == "most_frequent"
        ):
            transformed_series, fill_value = (
                _preview_most_frequent(
                    original_series
                )
            )

            explanation = (
                "Missing categorical values are "
                "temporarily replaced with the "
                "most frequent observed category."
            )

        else:
            transformed_series, fill_value = (
                _preview_unknown(
                    original_series
                )
            )

            explanation = (
                "Missing categorical values are "
                "temporarily represented by the "
                "'Unknown' category."
            )

        after_stats = (
            _categorical_stats(
                transformed_series
            )
        )

    after_sample = _sample_values(
        transformed_series
    )

    changed_count = int(
        original_series.isna().sum()
        - transformed_series.isna().sum()
    )

    return {
        "handled": True,
        "source": "modelmind-local",
        "engine":
            "preprocessing-preview",

        "filename": path.name,

        "feature": feature,

        "feature_type": (
            "numerical"
            if is_numeric
            else "categorical"
        ),

        "strategy":
            normalized_strategy,

        "fill_value":
            fill_value,

        "changed_count":
            changed_count,

        "explanation":
            explanation,

        "before": {
            "sample":
                before_sample,
            "statistics":
                before_stats,
        },

        "after": {
            "sample":
                after_sample,
            "statistics":
                after_stats,
        },

        "safety": {
            "preview_only": True,
            "dataset_modified": False,
            "written_to_disk": False,
            "message": (
                "This is an in-memory preview. "
                "The uploaded dataset was not "
                "modified."
            ),
        },
    }