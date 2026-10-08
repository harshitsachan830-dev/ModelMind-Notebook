from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import load_dataset


SUPPORTED_NUMERICAL_STRATEGIES = {
    "mean",
    "median",
}

SUPPORTED_CATEGORICAL_STRATEGIES = {
    "most_frequent",
    "unknown",
}


def _normalize_strategy(
    strategy: str,
) -> str:
    return (
        str(strategy)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _json_safe(
    value: Any,
) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _save_dataset(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save the dataframe using the original
    dataset file format.
    """

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(
            path,
            index=False,
        )
        return

    if suffix in {".xlsx", ".xls"}:
        df.to_excel(
            path,
            index=False,
        )
        return

    if suffix == ".json":
        df.to_json(
            path,
            orient="records",
            indent=2,
        )
        return

    raise ValueError(
        f"Unsupported dataset format: {suffix}"
    )


def _backup_path(
    path: Path,
) -> Path:
    """
    One original backup is preserved.

    Example:
    students.csv
        ->
    students.modelmind-original.csv
    """

    return path.with_name(
        f"{path.stem}.modelmind-original"
        f"{path.suffix}"
    )


def _create_backup(
    path: Path,
) -> Path:
    backup = _backup_path(path)

    # Preserve the FIRST original dataset.
    # Later preprocessing operations must not
    # overwrite the original backup.
    if not backup.exists():
        backup.write_bytes(
            path.read_bytes()
        )

    return backup


def _apply_mean(
    series: pd.Series,
) -> tuple[pd.Series, Any]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    value = numeric.mean()

    if pd.isna(value):
        raise ValueError(
            "Mean imputation cannot be applied "
            "because this feature has no usable "
            "numerical values."
        )

    return (
        numeric.fillna(value),
        _json_safe(value),
    )


def _apply_median(
    series: pd.Series,
) -> tuple[pd.Series, Any]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    value = numeric.median()

    if pd.isna(value):
        raise ValueError(
            "Median imputation cannot be applied "
            "because this feature has no usable "
            "numerical values."
        )

    return (
        numeric.fillna(value),
        _json_safe(value),
    )


def _apply_most_frequent(
    series: pd.Series,
) -> tuple[pd.Series, Any]:
    modes = series.dropna().mode()

    if modes.empty:
        raise ValueError(
            "Most-frequent imputation cannot be "
            "applied because this feature has "
            "no observed values."
        )

    value = modes.iloc[0]

    return (
        series.fillna(value),
        _json_safe(value),
    )


def _apply_unknown(
    series: pd.Series,
) -> tuple[pd.Series, str]:
    value = "Unknown"

    return (
        series.fillna(value),
        value,
    )


def apply_preprocessing(
    path: Path,
    feature: str,
    strategy: str,
) -> dict[str, Any]:
    """
    Apply ONE explicitly requested preprocessing
    operation to ONE feature.

    This function modifies the runtime copy of the
    dataset only after it has been explicitly called.

    The original runtime dataset is backed up before
    the first modification so it can be restored.
    """

    if not path.exists():
        raise ValueError(
            "Dataset does not exist."
        )

    df = load_dataset(path)

    if feature not in df.columns:
        raise ValueError(
            f'Feature "{feature}" was not found '
            "in the dataset."
        )

    normalized_strategy = (
        _normalize_strategy(strategy)
    )

    original_series = (
        df[feature].copy()
    )

    missing_before = int(
        original_series.isna().sum()
    )

    if missing_before == 0:
        return {
            "handled": True,
            "source": "modelmind-local",
            "engine":
                "preprocessing-apply",
            "filename": path.name,
            "feature": feature,
            "strategy":
                normalized_strategy,
            "applied": False,
            "changed_count": 0,
            "message": (
                "No preprocessing was applied "
                "because this feature has no "
                "missing values."
            ),
        }

    is_numeric = bool(
        pd.api.types.is_numeric_dtype(
            original_series
        )
    )

    if is_numeric:
        if (
            normalized_strategy
            not in
            SUPPORTED_NUMERICAL_STRATEGIES
        ):
            raise ValueError(
                "Unsupported numerical strategy. "
                "Supported strategies are mean "
                "and median."
            )

        if normalized_strategy == "mean":
            transformed, fill_value = (
                _apply_mean(
                    original_series
                )
            )
        else:
            transformed, fill_value = (
                _apply_median(
                    original_series
                )
            )

    else:
        if (
            normalized_strategy
            not in
            SUPPORTED_CATEGORICAL_STRATEGIES
        ):
            raise ValueError(
                "Unsupported categorical strategy. "
                "Supported strategies are "
                "most_frequent and unknown."
            )

        if (
            normalized_strategy
            == "most_frequent"
        ):
            transformed, fill_value = (
                _apply_most_frequent(
                    original_series
                )
            )
        else:
            transformed, fill_value = (
                _apply_unknown(
                    original_series
                )
            )

    missing_after = int(
        transformed.isna().sum()
    )

    changed_count = (
        missing_before
        - missing_after
    )

    if changed_count <= 0:
        raise ValueError(
            "The preprocessing operation did not "
            "resolve any missing values."
        )

    # IMPORTANT:
    # Backup BEFORE changing the runtime dataset.
    backup = _create_backup(path)

    df[feature] = transformed

    _save_dataset(
        df,
        path,
    )

    return {
        "handled": True,
        "source": "modelmind-local",
        "engine":
            "preprocessing-apply",

        "filename": path.name,
        "feature": feature,
        "strategy":
            normalized_strategy,

        "fill_value":
            fill_value,

        "applied": True,

        "changed_count":
            changed_count,

        "missing_before":
            missing_before,

        "missing_after":
            missing_after,

        "backup_created":
            backup.exists(),

        "backup_filename":
            backup.name,

        "message": (
            f'Applied "{normalized_strategy}" '
            f'to "{feature}". '
            f"{changed_count} missing value(s) "
            "were filled."
        ),
    }


def undo_preprocessing(
    path: Path,
) -> dict[str, Any]:
    """
    Restore the dataset to the original backup
    created before the first preprocessing change.
    """

    if not path.exists():
        raise ValueError(
            "Dataset does not exist."
        )

    backup = _backup_path(path)

    if not backup.exists():
        raise ValueError(
            "No preprocessing backup is available "
            "for this dataset."
        )

    path.write_bytes(
        backup.read_bytes()
    )

    return {
        "handled": True,
        "source": "modelmind-local",
        "engine":
            "preprocessing-undo",

        "filename": path.name,

        "restored": True,

        "backup_filename":
            backup.name,

        "message": (
            "The runtime dataset was restored "
            "to its original state."
        ),
    }