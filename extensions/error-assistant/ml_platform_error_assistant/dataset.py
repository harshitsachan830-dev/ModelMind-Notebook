from pathlib import Path

import pandas as pd


# =========================================================
# DATASET LOADER
# =========================================================

def load_dataset(
    path: Path,
) -> pd.DataFrame:

    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    elif extension in {
        ".xlsx",
        ".xls",
    }:
        return pd.read_excel(path)

    elif extension == ".json":
        return pd.read_json(path)

    raise ValueError(
        "Dataset Intelligence currently "
        "supports CSV, Excel and JSON."
    )


# =========================================================
# GENERAL DATASET ANALYSIS
# =========================================================

def analyze_dataset(
    path: Path,
) -> dict:

    try:
        df = load_dataset(path)

    except ValueError as error:
        return {
            "supported": False,
            "message": str(error),
        }

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    missing_values = (
        df.isnull()
        .sum()
        .to_dict()
    )

    missing_values = {
        str(key): int(value)
        for key, value
        in missing_values.items()
        if value > 0
    }

    preview_df = (
        df.head(10)
        .astype(object)
        .where(
            pd.notnull(
                df.head(10)
            ),
            None,
        )
    )

    preview = (
        preview_df
        .to_dict(
            orient="records"
        )
    )

    likely_id_columns = []

    for column in df.columns:
        column_name = str(
            column
        ).lower()

        if (
            column_name == "id"
            or column_name.endswith(
                "_id"
            )
            or column_name.startswith(
                "id_"
            )
        ):
            likely_id_columns.append(
                str(column)
            )

    return {
        "supported": True,

        "filename": path.name,

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "column_names": [
            str(column)
            for column
            in df.columns
        ],

        "numeric_columns": [
            str(column)
            for column
            in numeric_columns
        ],

        "categorical_columns": [
            str(column)
            for column
            in categorical_columns
        ],

        "missing_values":
            missing_values,

        "total_missing_values":
            int(
                df.isnull()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            ),

        "likely_id_columns":
            likely_id_columns,

        "dtypes": {
            str(column):
                str(dtype)

            for column, dtype
            in df.dtypes.items()
        },

        "preview":
            preview,
    }


# =========================================================
# TARGET ANALYSIS
# =========================================================

def analyze_target(
    path: Path,
    target_column: str,
) -> dict:

    df = load_dataset(path)

    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' "
            "does not exist in this dataset."
        )

    target = df[target_column]

    total_rows = int(
        len(target)
    )

    missing_count = int(
        target.isnull().sum()
    )

    non_null_target = (
        target.dropna()
    )

    non_null_count = int(
        len(non_null_target)
    )

    unique_count = int(
        non_null_target.nunique()
    )

    dtype = str(
        target.dtype
    )

    is_numeric = bool(
        pd.api.types
        .is_numeric_dtype(
            target
        )
    )

    # =====================================================
    # DETECT ML TASK
    # =====================================================

    task = "unknown"

    task_type = "Unknown"

    if non_null_count == 0:

        task = "unknown"
        task_type = "Unknown"

    elif not is_numeric:

        task = "classification"

        if unique_count == 2:
            task_type = (
                "Binary Classification"
            )

        else:
            task_type = (
                "Multiclass Classification"
            )

    else:

        unique_ratio = (
            unique_count /
            non_null_count
        )

# Numeric targets require special
# handling.
#
# A small number of unique numeric
# values usually represents encoded
# classes such as:
#
# 0 / 1
# 1 / 2 / 3

        if unique_count == 2 or (
            unique_count <= 20
            and unique_ratio <= 0.05
        ):

            task = "classification"

            if unique_count == 2:
                task_type = (
                    "Binary Classification"
                )

            else:
                task_type = (
                    "Multiclass Classification"
                )

        else:

            task = "regression"
            task_type = "Regression"

    # =====================================================
    # COMMON RESPONSE
    # =====================================================

    result = {
        "target": target_column,

        "task": task,

        "task_type": task_type,

        "dtype": dtype,

        "is_numeric":
            is_numeric,

        "total_rows":
            total_rows,

        "non_null_count":
            non_null_count,

        "missing_count":
            missing_count,

        "unique_count":
            unique_count,
    }

    # =====================================================
    # CLASSIFICATION ANALYSIS
    # =====================================================

    if task == "classification":

        counts = (
            non_null_target
            .value_counts(
                dropna=True
            )
        )

        distribution = []

        for value, count in counts.items():

            percentage = (
                (
                    int(count) /
                    non_null_count
                )
                * 100
                if non_null_count > 0
                else 0
            )

            distribution.append({
                "value": str(value),

                "count":
                    int(count),

                "percentage":
                    round(
                        percentage,
                        2,
                    ),
            })

        largest_class_percentage = (
            distribution[0][
                "percentage"
            ]
            if distribution
            else 0
        )

        smallest_class_percentage = (
            distribution[-1][
                "percentage"
            ]
            if distribution
            else 0
        )

        imbalance_ratio = None

        if (
            distribution
            and distribution[-1][
                "count"
            ] > 0
        ):

            imbalance_ratio = round(
                distribution[0][
                    "count"
                ]
                /
                distribution[-1][
                    "count"
                ],
                2,
            )

        # ---------------------------------
        # SIMPLE IMBALANCE CLASSIFICATION
        # ---------------------------------

        imbalance_level = "balanced"

        if largest_class_percentage >= 90:

            imbalance_level = "severe"

        elif largest_class_percentage >= 75:

            imbalance_level = "high"

        elif largest_class_percentage >= 60:

            imbalance_level = "moderate"

        result.update({
            "class_distribution":
                distribution,

            "class_counts": {
                item["value"]: item["count"]
                for item in distribution
            },

            "class_percentages": {
                item["value"]: item["percentage"]
                for item in distribution
            },

            "largest_class_percentage":
                largest_class_percentage,

            "smallest_class_percentage":
                smallest_class_percentage,

            "imbalance_ratio":
                imbalance_ratio,

            "imbalance_level":
                imbalance_level,

            "is_imbalanced":
                imbalance_level != "balanced",
        })

    # =====================================================
    # REGRESSION ANALYSIS
    # =====================================================

    elif task == "regression":

        numeric_target = (
            pd.to_numeric(
                non_null_target,
                errors="coerce",
            )
            .dropna()
        )

        if len(numeric_target) > 0:
            minimum = float(numeric_target.min())
            maximum = float(numeric_target.max())
            mean = float(numeric_target.mean())
            median = float(numeric_target.median())
            standard_deviation = (
                float(numeric_target.std())
                if len(numeric_target) > 1
                else 0.0
            )

            q1 = float(numeric_target.quantile(0.25))
            q3 = float(numeric_target.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (
                (numeric_target < lower_bound)
                | (numeric_target > upper_bound)
            )
            outlier_count = int(outlier_mask.sum())
            outlier_percentage = (
                outlier_count / len(numeric_target)
            ) * 100

            skewness = float(numeric_target.skew())
            if skewness > 1:
                distribution_shape = "strongly_right_skewed"
            elif skewness > 0.5:
                distribution_shape = "right_skewed"
            elif skewness < -1:
                distribution_shape = "strongly_left_skewed"
            elif skewness < -0.5:
                distribution_shape = "left_skewed"
            else:
                distribution_shape = "approximately_symmetric"

            negative_count = int((numeric_target < 0).sum())
            zero_count = int((numeric_target == 0).sum())
            missing_percentage = (
                (missing_count / total_rows) * 100
                if total_rows > 0
                else 0
            )

            result.update({
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "median": median,
                "standard_deviation": standard_deviation,
                "std": standard_deviation,
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "lower_outlier_bound": float(lower_bound),
                "upper_outlier_bound": float(upper_bound),
                "outlier_count": outlier_count,
                "outlier_percentage": round(outlier_percentage, 2),
                "skewness": round(skewness, 4),
                "distribution_shape": distribution_shape,
                "negative_count": negative_count,
                "zero_count": zero_count,
                "missing_percentage": round(missing_percentage, 2),
            })

    return result
# =========================================================
# FEATURE ANALYSIS
# =========================================================

# =========================================================
# FEATURE ANALYSIS
# =========================================================


def analyze_feature(
    path: Path,
    feature_column: str,
) -> dict:

    df = load_dataset(path)

    if feature_column not in df.columns:
        raise ValueError(
            f"Column '{feature_column}' "
            "does not exist in this dataset."
        )

    feature = df[feature_column]

    total_rows = int(len(feature))

    missing_count = int(
        feature.isnull().sum()
    )

    missing_percentage = (
        (missing_count / total_rows) * 100
        if total_rows > 0
        else 0
    )

    non_null = feature.dropna()

    non_null_count = int(
        len(non_null)
    )

    unique_count = int(
        non_null.nunique()
    )

    dtype = str(feature.dtype)

    is_numeric = bool(
        pd.api.types.is_numeric_dtype(
            feature
        )
    )

    result = {
        "feature": feature_column,
        "dtype": dtype,
        "is_numeric": is_numeric,
        "total_rows": total_rows,
        "non_null_count": non_null_count,
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_percentage,
            2,
        ),
        "unique_count": unique_count,
    }

    # =====================================================
    # NUMERICAL FEATURE
    # =====================================================

    if is_numeric:

        numeric_feature = (
            pd.to_numeric(
                non_null,
                errors="coerce",
            )
            .dropna()
        )

        if len(numeric_feature) == 0:

            result.update({
                "feature_type": "numerical",
                "histogram_counts": [],
                "histogram_edges": [],
                "kde_x": [],
                "kde_y": [],
            })

            return result

        # -------------------------------------------------
        # BASIC STATISTICS
        # -------------------------------------------------

        mean = float(
            numeric_feature.mean()
        )

        median = float(
            numeric_feature.median()
        )

        minimum = float(
            numeric_feature.min()
        )

        maximum = float(
            numeric_feature.max()
        )

        standard_deviation = (
            float(
                numeric_feature.std()
            )
            if len(numeric_feature) > 1
            else 0.0
        )

        # -------------------------------------------------
        # QUARTILES + IQR
        # -------------------------------------------------

        q1 = float(
            numeric_feature.quantile(
                0.25
            )
        )

        q3 = float(
            numeric_feature.quantile(
                0.75
            )
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        # -------------------------------------------------
        # OUTLIERS
        # -------------------------------------------------

        outlier_mask = (
            (numeric_feature < lower_bound)
            |
            (numeric_feature > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        outlier_percentage = (
            (
                outlier_count
                / len(numeric_feature)
            )
            * 100
            if len(numeric_feature) > 0
            else 0
        )

        # -------------------------------------------------
        # SKEWNESS + DISTRIBUTION SHAPE
        # -------------------------------------------------

        if len(numeric_feature) > 2:

            skewness = float(
                numeric_feature.skew()
            )

            # NaN protection for constant /
            # unusual numerical columns.
            if pd.isna(skewness):
                skewness = 0.0

        else:
            skewness = 0.0

        if skewness > 1:

            distribution_shape = (
                "strongly_right_skewed"
            )

        elif skewness > 0.5:

            distribution_shape = (
                "right_skewed"
            )

        elif skewness < -1:

            distribution_shape = (
                "strongly_left_skewed"
            )

        elif skewness < -0.5:

            distribution_shape = (
                "left_skewed"
            )

        else:

            distribution_shape = (
                "approximately_symmetric"
            )

        # -------------------------------------------------
        # SPECIAL VALUES
        # -------------------------------------------------

        negative_count = int(
            (numeric_feature < 0).sum()
        )

        zero_count = int(
            (numeric_feature == 0).sum()
        )

        # =================================================
        # HISTOGRAM + KDE DATA
        # =================================================

        histogram_counts = []
        histogram_edges = []

        kde_x = []
        kde_y = []

        if len(numeric_feature) > 0:

            import numpy as np

            # ---------------------------------------------
            # HISTOGRAM
            # ---------------------------------------------

            try:

                counts, edges = (
                    np.histogram(
                        numeric_feature,
                        bins="auto",
                    )
                )

                histogram_counts = [
                    int(value)
                    for value in counts
                ]

                histogram_edges = [
                    float(value)
                    for value in edges
                ]

            except Exception:

                histogram_counts = []
                histogram_edges = []

            # ---------------------------------------------
            # KDE
            # Kernel Density Estimation
            # ---------------------------------------------

            if (
                len(numeric_feature) >= 2
                and
                numeric_feature.nunique() > 1
            ):

                try:

                    from scipy.stats import (
                        gaussian_kde
                    )

                    values = (
                        numeric_feature
                        .astype(float)
                        .to_numpy()
                    )

                    kde = gaussian_kde(
                        values
                    )

                    x_values = np.linspace(
                        float(values.min()),
                        float(values.max()),
                        120,
                    )

                    y_values = kde(
                        x_values
                    )

                    kde_x = [
                        float(value)
                        for value
                        in x_values
                    ]

                    kde_y = [
                        float(value)
                        for value
                        in y_values
                    ]

                except Exception:

                    # KDE is only a visualization
                    # helper. The rest of Feature
                    # Analyzer should continue even
                    # when KDE cannot be calculated.

                    kde_x = []
                    kde_y = []

        # =================================================
        # IMPUTATION ADVISOR
        # =================================================

        if missing_count == 0:

            imputation_strategy = (
                "none"
            )

            imputation_reason = (
                "This feature has no missing "
                "values, so imputation is not "
                "currently required."
            )

        elif (
            abs(skewness) <= 0.5
            and
            outlier_percentage < 5
        ):

            imputation_strategy = (
                "mean"
            )

            imputation_reason = (
                "The numerical feature is "
                "approximately symmetric and "
                "does not contain many IQR "
                "outliers. Mean imputation is "
                "a reasonable starting option."
            )

        else:

            imputation_strategy = (
                "median"
            )

            imputation_reason = (
                "The numerical feature is "
                "skewed or contains notable "
                "outliers. The median is more "
                "resistant to extreme values "
                "than the mean."
            )

        # =================================================
        # SCALING ADVISOR
        # =================================================

        if outlier_percentage >= 5:

            scaling_strategy = (
                "robust"
            )

            scaling_reason = (
                "This feature contains a "
                "notable proportion of IQR "
                "outliers. RobustScaler uses "
                "the median and IQR and is "
                "less sensitive to extreme "
                "values."
            )

        elif abs(skewness) <= 0.5:

            scaling_strategy = (
                "standard"
            )

            scaling_reason = (
                "The feature is approximately "
                "symmetric without strong "
                "outlier evidence. "
                "StandardScaler is a useful "
                "option for models that are "
                "sensitive to feature scale."
            )

        else:

            scaling_strategy = (
                "inspect_after_transform"
            )

            scaling_reason = (
                "The feature is skewed. "
                "ModelMind recommends checking "
                "whether a transformation is "
                "appropriate before choosing "
                "the final scaling strategy."
            )

        # =================================================
        # TRANSFORMATION ADVISOR
        # =================================================

        if abs(skewness) <= 0.5:

            transformation_strategy = (
                "none"
            )

            transformation_reason = (
                "No strong skewness was "
                "detected."
            )

        elif minimum >= 0:

            transformation_strategy = (
                "log1p_candidate"
            )

            transformation_reason = (
                "The feature is skewed and "
                "contains no negative values. "
                "A log1p transformation can "
                "be explored, but should be "
                "validated using the resulting "
                "distribution and model "
                "performance."
            )

        else:

            transformation_strategy = (
                "power_transform_candidate"
            )

            transformation_reason = (
                "The feature is skewed and "
                "contains negative values. "
                "A power transformation such "
                "as Yeo-Johnson can be "
                "explored because a simple "
                "log1p transformation may not "
                "be appropriate."
            )

        # =================================================
        # NUMERICAL RESPONSE
        # =================================================

        result.update({

            "feature_type":
                "numerical",

            "minimum":
                minimum,

            "maximum":
                maximum,

            "mean":
                mean,

            "median":
                median,

            "standard_deviation":
                standard_deviation,

            "q1":
                q1,

            "q3":
                q3,

            "iqr":
                float(iqr),

            "lower_outlier_bound":
                float(lower_bound),

            "upper_outlier_bound":
                float(upper_bound),

            "outlier_count":
                outlier_count,

            "outlier_percentage":
                round(
                    outlier_percentage,
                    2,
                ),

            "skewness":
                round(
                    skewness,
                    4,
                ),

            "distribution_shape":
                distribution_shape,

            "negative_count":
                negative_count,

            "zero_count":
                zero_count,

            # Histogram data
            "histogram_counts":
                histogram_counts,

            "histogram_edges":
                histogram_edges,

            # KDE data
            "kde_x":
                kde_x,

            "kde_y":
                kde_y,

            # Imputation
            "imputation_strategy":
                imputation_strategy,

            "imputation_reason":
                imputation_reason,

            # Scaling
            "scaling_strategy":
                scaling_strategy,

            "scaling_reason":
                scaling_reason,

            # Transformation
            "transformation_strategy":
                transformation_strategy,

            "transformation_reason":
                transformation_reason,
        })

    # =====================================================
    # CATEGORICAL FEATURE
    # =====================================================

    else:

        counts = (
            non_null
            .astype(str)
            .value_counts()
        )

        category_distribution = []

        for value, count in counts.items():

            percentage = (
                (
                    int(count)
                    / non_null_count
                )
                * 100
                if non_null_count > 0
                else 0
            )

            category_distribution.append({
                "value":
                    str(value),

                "count":
                    int(count),

                "percentage":
                    round(
                        percentage,
                        2,
                    ),
            })

        # =================================================
        # CATEGORICAL IMPUTATION ADVISOR
        # =================================================

        if missing_count > 0:

            imputation_strategy = (
                "most_frequent_or_unknown"
            )

            imputation_reason = (
                "This is a categorical "
                "feature with missing values. "
                "ModelMind recommends "
                "comparing most-frequent "
                "imputation with an explicit "
                "'Unknown' category depending "
                "on what missing means."
            )

        else:

            imputation_strategy = (
                "none"
            )

            imputation_reason = (
                "This categorical feature "
                "has no missing values."
            )

        # =================================================
        # CATEGORICAL RESPONSE
        # =================================================

        result.update({

            "feature_type":
                "categorical",

            "category_distribution":
                category_distribution,

            "imputation_strategy":
                imputation_strategy,

            "imputation_reason":
                imputation_reason,
        })

    return result