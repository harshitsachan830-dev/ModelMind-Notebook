"""
ModelMind — Intelligent Model Recommender
==========================================
Analyses dataset properties and recommends the best-fit ML models
ranked by suitability, with complete ready-to-run Python code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _round(v: float, n: int = 4) -> float:
    return round(float(v), n)


def _infer_task(
    df: pd.DataFrame,
    target_col: str,
) -> str:
    """Infer whether the task is classification or regression."""
    series = df[target_col].dropna()

    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    n_unique = series.nunique()
    n_total = len(series)

    # If very few unique values relative to total → classification
    if n_unique == 2 or (
        n_unique <= 20 and n_unique / n_total < 0.05
    ):
        return "classification"

    return "regression"


def _dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    total_missing = int(df.isnull().sum().sum())
    missing_pct = _round(total_missing / max(rows * cols, 1) * 100)

    has_missing = total_missing > 0
    high_cardinality = any(
        df[c].nunique() > 50 for c in categorical_cols
    )
    n_features = cols - 1  # excluding target

    return {
        "rows": rows,
        "cols": cols,
        "n_features": n_features,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "total_missing": total_missing,
        "missing_pct": missing_pct,
        "has_missing": has_missing,
        "high_cardinality": high_cardinality,
        "is_small": rows < 1000,
        "is_medium": 1000 <= rows < 50_000,
        "is_large": rows >= 50_000,
        "is_wide": n_features >= 50,
        "is_narrow": n_features < 10,
    }


def _add_categorical_target_encoding(code: str) -> str:
    code = code.replace("\\n", "\n")
    code = re.sub(
        r"(?m)^if y\.dtype == object:$",
        "target_is_categorical = not pd.api.types.is_numeric_dtype(y)\n"
        "if target_is_categorical:",
        code,
    )
    code = code.replace("le = LabelEncoder()", "target_encoder = LabelEncoder()")
    code = code.replace(
        "y = le.fit_transform(y)",
        "y = target_encoder.fit_transform(y)\n"
        '    print("Target label encoding:", '
        "dict(enumerate(target_encoder.classes_)))",
    )

    prediction_match = re.search(
        r"(?m)^(y_pred(?:_ridge)?) = "
        r"(?:pipeline|grid_search|ridge_pipeline)\.predict\(X_test\)$",
        code,
    )
    if not prediction_match or "if target_is_categorical:" not in code:
        raise ValueError(
            "Could not safely add categorical target encoding to model code."
        )

    prediction_variable = prediction_match.group(1)
    return re.sub(
        r"(?m)^(y_pred(?:_ridge)?) = "
        r"((?:pipeline|grid_search|ridge_pipeline)\.predict\(X_test\))$",
        rf"\1 = \2\n"
        "if target_is_categorical:\n"
        "    y_test = target_encoder.inverse_transform(y_test.astype(int))\n"
        rf"    \1 = target_encoder.inverse_transform({prediction_variable}.astype(int))",
        code,
        count=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL CODE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _code_linear_regression(target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    return f'''\
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing ──
numeric_features   = {[c for c in num_cols if c != target]!r}
categorical_features = {cat_cols!r}

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
], remainder="drop")

# ── Pipeline ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Evaluation ──
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE : {{rmse:.4f}}")
print(f"R²   : {{r2:.4f}}")
'''


def _code_logistic_regression(target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    return f'''\
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

# Encode target if it is string
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing ──
numeric_features     = {[c for c in num_cols if c != target]!r}
categorical_features = {cat_cols!r}

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
], remainder="drop")

# ── Pipeline ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Evaluation ──
print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))
'''


def _code_random_forest(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    estimator = (
        "RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)"
        if task == "classification"
        else "RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)"
    )
    cls = "RandomForestClassifier" if task == "classification" else "RandomForestRegressor"
    metric_code = (
        "from sklearn.metrics import accuracy_score, classification_report\n"
        "print(f\"Accuracy : {accuracy_score(y_test, y_pred):.4f}\")\n"
        "print(classification_report(y_test, y_pred))"
        if task == "classification"
        else "from sklearn.metrics import mean_squared_error, r2_score\nimport numpy as np\n"
             "print(f\"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\")\n"
             "print(f\"R²   : {r2_score(y_test, y_pred):.4f}\")"
    )
    return f'''\
import pandas as pd
from sklearn.ensemble import {cls}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{metric_code.split(chr(10))[0]}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"# Encode target if string\\nif y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing (Random Forest handles missing values poorly — impute first) ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── Pipeline ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {estimator}),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Feature Importance ──
import numpy as np
feature_names = (
    {[c for c in num_cols if c != target]!r} +
    list(pipeline.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .named_steps["encoder"]
         .get_feature_names_out({cat_cols!r}))
)
importances = pipeline.named_steps["model"].feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("Top 10 features:")
for i in top_idx:
    print(f"  {{feature_names[i]}}: {{importances[i]:.4f}}")

# ── Evaluation ──
{metric_code}
'''


def _code_xgboost(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    cls = "XGBClassifier" if task == "classification" else "XGBRegressor"
    metric_code = (
        "from sklearn.metrics import accuracy_score, classification_report\n"
        "print(f\"Accuracy : {accuracy_score(y_test, y_pred):.4f}\")\n"
        "print(classification_report(y_test, y_pred))"
        if task == "classification"
        else "from sklearn.metrics import mean_squared_error, r2_score\nimport numpy as np\n"
             "print(f\"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\")\n"
             "print(f\"R²   : {r2_score(y_test, y_pred):.4f}\")"
    )
    return f'''\
import pandas as pd
from xgboost import {cls}
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{metric_code.split(chr(10))[0]}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"# Encode target if string\\nif y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing (XGBoost handles missing natively, but encoding needed) ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── XGBoost Model ──
xgb_model = {cls}(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="{"logloss" if task == "classification" else "rmse"}",
    random_state=42,
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Cross Validation ──
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
    scoring="{"accuracy" if task == "classification" else "r2"}")
print(f"CV Score: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

# ── Evaluation ──
{metric_code}
'''


def _code_svm(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    cls = "SVC" if task == "classification" else "SVR"
    return f'''\
import pandas as pd
from sklearn.svm import {cls}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{"from sklearn.metrics import accuracy_score, classification_report" if task == "classification" else "from sklearn.metrics import mean_squared_error, r2_score\\nimport numpy as np"}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"# Encode target\\nif y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing (SVM requires scaling!) ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── SVM Pipeline ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {cls}(C=1.0, kernel="rbf", {"probability=True" if task == "classification" else "epsilon=0.1"})),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Evaluation ──
{"print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')\\nprint(classification_report(y_test, y_pred))" if task == "classification" else "print(f'RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\\nprint(f'R²   : {r2_score(y_test, y_pred):.4f}')"}

# ⚠️  Note: SVM is slow on large datasets (>10k rows). Use LinearSVC for speed.
'''


def _code_knn(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    cls = "KNeighborsClassifier" if task == "classification" else "KNeighborsRegressor"
    return f'''\
import pandas as pd
from sklearn.neighbors import {cls}
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{"from sklearn.metrics import accuracy_score, classification_report" if task == "classification" else "from sklearn.metrics import mean_squared_error, r2_score\\nimport numpy as np"}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"if y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing (KNN requires scaling and no missing values) ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── KNN Pipeline with auto k-tuning ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {cls}(n_neighbors=5)),
])

# Optional: Find best k
param_grid = {{"model__n_neighbors": [3, 5, 7, 9, 11]}}
grid_search = GridSearchCV(pipeline, param_grid, cv=5,
    scoring="{"accuracy" if task == "classification" else "r2"}", n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best k: {{grid_search.best_params_}}")

y_pred = grid_search.predict(X_test)

# ── Evaluation ──
{"print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')\\nprint(classification_report(y_test, y_pred))" if task == "classification" else "print(f'RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\\nprint(f'R²   : {r2_score(y_test, y_pred):.4f}')"}
'''


def _code_gradient_boosting(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    cls = "GradientBoostingClassifier" if task == "classification" else "GradientBoostingRegressor"
    return f'''\
import pandas as pd
from sklearn.ensemble import {cls}
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{"from sklearn.metrics import accuracy_score, classification_report" if task == "classification" else "from sklearn.metrics import mean_squared_error, r2_score\\nimport numpy as np"}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"if y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── Gradient Boosting Pipeline ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {cls}(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Cross Validation ──
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
    scoring="{"accuracy" if task == "classification" else "r2"}")
print(f"CV Score: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

# ── Evaluation ──
{"print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')\\nprint(classification_report(y_test, y_pred))" if task == "classification" else "print(f'RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\\nprint(f'R²   : {r2_score(y_test, y_pred):.4f}')"}
'''


def _code_decision_tree(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    cls = "DecisionTreeClassifier" if task == "classification" else "DecisionTreeRegressor"
    return f'''\
import pandas as pd
from sklearn.tree import {cls}, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
{"from sklearn.metrics import accuracy_score, classification_report" if task == "classification" else "from sklearn.metrics import mean_squared_error, r2_score\\nimport numpy as np"}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"if y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── Decision Tree (interpretable, limit depth to avoid overfitting) ──
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {cls}(max_depth=5, random_state=42)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Print Tree Structure ──
tree_rules = export_text(pipeline.named_steps["model"], max_depth=3)
print(tree_rules[:2000])  # first 2000 chars

# ── Evaluation ──
{"print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')\\nprint(classification_report(y_test, y_pred))" if task == "classification" else "print(f'RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\\nprint(f'R²   : {r2_score(y_test, y_pred):.4f}')"}

# ── Optional: Visualise tree ──
plt.figure(figsize=(20, 8))
plot_tree(pipeline.named_steps["model"], max_depth=3, filled=True, fontsize=8)
plt.title("Decision Tree Visualisation")
plt.tight_layout()
plt.show()
'''


def _code_naive_bayes(target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    return f'''\
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{[c for c in num_cols if c != target]!r}]   # Gaussian NB works best with numeric features
y = df["{target}"]

if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Pipeline (Gaussian NB works with numeric data) ──
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler()),
    ("model",   GaussianNB()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── Evaluation ──
print(f"Accuracy : {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))

# ── Posterior Probabilities ──
proba = pipeline.predict_proba(X_test[:5])
print("Class probabilities (first 5 samples):", proba)
'''


def _code_ridge_lasso(task: str, target: str, num_cols: list[str], cat_cols: list[str]) -> str:
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    cls = "RidgeClassifier" if task == "classification" else "Ridge"
    return f'''\
import pandas as pd
from sklearn.linear_model import {cls}, Lasso
from sklearn.model_selection import train_test_split, RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{"from sklearn.metrics import accuracy_score, classification_report" if task == "classification" else "from sklearn.metrics import mean_squared_error, r2_score\\nimport numpy as np"}

# ── Load & Split ──
df = pd.read_csv("your_dataset.csv")   # replace with actual path
X = df[{feature_cols!r}]
y = df["{target}"]

{"if y.dtype == object:\\n    le = LabelEncoder()\\n    y = le.fit_transform(y)\\n" if task == "classification" else ""}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing ──
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,     {[c for c in num_cols if c != target]!r}),
    ("cat", categorical_pipeline, {cat_cols!r}),
], remainder="drop")

# ── Ridge Model (L2 regularisation) ──
ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {cls}(alpha=1.0)),
])
ridge_pipeline.fit(X_train, y_train)
y_pred_ridge = ridge_pipeline.predict(X_test)

{"print(f'Ridge Accuracy : {accuracy_score(y_test, y_pred_ridge):.4f}')" if task == "classification" else "# Lasso (L1 regularisation — performs feature selection)\\nlasso_pipeline = Pipeline(steps=[\\n    ('preprocessor', preprocessor),\\n    ('model', Lasso(alpha=0.01, max_iter=5000)),\\n])\\nlasso_pipeline.fit(X_train, y_train)\\ny_pred_lasso = lasso_pipeline.predict(X_test)\\nprint(f'Ridge R² : {r2_score(y_test, y_pred_ridge):.4f}')\\nprint(f'Lasso R² : {r2_score(y_test, lasso_pipeline.predict(X_test)):.4f}')"}
'''


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING & RANKING
# ─────────────────────────────────────────────────────────────────────────────

def _score_models(
    task: str,
    profile: dict[str, Any],
    target_col: str,
) -> list[dict[str, Any]]:
    """
    Return a list of model dicts, ordered best→worst by suitability_score.
    Score is 0–100 (heuristic, not cross-validated).
    """
    rows = profile["rows"]
    n_feat = profile["n_features"]
    has_missing = profile["has_missing"]
    is_wide = profile["is_wide"]
    is_large = profile["is_large"]
    is_small = profile["is_small"]
    num_cols = profile["numeric_cols"]
    cat_cols = [c for c in profile["categorical_cols"] if c != target_col]
    num_feat_cols = [c for c in num_cols if c != target_col]

    models: list[dict[str, Any]] = []

    if task == "regression":
        # ── Linear Regression ──────────────────────────────────────────────
        score = 70
        reasons: list[str] = []
        if not is_wide and not has_missing:
            score += 10
            reasons.append("Numeric features with no missing values: ideal for linear models.")
        if is_small:
            score += 5
            reasons.append("Small dataset: linear models fit quickly.")
        if is_wide:
            score -= 15
            reasons.append("Many features: multicollinearity risk — consider Ridge/Lasso.")
        models.append({
            "id": "linear_regression",
            "name": "Linear Regression",
            "score": min(score, 100),
            "tags": ["Interpretable", "Fast", "Baseline"],
            "why": "Excellent starting baseline for regression. "
                   "Works best when the relationship between features and target is roughly linear.",
            "strengths": ["Very fast to train", "Highly interpretable coefficients",
                          "Works well with normally distributed errors"],
            "weaknesses": ["Assumes linear relationship", "Sensitive to outliers",
                           "Poor with complex non-linear patterns"],
            "best_for": "Continuous targets, low-complexity datasets, baseline model",
            "complexity": "Low",
            "training_speed": "⚡ Very Fast",
            "interpretability": "★★★★★",
            "code": _code_linear_regression(target_col, num_feat_cols, cat_cols),
            "reasons": reasons or ["Solid baseline for regression tasks."],
        })

        # ── Ridge / Lasso ──────────────────────────────────────────────────
        score2 = 72
        r2 = []
        if is_wide:
            score2 += 15
            r2.append("Wide dataset: regularisation (L1/L2) prevents overfitting.")
        if has_missing:
            score2 -= 5
        models.append({
            "id": "ridge_lasso",
            "name": "Ridge / Lasso Regression",
            "score": min(score2, 100),
            "tags": ["Regularised", "Feature Selection", "Interpretable"],
            "why": "Adds L1 (Lasso) or L2 (Ridge) penalty to linear regression. "
                   "Ridge handles correlated features; Lasso zeroes out irrelevant ones.",
            "strengths": ["Handles multicollinearity", "Lasso does automatic feature selection",
                          "Stable with wide datasets"],
            "weaknesses": ["Still assumes linear relationship", "Requires careful alpha tuning"],
            "best_for": "Wide datasets with many features, correlated predictors",
            "complexity": "Low–Medium",
            "training_speed": "⚡ Very Fast",
            "interpretability": "★★★★☆",
            "code": _code_ridge_lasso(task, target_col, num_feat_cols, cat_cols),
            "reasons": r2 or ["Good regularised alternative to plain Linear Regression."],
        })

        # ── Random Forest Regressor ────────────────────────────────────────
        score3 = 80
        r3 = []
        if has_missing:
            score3 -= 5
            r3.append("Has missing values — imputation required before Random Forest.")
        if is_large:
            score3 -= 8
            r3.append("Large dataset: training may be slow — consider XGBoost.")
        if n_feat >= 5:
            score3 += 5
            r3.append("Multiple features: Random Forest exploits feature interactions well.")
        models.append({
            "id": "random_forest_reg",
            "name": "Random Forest Regressor",
            "score": min(score3, 100),
            "tags": ["Ensemble", "Non-linear", "Robust"],
            "why": "Aggregates hundreds of decision trees. "
                   "Naturally handles non-linear relationships and feature interactions.",
            "strengths": ["Handles non-linearity", "Robust to outliers",
                          "Built-in feature importance", "Less prone to overfitting"],
            "weaknesses": ["Slower than linear models on large data",
                           "Less interpretable than single trees", "Memory-intensive"],
            "best_for": "Medium-sized datasets with complex relationships",
            "complexity": "Medium",
            "training_speed": "🚀 Fast",
            "interpretability": "★★★☆☆",
            "code": _code_random_forest(task, target_col, num_feat_cols, cat_cols),
            "reasons": r3 or ["Powerful ensemble that handles non-linear patterns."],
        })

        # ── XGBoost Regressor ──────────────────────────────────────────────
        score4 = 87
        r4 = []
        if is_large:
            score4 += 8
            r4.append("Large dataset: XGBoost is optimised for speed and performance.")
        if has_missing:
            score4 += 3
            r4.append("Has missing values: XGBoost handles them natively.")
        models.append({
            "id": "xgboost_reg",
            "name": "XGBoost Regressor",
            "score": min(score4, 100),
            "tags": ["Gradient Boosting", "State-of-the-Art", "Fast"],
            "why": "State-of-the-art gradient boosting that wins most Kaggle tabular competitions. "
                   "Handles missing values natively and is highly configurable.",
            "strengths": ["Top performance on tabular data", "Handles missing values natively",
                          "Fast with GPU support", "Regularised to prevent overfitting"],
            "weaknesses": ["Many hyperparameters to tune", "Black-box model",
                           "Requires XGBoost library"],
            "best_for": "Medium-to-large datasets where maximum accuracy is needed",
            "complexity": "Medium–High",
            "training_speed": "⚡ Fast (with GPU)",
            "interpretability": "★★☆☆☆",
            "code": _code_xgboost(task, target_col, num_feat_cols, cat_cols),
            "reasons": r4 or ["Best-in-class for tabular regression tasks."],
        })

        # ── Gradient Boosting ──────────────────────────────────────────────
        score5 = 83
        models.append({
            "id": "gradient_boosting_reg",
            "name": "Gradient Boosting Regressor",
            "score": min(score5, 100),
            "tags": ["Ensemble", "Boosting", "Robust"],
            "why": "sklearn's built-in gradient boosting — no extra library needed. "
                   "Excellent accuracy for tabular data.",
            "strengths": ["No extra library needed", "Often outperforms Random Forest",
                          "Handles non-linearity"],
            "weaknesses": ["Slower than XGBoost", "Prone to overfitting without tuning"],
            "best_for": "When XGBoost is not available or a pure sklearn pipeline is needed",
            "complexity": "Medium",
            "training_speed": "🚀 Moderate",
            "interpretability": "★★☆☆☆",
            "code": _code_gradient_boosting(task, target_col, num_feat_cols, cat_cols),
            "reasons": ["Solid gradient boosting without external dependencies."],
        })

        # ── SVR ────────────────────────────────────────────────────────────
        score6 = 65
        if is_large:
            score6 -= 20
        if is_small:
            score6 += 10
        models.append({
            "id": "svr",
            "name": "Support Vector Regressor (SVR)",
            "score": max(score6, 30),
            "tags": ["SVM", "Kernel-based", "Small Data"],
            "why": "Effective in high-dimensional spaces. "
                   "Uses kernel trick to capture non-linear patterns.",
            "strengths": ["Works well with small datasets", "Handles high-dim features",
                          "Robust to outliers (with appropriate kernel)"],
            "weaknesses": ["Very slow on large datasets (>10k rows)",
                           "Requires feature scaling", "Hard to interpret"],
            "best_for": "Small datasets (<5k rows) with complex non-linear patterns",
            "complexity": "High",
            "training_speed": "🐢 Slow on large data",
            "interpretability": "★☆☆☆☆",
            "code": _code_svm(task, target_col, num_feat_cols, cat_cols),
            "reasons": ["Good for small, high-dimensional datasets."],
        })

    else:  # classification
        # ── Logistic Regression ────────────────────────────────────────────
        score = 70
        reasons: list[str] = []
        if not has_missing and len(cat_cols) == 0:
            score += 10
        if is_large:
            score += 5
        models.append({
            "id": "logistic_regression",
            "name": "Logistic Regression",
            "score": min(score, 100),
            "tags": ["Interpretable", "Fast", "Baseline", "Probabilistic"],
            "why": "The classic classification baseline. "
                   "Outputs calibrated probabilities and is highly interpretable.",
            "strengths": ["Fast training", "Interpretable coefficients",
                          "Gives probability outputs", "Works well with linear boundaries"],
            "weaknesses": ["Assumes linear decision boundary",
                           "Poor with complex non-linear patterns",
                           "Sensitive to outliers"],
            "best_for": "Binary or multi-class problems with linearly separable data",
            "complexity": "Low",
            "training_speed": "⚡ Very Fast",
            "interpretability": "★★★★★",
            "code": _code_logistic_regression(target_col, num_feat_cols, cat_cols),
            "reasons": reasons or ["Essential baseline for classification."],
        })

        # ── Decision Tree ──────────────────────────────────────────────────
        score2 = 68
        r2 = []
        if is_small:
            score2 += 8
            r2.append("Small dataset: Decision Tree avoids overfitting risk.")
        models.append({
            "id": "decision_tree",
            "name": "Decision Tree",
            "score": min(score2, 100),
            "tags": ["Interpretable", "Visualisable", "No Scaling Needed"],
            "why": "Highly interpretable model that creates human-readable if-then rules. "
                   "Great for understanding what features drive the prediction.",
            "strengths": ["Fully interpretable and visualisable",
                          "No feature scaling required", "Handles both numeric and categorical"],
            "weaknesses": ["Prone to overfitting without depth limit",
                           "Unstable (small data change → very different tree)",
                           "Lower accuracy than ensembles"],
            "best_for": "When explainability is critical (healthcare, finance, legal)",
            "complexity": "Low",
            "training_speed": "⚡ Very Fast",
            "interpretability": "★★★★★",
            "code": _code_decision_tree(task, target_col, num_feat_cols, cat_cols),
            "reasons": r2 or ["Best explainability of any model."],
        })

        # ── Random Forest Classifier ───────────────────────────────────────
        score3 = 82
        r3 = []
        if has_missing:
            r3.append("Has missing values — imputation built into pipeline.")
        if n_feat >= 5:
            score3 += 5
        if is_large:
            score3 -= 5
        models.append({
            "id": "random_forest_clf",
            "name": "Random Forest Classifier",
            "score": min(score3, 100),
            "tags": ["Ensemble", "Robust", "Feature Importance"],
            "why": "Aggregates many decision trees using bagging. "
                   "One of the most reliable classifiers for tabular data.",
            "strengths": ["Handles non-linear patterns", "Built-in feature importance",
                          "Robust to outliers", "Less tuning needed than boosting models"],
            "weaknesses": ["Slower inference than single models",
                           "Memory intensive", "Less accurate than boosting on large data"],
            "best_for": "Most tabular classification problems as a strong general model",
            "complexity": "Medium",
            "training_speed": "🚀 Fast",
            "interpretability": "★★★☆☆",
            "code": _code_random_forest(task, target_col, num_feat_cols, cat_cols),
            "reasons": r3 or ["Robust all-round classifier."],
        })

        # ── XGBoost Classifier ─────────────────────────────────────────────
        score4 = 89
        r4 = []
        if is_large:
            score4 += 8
            r4.append("Large dataset: XGBoost excels with big data.")
        if has_missing:
            score4 += 3
            r4.append("Has missing values: XGBoost handles them natively.")
        models.append({
            "id": "xgboost_clf",
            "name": "XGBoost Classifier",
            "score": min(score4, 100),
            "tags": ["State-of-the-Art", "Gradient Boosting", "Kaggle Winner"],
            "why": "The go-to model for tabular ML competitions. "
                   "Combines gradient boosting with regularisation for top accuracy.",
            "strengths": ["Best accuracy on tabular data", "Handles missing values natively",
                          "GPU support for speed", "Built-in cross-validation"],
            "weaknesses": ["Many hyperparameters", "Requires XGBoost library",
                           "Slower training than Random Forest"],
            "best_for": "When you need maximum accuracy on tabular classification data",
            "complexity": "Medium–High",
            "training_speed": "⚡ Fast (with GPU)",
            "interpretability": "★★☆☆☆",
            "code": _code_xgboost(task, target_col, num_feat_cols, cat_cols),
            "reasons": r4 or ["Industry-standard for tabular classification."],
        })

        # ── Naive Bayes ────────────────────────────────────────────────────
        score5 = 62
        if is_large:
            score5 += 10
        if len(cat_cols) > 0:
            score5 -= 5
        models.append({
            "id": "naive_bayes",
            "name": "Naive Bayes (Gaussian)",
            "score": min(score5, 100),
            "tags": ["Probabilistic", "Very Fast", "Text-Friendly"],
            "why": "Extremely fast classifier based on Bayes' theorem. "
                   "Works surprisingly well even with the naive independence assumption.",
            "strengths": ["Blazing fast training and inference", "Works well with large data",
                          "Naturally outputs probabilities", "Simple to understand"],
            "weaknesses": ["Assumes feature independence (rarely true)",
                           "Poor with correlated features", "Works best with numeric features"],
            "best_for": "Very large datasets, text classification, real-time systems",
            "complexity": "Low",
            "training_speed": "⚡ Extremely Fast",
            "interpretability": "★★★☆☆",
            "code": _code_naive_bayes(target_col, num_feat_cols, cat_cols),
            "reasons": ["Very fast — good for large datasets."],
        })

        # ── SVM Classifier ─────────────────────────────────────────────────
        score6 = 72
        if is_large:
            score6 -= 25
        if is_small:
            score6 += 12
        models.append({
            "id": "svm_clf",
            "name": "Support Vector Machine (SVM)",
            "score": max(score6, 30),
            "tags": ["Kernel-based", "High-Dimensional", "Small Data"],
            "why": "Uses a hyperplane (or kernel) to separate classes. "
                   "Excellent in high-dimensional spaces where other models struggle.",
            "strengths": ["Very effective in high-dimensional spaces",
                          "Robust to overfitting with right C/kernel",
                          "Works well with small, clean datasets"],
            "weaknesses": ["Very slow on large datasets",
                           "Requires feature scaling",
                           "Hard to interpret"],
            "best_for": "Small datasets (<5k rows), high-dimensional feature spaces",
            "complexity": "High",
            "training_speed": "🐢 Slow on large data",
            "interpretability": "★☆☆☆☆",
            "code": _code_svm(task, target_col, num_feat_cols, cat_cols),
            "reasons": ["Best for small, high-dimensional datasets."],
        })

        # ── KNN ────────────────────────────────────────────────────────────
        score7 = 60
        if is_large:
            score7 -= 20
        if is_small:
            score7 += 10
        models.append({
            "id": "knn_clf",
            "name": "K-Nearest Neighbours (KNN)",
            "score": max(score7, 25),
            "tags": ["Instance-based", "No Training", "Simple"],
            "why": "Classifies based on the k most similar training examples. "
                   "No training phase — prediction uses all training data.",
            "strengths": ["No training required", "Naturally handles multi-class",
                          "Simple to understand", "No assumption about data distribution"],
            "weaknesses": ["Slow prediction on large datasets (O(n) per query)",
                           "Memory intensive (stores all training data)",
                           "Poor with high-dimensional data (curse of dimensionality)"],
            "best_for": "Small datasets (<3k rows) where patterns are local",
            "complexity": "Low (conceptually) / High (computationally)",
            "training_speed": "🐢 Slow prediction",
            "interpretability": "★★★★☆",
            "code": _code_knn(task, target_col, num_feat_cols, cat_cols),
            "reasons": ["Simple, intuitive model for small datasets."],
        })

    # Sort by score descending and add rank
    models.sort(key=lambda m: m["score"], reverse=True)
    for i, m in enumerate(models):
        m["rank"] = i + 1
        m["medal"] = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣"][min(i, 6)]

    return models


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def recommend_models(
    path: Path,
    target_col: str | None = None,
) -> dict[str, Any]:
    """
    Main entry point. Returns ranked model recommendations for the dataset.
    """
    try:
        df = load_dataset(path)
    except ValueError as exc:
        return {"handled": False, "error": str(exc)}

    rows, cols = df.shape

    if cols < 2:
        return {
            "handled": False,
            "error": "Dataset must have at least 2 columns (features + target).",
        }

    # Auto-detect target if not provided
    if not target_col or target_col not in df.columns:
        target_col = str(df.columns[-1])

    task = _infer_task(df, target_col)
    profile = _dataset_profile(df)
    ranked = _score_models(task, profile, target_col)
    target_is_categorical = (
        task == "classification"
        and not pd.api.types.is_numeric_dtype(df[target_col])
    )
    if target_is_categorical:
        for model in ranked:
            model["code"] = _add_categorical_target_encoding(model["code"])

    num_cols_for_code = [c for c in profile["numeric_cols"] if c != target_col]
    cat_cols_for_code = [c for c in profile["categorical_cols"] if c != target_col]

    return {
        "handled": True,
        "filename": path.name,
        "rows": rows,
        "columns": cols,
        "target_column": target_col,
        "task": task,
        "target_is_categorical": target_is_categorical,
        "target_encoding": "LabelEncoder" if target_is_categorical else None,
        "task_label": (
            "📈 Regression — predicting a continuous numeric value"
            if task == "regression"
            else "🏷️ Classification — predicting a category or class label"
        ),
        "n_numeric_features": len(num_cols_for_code),
        "n_categorical_features": len(cat_cols_for_code),
        "has_missing": profile["has_missing"],
        "missing_pct": profile["missing_pct"],
        "dataset_size_label": (
            "Small (<1k rows)"
            if profile["is_small"]
            else "Medium (1k–50k rows)"
            if profile["is_medium"]
            else "Large (>50k rows)"
        ),
        "models": ranked,
    }
