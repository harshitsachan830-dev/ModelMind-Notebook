import ast
from typing import Any


VALID_LEVELS = {"basic", "medium", "advanced"}

LEVEL_ALIASES = {
    "beginner": "basic",
    "basic": "basic",
    "intermediate": "medium",
    "medium": "medium",
    "advanced": "advanced",
    "expert": "advanced",
}


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------


def analyze_ml_mistakes(
    code: str,
    level: str = "Basic",
) -> dict[str, Any]:
    """
    Analyze Python / ML code that may run successfully but contains
    risky or incorrect machine-learning methodology.

    This function does NOT execute user code.
    It performs static AST-based analysis.
    """

    learning_level = normalize_level(level)

    result = {
        "handled": True,
        "source": "modelmind-local",
        "engine": "ml-mistake-detector",
        "level": learning_level,
        "parse_success": True,
        "findings": [],
        "finding_count": 0,
    }

    if not code or not code.strip():
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "handled": False,
            "source": "modelmind-local",
            "engine": "ml-mistake-detector",
            "level": learning_level,
            "parse_success": False,
            "findings": [],
            "finding_count": 0,
            "message": (
                "ML Mistake Detector only analyzes code that can be "
                "parsed successfully. Use Error Intelligence for the "
                "syntax error first."
            ),
            "line_number": exc.lineno,
        }

    context = collect_context(tree)

    findings: list[dict[str, Any]] = []

    detectors = [
        detect_preprocessing_before_split,
        detect_test_fit_transform,
        detect_fit_on_test_data,
        detect_training_only_evaluation,
        detect_target_leakage,
        detect_feature_selection_before_split,
        detect_resampling_before_split,
        detect_accuracy_imbalance_risk,
        detect_missing_random_state,
        detect_unnecessary_tree_scaling,
        detect_missing_scaling_for_sensitive_models,
        detect_metric_model_mismatch,
    ]

    for detector in detectors:
        detector_findings = detector(
            tree=tree,
            context=context,
            level=learning_level,
        )

        findings.extend(detector_findings)

    findings = remove_duplicate_findings(findings)

    findings.sort(
        key=lambda finding: (
            severity_rank(finding.get("severity", "info")),
            finding.get("line_number") or 10**9,
        )
    )

    result["findings"] = findings
    result["finding_count"] = len(findings)

    return result


# ---------------------------------------------------------
# Level helpers
# ---------------------------------------------------------


def normalize_level(level: str | None) -> str:
    if not level:
        return "basic"

    normalized = str(level).strip().lower()

    return LEVEL_ALIASES.get(normalized, "basic")


def choose_level_text(
    level: str,
    basic: str,
    medium: str,
    advanced: str,
) -> str:
    if level == "advanced":
        return advanced

    if level == "medium":
        return medium

    return basic


# ---------------------------------------------------------
# Finding builder
# ---------------------------------------------------------


def make_finding(
    *,
    finding_id: str,
    severity: str,
    confidence: float,
    category: str,
    title: str,
    what_happened: str,
    why_basic: str,
    why_medium: str,
    why_advanced: str,
    recommendation_basic: str,
    recommendation_medium: str,
    recommendation_advanced: str,
    code_basic: str = "",
    code_medium: str = "",
    code_advanced: str = "",
    line_number: int | None = None,
    level: str = "basic",
    evidence: str = "",
) -> dict[str, Any]:

    return {
        "id": finding_id,
        "severity": severity,
        "confidence": confidence,
        "category": category,
        "title": title,
        "what_happened": what_happened,
        "why_it_matters": choose_level_text(
            level,
            why_basic,
            why_medium,
            why_advanced,
        ),
        "recommendation": choose_level_text(
            level,
            recommendation_basic,
            recommendation_medium,
            recommendation_advanced,
        ),
        "code_example": choose_level_text(
            level,
            code_basic,
            code_medium,
            code_advanced,
        ),
        "line_number": line_number,
        "learning_level": level,
        "evidence": evidence,
        "safe_to_apply": False,
        "source": "modelmind-local",
    }


# ---------------------------------------------------------
# AST helpers
# ---------------------------------------------------------


def get_call_name(node: ast.AST) -> str:
    """
    Examples:

    train_test_split(...)              -> train_test_split
    scaler.fit_transform(...)          -> scaler.fit_transform
    sklearn.metrics.accuracy_score(...) -> sklearn.metrics.accuracy_score
    """

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = get_call_name(node.value)

        if parent:
            return f"{parent}.{node.attr}"

        return node.attr

    return ""


def get_simple_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return ""


def expression_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def get_assignment_names(node: ast.AST) -> list[str]:
    names: list[str] = []

    if isinstance(node, ast.Name):
        names.append(node.id)

    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            names.extend(get_assignment_names(element))

    return names


def get_literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    return None


def get_literal_list_strings(node: ast.AST) -> list[str]:
    values: list[str] = []

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for element in node.elts:
            value = get_literal_string(element)

            if value is not None:
                values.append(value)

    return values


def has_keyword(call: ast.Call, keyword_name: str) -> bool:
    return any(
        keyword.arg == keyword_name
        for keyword in call.keywords
        if keyword.arg is not None
    )


def get_keyword(call: ast.Call, keyword_name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value

    return None


def call_contains_argument_name(
    call: ast.Call,
    names: set[str],
) -> bool:

    for argument in call.args:
        if isinstance(argument, ast.Name):
            if argument.id in names:
                return True

    for keyword in call.keywords:
        value = keyword.value

        if isinstance(value, ast.Name):
            if value.id in names:
                return True

    return False


def severity_rank(severity: str) -> int:
    ranks = {
        "critical": 0,
        "high": 1,
        "warning": 2,
        "info": 3,
    }

    return ranks.get(severity, 4)


# ---------------------------------------------------------
# Context collection
# ---------------------------------------------------------


def collect_context(tree: ast.AST) -> dict[str, Any]:

    context: dict[str, Any] = {
        "calls": [],
        "assignments": [],
        "constructors": {},
        "split_lines": [],
        "target_columns": set(),
        "feature_columns": set(),
        "feature_assignments": {},
        "target_assignments": {},
        "model_variables": {},
        "scaler_variables": set(),
        "preprocessor_variables": set(),
        "feature_selector_variables": set(),
        "resampler_variables": set(),
        "classification_models": set(),
        "regression_models": set(),
        "tree_models": set(),
        "scale_sensitive_models": set(),
        "metric_calls": [],
    }

    for node in ast.walk(tree):

        if isinstance(node, ast.Call):
            call_name = get_call_name(node.func)

            context["calls"].append(
                {
                    "node": node,
                    "name": call_name,
                    "line": getattr(node, "lineno", None),
                }
            )

            short_name = call_name.split(".")[-1]

            if short_name == "train_test_split":
                context["split_lines"].append(
                    getattr(node, "lineno", None)
                )

            if short_name in {
                "accuracy_score",
                "precision_score",
                "recall_score",
                "f1_score",
                "roc_auc_score",
                "mean_squared_error",
                "mean_absolute_error",
                "r2_score",
            }:
                context["metric_calls"].append(
                    {
                        "name": short_name,
                        "node": node,
                        "line": getattr(node, "lineno", None),
                    }
                )

        if isinstance(node, (ast.Assign, ast.AnnAssign)):

            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value

            if value is None:
                continue

            assigned_names: list[str] = []

            for target in targets:
                assigned_names.extend(
                    get_assignment_names(target)
                )

            context["assignments"].append(
                {
                    "names": assigned_names,
                    "value": value,
                    "line": getattr(node, "lineno", None),
                }
            )

            if isinstance(value, ast.Call):
                constructor_name = get_call_name(value.func)
                short_constructor = constructor_name.split(".")[-1]

                for assigned_name in assigned_names:
                    context["constructors"][
                        assigned_name
                    ] = short_constructor

                    register_model_or_transformer(
                        variable_name=assigned_name,
                        constructor_name=short_constructor,
                        context=context,
                    )

            register_dataframe_assignment(
                assigned_names=assigned_names,
                value=value,
                line_number=getattr(node, "lineno", None),
                context=context,
            )

    return context


def register_model_or_transformer(
    *,
    variable_name: str,
    constructor_name: str,
    context: dict[str, Any],
) -> None:

    classification_models = {
        "LogisticRegression",
        "KNeighborsClassifier",
        "SVC",
        "LinearSVC",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
        "GaussianNB",
        "MultinomialNB",
        "BernoulliNB",
        "MLPClassifier",
        "Perceptron",
    }

    regression_models = {
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNeighborsRegressor",
        "SVR",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor",
        "CatBoostRegressor",
        "MLPRegressor",
    }

    tree_models = {
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
    }

    scale_sensitive_models = {
        "LogisticRegression",
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "SVC",
        "SVR",
        "LinearSVC",
        "Perceptron",
        "MLPClassifier",
        "MLPRegressor",
    }

    scalers = {
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "MaxAbsScaler",
        "Normalizer",
    }

    preprocessors = {
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
        "OneHotEncoder",
        "OrdinalEncoder",
        "LabelEncoder",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "MaxAbsScaler",
        "Normalizer",
        "PolynomialFeatures",
        "PowerTransformer",
        "QuantileTransformer",
    }

    feature_selectors = {
        "SelectKBest",
        "SelectPercentile",
        "RFE",
        "RFECV",
        "VarianceThreshold",
        "SelectFromModel",
    }

    resamplers = {
        "SMOTE",
        "ADASYN",
        "RandomOverSampler",
        "RandomUnderSampler",
        "NearMiss",
    }

    if constructor_name in classification_models:
        context["classification_models"].add(variable_name)
        context["model_variables"][variable_name] = constructor_name

    if constructor_name in regression_models:
        context["regression_models"].add(variable_name)
        context["model_variables"][variable_name] = constructor_name

    if constructor_name in tree_models:
        context["tree_models"].add(variable_name)

    if constructor_name in scale_sensitive_models:
        context["scale_sensitive_models"].add(variable_name)

    if constructor_name in scalers:
        context["scaler_variables"].add(variable_name)

    if constructor_name in preprocessors:
        context["preprocessor_variables"].add(variable_name)

    if constructor_name in feature_selectors:
        context["feature_selector_variables"].add(variable_name)

    if constructor_name in resamplers:
        context["resampler_variables"].add(variable_name)


def register_dataframe_assignment(
    *,
    assigned_names: list[str],
    value: ast.AST,
    line_number: int | None,
    context: dict[str, Any],
) -> None:

    if not assigned_names:
        return

    # Example:
    # y = df["target"]

    if isinstance(value, ast.Subscript):
        column = extract_dataframe_column(value)

        if column is not None:
            for assigned_name in assigned_names:
                if assigned_name.lower() in {
                    "y",
                    "target",
                    "label",
                    "labels",
                }:
                    context["target_columns"].add(column)
                    context["target_assignments"][assigned_name] = {
                        "column": column,
                        "line": line_number,
                    }

        columns = extract_dataframe_columns(value)

        if columns:
            for assigned_name in assigned_names:
                if assigned_name.lower() in {
                    "x",
                    "features",
                    "x_data",
                    "feature_data",
                }:
                    context["feature_columns"].update(columns)
                    context["feature_assignments"][assigned_name] = {
                        "columns": columns,
                        "line": line_number,
                    }


def extract_dataframe_column(
    node: ast.Subscript,
) -> str | None:

    slice_node = node.slice

    return get_literal_string(slice_node)


def extract_dataframe_columns(
    node: ast.Subscript,
) -> list[str]:

    slice_node = node.slice

    return get_literal_list_strings(slice_node)


# ---------------------------------------------------------
# Detector 1
# Preprocessing before train/test split
# ---------------------------------------------------------


def detect_preprocessing_before_split(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    split_lines = [
        line
        for line in context["split_lines"]
        if line is not None
    ]

    if not split_lines:
        return findings

    first_split_line = min(split_lines)

    risky_methods = {
        "fit",
        "fit_transform",
    }

    for call_info in context["calls"]:
        call = call_info["node"]
        call_name = call_info["name"]
        line = call_info["line"]

        if line is None or line >= first_split_line:
            continue

        parts = call_name.split(".")

        if len(parts) < 2:
            continue

        variable = parts[-2]
        method = parts[-1]

        if method not in risky_methods:
            continue

        is_preprocessor = (
            variable in context["preprocessor_variables"]
            or variable in context["scaler_variables"]
        )

        if not is_preprocessor:
            continue

        findings.append(
            make_finding(
                finding_id="preprocessing-before-split",
                severity="high",
                confidence=0.99,
                category="data-leakage",
                title="Preprocessing fitted before train/test split",
                what_happened=(
                    f"`{call_name}()` is called before "
                    "`train_test_split()`."
                ),
                why_basic=(
                    "The preprocessing step can learn information from "
                    "data that later becomes test data. That makes the "
                    "test result less trustworthy."
                ),
                why_medium=(
                    "Fitting preprocessing on the complete dataset "
                    "allows statistics such as mean, median, scale or "
                    "category information from the future test set to "
                    "influence training."
                ),
                why_advanced=(
                    "The preprocessing estimator is fitted outside the "
                    "training boundary. This leaks distributional "
                    "information from the holdout set into the learned "
                    "transformation and can bias generalization metrics."
                ),
                recommendation_basic=(
                    "Split the data first. Learn preprocessing from "
                    "X_train, then use the same learned transformation "
                    "on X_test."
                ),
                recommendation_medium=(
                    "Run train_test_split first. Use fit_transform on "
                    "X_train and transform only on X_test."
                ),
                recommendation_advanced=(
                    "Move preprocessing inside a Pipeline or "
                    "ColumnTransformer so fitting occurs only on the "
                    "training partition and remains leakage-safe during "
                    "cross-validation."
                ),
                code_basic=(
                    "X_train, X_test, y_train, y_test = "
                    "train_test_split(X, y, test_size=0.2)\n"
                    "X_train = scaler.fit_transform(X_train)\n"
                    "X_test = scaler.transform(X_test)"
                ),
                code_medium=(
                    "X_train, X_test, y_train, y_test = "
                    "train_test_split(\n"
                    "    X, y, test_size=0.2, random_state=42\n"
                    ")\n"
                    "X_train = scaler.fit_transform(X_train)\n"
                    "X_test = scaler.transform(X_test)"
                ),
                code_advanced=(
                    "pipeline = Pipeline([\n"
                    '    ("scaler", StandardScaler()),\n'
                    '    ("model", LogisticRegression())\n'
                    "])\n\n"
                    "pipeline.fit(X_train, y_train)\n"
                    "predictions = pipeline.predict(X_test)"
                ),
                line_number=line,
                level=level,
                evidence=expression_text(call),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 2
# fit_transform on test data
# ---------------------------------------------------------


def detect_test_fit_transform(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    for call_info in context["calls"]:
        call = call_info["node"]
        call_name = call_info["name"]

        if not call_name.endswith(".fit_transform"):
            continue

        if not call.args:
            continue

        first_argument = expression_text(call.args[0]).lower()

        if not looks_like_test_data(first_argument):
            continue

        findings.append(
            make_finding(
                finding_id="fit-transform-test-data",
                severity="high",
                confidence=0.99,
                category="data-leakage",
                title="Preprocessor is being fitted on test data",
                what_happened=(
                    f"`{call_name}()` is being applied to "
                    f"`{expression_text(call.args[0])}`."
                ),
                why_basic=(
                    "The test data should only be transformed using "
                    "what was learned from the training data."
                ),
                why_medium=(
                    "fit_transform learns new preprocessing statistics "
                    "from the test set instead of preserving the "
                    "training transformation."
                ),
                why_advanced=(
                    "Calling fit_transform on the holdout partition "
                    "creates an independent transformation fitted to "
                    "test-distribution statistics, invalidating the "
                    "intended train/test isolation."
                ),
                recommendation_basic=(
                    "Use fit_transform on X_train and transform on "
                    "X_test."
                ),
                recommendation_medium=(
                    "Fit the transformer once on training data, then "
                    "reuse that fitted transformer for test data."
                ),
                recommendation_advanced=(
                    "Use a fitted Pipeline/ColumnTransformer and call "
                    "predict directly on untouched X_test."
                ),
                code_basic=(
                    "X_train = scaler.fit_transform(X_train)\n"
                    "X_test = scaler.transform(X_test)"
                ),
                code_medium=(
                    "scaler.fit(X_train)\n"
                    "X_train_scaled = scaler.transform(X_train)\n"
                    "X_test_scaled = scaler.transform(X_test)"
                ),
                code_advanced=(
                    "pipeline.fit(X_train, y_train)\n"
                    "predictions = pipeline.predict(X_test)"
                ),
                line_number=call_info["line"],
                level=level,
                evidence=expression_text(call),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 3
# fit model on test data
# ---------------------------------------------------------


def detect_fit_on_test_data(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    for call_info in context["calls"]:
        call = call_info["node"]
        call_name = call_info["name"]

        if not call_name.endswith(".fit"):
            continue

        parts = call_name.split(".")

        if len(parts) < 2:
            continue

        variable = parts[-2]

        if variable not in context["model_variables"]:
            continue

        if not call.args:
            continue

        argument_text = " ".join(
            expression_text(argument).lower()
            for argument in call.args
        )

        if not (
            "x_test" in argument_text
            or "y_test" in argument_text
            or "test_x" in argument_text
            or "test_y" in argument_text
        ):
            continue

        findings.append(
            make_finding(
                finding_id="model-fit-on-test-data",
                severity="critical",
                confidence=0.99,
                category="data-leakage",
                title="Model is being trained on test data",
                what_happened=(
                    f"`{call_name}()` receives variables that appear "
                    "to belong to the test set."
                ),
                why_basic=(
                    "The test set is supposed to check the model after "
                    "training. Training on it means it is no longer a "
                    "fair test."
                ),
                why_medium=(
                    "Using test samples during fit contaminates the "
                    "holdout set and produces overly optimistic "
                    "evaluation results."
                ),
                why_advanced=(
                    "The holdout partition has entered parameter "
                    "estimation. This destroys independence between "
                    "training and evaluation and invalidates the "
                    "generalization estimate."
                ),
                recommendation_basic=(
                    "Train the model using X_train and y_train only."
                ),
                recommendation_medium=(
                    "Keep X_test and y_test completely outside model "
                    "fitting and use them only for final evaluation."
                ),
                recommendation_advanced=(
                    "Enforce a strict training/evaluation boundary. "
                    "Use cross-validation inside the training set for "
                    "model selection and reserve the holdout set for "
                    "final evaluation."
                ),
                code_basic=(
                    "model.fit(X_train, y_train)\n"
                    "predictions = model.predict(X_test)"
                ),
                code_medium=(
                    "model.fit(X_train, y_train)\n"
                    "y_pred = model.predict(X_test)"
                ),
                code_advanced=(
                    "search.fit(X_train, y_train)\n"
                    "final_predictions = search.predict(X_test)"
                ),
                line_number=call_info["line"],
                level=level,
                evidence=expression_text(call),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 4
# Training-only evaluation
# ---------------------------------------------------------


def detect_training_only_evaluation(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    training_score_calls = []
    test_score_found = False

    for call_info in context["calls"]:
        call = call_info["node"]
        call_name = call_info["name"]

        if call_name.endswith(".score"):
            arguments = " ".join(
                expression_text(argument).lower()
                for argument in call.args
            )

            if "test" in arguments:
                test_score_found = True

            if "train" in arguments:
                training_score_calls.append(call_info)

    if test_score_found:
        return findings

    for call_info in training_score_calls:
        findings.append(
            make_finding(
                finding_id="training-only-evaluation",
                severity="warning",
                confidence=0.90,
                category="evaluation",
                title="Model appears to be evaluated only on training data",
                what_happened=(
                    "A score is calculated using training variables, "
                    "but no test score was detected."
                ),
                why_basic=(
                    "A model can perform very well on data it already "
                    "saw. You also need to test it on unseen data."
                ),
                why_medium=(
                    "Training performance measures fit quality but does "
                    "not provide a reliable estimate of generalization."
                ),
                why_advanced=(
                    "In-sample performance cannot substitute for an "
                    "out-of-sample generalization estimate. Compare "
                    "training and validation/test performance to "
                    "diagnose variance and overfitting."
                ),
                recommendation_basic=(
                    "Also calculate the model score using X_test and "
                    "y_test."
                ),
                recommendation_medium=(
                    "Report both training and test performance and "
                    "compare the gap."
                ),
                recommendation_advanced=(
                    "Use validation/cross-validation for model "
                    "selection and a separate holdout set for final "
                    "generalization assessment."
                ),
                code_basic=(
                    "print(model.score(X_train, y_train))\n"
                    "print(model.score(X_test, y_test))"
                ),
                code_medium=(
                    "train_score = model.score(X_train, y_train)\n"
                    "test_score = model.score(X_test, y_test)"
                ),
                code_advanced=(
                    "cv_scores = cross_val_score(\n"
                    "    model, X_train, y_train, cv=5\n"
                    ")\n"
                    "test_score = model.score(X_test, y_test)"
                ),
                line_number=call_info["line"],
                level=level,
                evidence=expression_text(call_info["node"]),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 5
# Target leakage
# ---------------------------------------------------------


def detect_target_leakage(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    overlap = (
        context["target_columns"]
        & context["feature_columns"]
    )

    for column in sorted(overlap):
        feature_line = None

        for assignment in context["feature_assignments"].values():
            if column in assignment.get("columns", []):
                feature_line = assignment.get("line")
                break

        findings.append(
            make_finding(
                finding_id=f"target-leakage-{column}",
                severity="critical",
                confidence=0.99,
                category="target-leakage",
                title="Target column is included in the feature matrix",
                what_happened=(
                    f'The column "{column}" appears in both the '
                    "target and feature definitions."
                ),
                why_basic=(
                    "The model is being given the answer as one of its "
                    "inputs, so its result will not represent real "
                    "prediction."
                ),
                why_medium=(
                    "Including the target in X creates direct target "
                    "leakage and can produce unrealistically high "
                    "evaluation scores."
                ),
                why_advanced=(
                    "The predictor matrix contains the response "
                    "variable itself, violating the information set "
                    "available at inference time and invalidating "
                    "generalization metrics."
                ),
                recommendation_basic=(
                    "Remove the target column from X."
                ),
                recommendation_medium=(
                    "Create y from the target column and create X by "
                    "dropping that target."
                ),
                recommendation_advanced=(
                    "Define the inference-time feature schema "
                    "explicitly and exclude the response plus any "
                    "post-outcome proxy variables."
                ),
                code_basic=(
                    f'y = df["{column}"]\n'
                    f'X = df.drop(columns=["{column}"])'
                ),
                code_medium=(
                    f'target = "{column}"\n'
                    "y = df[target]\n"
                    "X = df.drop(columns=[target])"
                ),
                code_advanced=(
                    f'target = "{column}"\n'
                    "feature_columns = [\n"
                    "    col for col in df.columns\n"
                    "    if col != target\n"
                    "]\n"
                    "X = df[feature_columns]\n"
                    "y = df[target]"
                ),
                line_number=feature_line,
                level=level,
                evidence=(
                    f'Target column "{column}" also appears in X.'
                ),
            )
        )

    return findings
# ---------------------------------------------------------
# Detector 6
# Feature selection before split
# ---------------------------------------------------------


def detect_feature_selection_before_split(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    split_lines = [
        line
        for line in context["split_lines"]
        if line is not None
    ]

    if not split_lines:
        return findings

    first_split_line = min(split_lines)

    for call_info in context["calls"]:
        call_name = call_info["name"]
        line = call_info["line"]

        if line is None or line >= first_split_line:
            continue

        if not (
            call_name.endswith(".fit")
            or call_name.endswith(".fit_transform")
        ):
            continue

        parts = call_name.split(".")

        if len(parts) < 2:
            continue

        variable = parts[-2]

        if variable not in context["feature_selector_variables"]:
            continue

        findings.append(
            make_finding(
                finding_id="feature-selection-before-split",
                severity="high",
                confidence=0.98,
                category="data-leakage",
                title="Feature selection fitted before data split",
                what_happened=(
                    "The feature selector learns from the complete "
                    "dataset before the holdout set is created."
                ),
                why_basic=(
                    "The feature selector can learn from data that "
                    "should have been kept unseen."
                ),
                why_medium=(
                    "Supervised or statistical feature selection before "
                    "splitting can leak test-set information into the "
                    "selected feature set."
                ),
                why_advanced=(
                    "Feature-selection parameters are being estimated "
                    "outside the training fold, causing selection bias "
                    "and optimistic validation estimates."
                ),
                recommendation_basic=(
                    "Split first, then fit feature selection using "
                    "training data."
                ),
                recommendation_medium=(
                    "Fit the selector only on X_train and apply its "
                    "transform to X_test."
                ),
                recommendation_advanced=(
                    "Place feature selection inside a Pipeline so it "
                    "is refitted independently inside every "
                    "cross-validation fold."
                ),
                code_basic=(
                    "X_train, X_test, y_train, y_test = "
                    "train_test_split(X, y)\n"
                    "X_train = selector.fit_transform(X_train, y_train)\n"
                    "X_test = selector.transform(X_test)"
                ),
                code_medium=(
                    "selector.fit(X_train, y_train)\n"
                    "X_train_selected = selector.transform(X_train)\n"
                    "X_test_selected = selector.transform(X_test)"
                ),
                code_advanced=(
                    "pipeline = Pipeline([\n"
                    '    ("selection", SelectKBest(k=5)),\n'
                    '    ("model", LogisticRegression())\n'
                    "])"
                ),
                line_number=line,
                level=level,
                evidence=expression_text(call_info["node"]),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 7
# Resampling before split
# ---------------------------------------------------------


def detect_resampling_before_split(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    split_lines = [
        line
        for line in context["split_lines"]
        if line is not None
    ]

    if not split_lines:
        return findings

    first_split_line = min(split_lines)

    for call_info in context["calls"]:
        call_name = call_info["name"]
        line = call_info["line"]

        if line is None or line >= first_split_line:
            continue

        if not call_name.endswith(".fit_resample"):
            continue

        parts = call_name.split(".")

        if len(parts) < 2:
            continue

        variable = parts[-2]

        if variable not in context["resampler_variables"]:
            continue

        findings.append(
            make_finding(
                finding_id="resampling-before-split",
                severity="critical",
                confidence=0.99,
                category="data-leakage",
                title="Resampling performed before train/test split",
                what_happened=(
                    f"`{call_name}()` is called before the dataset "
                    "is split."
                ),
                why_basic=(
                    "Synthetic or resampled information can enter both "
                    "training and test data, making the test unfair."
                ),
                why_medium=(
                    "Oversampling before splitting can create related "
                    "or synthetic observations that contaminate the "
                    "holdout set."
                ),
                why_advanced=(
                    "Resampling outside the training boundary allows "
                    "information derived from the complete sample to "
                    "affect the holdout distribution and biases model "
                    "selection/evaluation."
                ),
                recommendation_basic=(
                    "Split the original data first. Resample only the "
                    "training data."
                ),
                recommendation_medium=(
                    "Apply SMOTE or other sampling methods only to "
                    "X_train and y_train."
                ),
                recommendation_advanced=(
                    "Use an imbalanced-learn Pipeline so resampling "
                    "occurs independently inside each training fold."
                ),
                code_basic=(
                    "X_train, X_test, y_train, y_test = "
                    "train_test_split(X, y)\n"
                    "X_train, y_train = smote.fit_resample(\n"
                    "    X_train, y_train\n"
                    ")"
                ),
                code_medium=(
                    "X_train_resampled, y_train_resampled = "
                    "smote.fit_resample(X_train, y_train)"
                ),
                code_advanced=(
                    "pipeline = ImbPipeline([\n"
                    '    ("smote", SMOTE(random_state=42)),\n'
                    '    ("model", LogisticRegression())\n'
                    "])"
                ),
                line_number=line,
                level=level,
                evidence=expression_text(call_info["node"]),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 8
# Accuracy + imbalance risk
# ---------------------------------------------------------


def detect_accuracy_imbalance_risk(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []
        # Accuracy imbalance guidance only applies when this code
    # actually contains a classification estimator.
    #
    # If a regression estimator uses accuracy_score, the metric/model
    # mismatch detector will handle that separately.
    if (
        context["regression_models"]
        and not context["classification_models"]
    ):
        return findings

    accuracy_calls = [
        metric
        for metric in context["metric_calls"]
        if metric["name"] == "accuracy_score"
    ]

    if not accuracy_calls:
        return findings

    # Static analysis cannot prove that the dataset is imbalanced.
    # Therefore this is intentionally an informational risk notice,
    # not a claim that imbalance exists.

    for metric in accuracy_calls:
        findings.append(
            make_finding(
                finding_id="accuracy-imbalance-risk",
                severity="info",
                confidence=0.72,
                category="evaluation",
                title="Check class balance before relying on accuracy",
                what_happened=(
                    "The classification workflow uses accuracy_score."
                ),
                why_basic=(
                    "Accuracy can look high when one class appears much "
                    "more often than another."
                ),
                why_medium=(
                    "Accuracy alone may hide poor minority-class "
                    "performance when the target distribution is "
                    "imbalanced."
                ),
                why_advanced=(
                    "Accuracy weights every observation equally and can "
                    "be uninformative under asymmetric prevalence or "
                    "misclassification costs. This static check does "
                    "not claim that imbalance actually exists."
                ),
                recommendation_basic=(
                    "Check how many examples belong to each class. If "
                    "the classes are uneven, also inspect precision, "
                    "recall and F1-score."
                ),
                recommendation_medium=(
                    "Inspect class frequencies and supplement accuracy "
                    "with precision, recall, F1 and a confusion matrix "
                    "when appropriate."
                ),
                recommendation_advanced=(
                    "Inspect prevalence and task costs, then choose "
                    "metrics such as class-specific precision/recall, "
                    "F1, balanced accuracy, ROC-AUC or PR-AUC according "
                    "to the decision objective."
                ),
                code_basic=(
                    "print(y.value_counts())\n"
                    "print(accuracy_score(y_test, y_pred))\n"
                    "print(f1_score(y_test, y_pred))"
                ),
                code_medium=(
                    "print(y_train.value_counts(normalize=True))\n"
                    "print(classification_report(y_test, y_pred))"
                ),
                code_advanced=(
                    "print(y_train.value_counts(normalize=True))\n"
                    "print(classification_report(y_test, y_pred))\n"
                    "# Choose metrics according to prevalence and "
                    "error costs."
                ),
                line_number=metric["line"],
                level=level,
                evidence=expression_text(metric["node"]),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 9
# Missing random_state
# ---------------------------------------------------------


def detect_missing_random_state(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:
    findings = []

    for call_info in context["calls"]:
        call = call_info["node"]
        short_name = call_info["name"].split(".")[-1]

        if short_name != "train_test_split":
            continue

        if has_keyword(call, "random_state"):
            continue

        findings.append(
            make_finding(
                finding_id="missing-random-state",
                severity="info",
                confidence=0.99,
                category="reproducibility",
                title="Train/test split has no random_state",
                what_happened=(
                    "`train_test_split()` does not specify "
                    "`random_state`."
                ),
                why_basic=(
                    "Your train and test rows may change when you run "
                    "the notebook again."
                ),
                why_medium=(
                    "Without a fixed random seed, repeated executions "
                    "can produce different splits and different "
                    "evaluation results."
                ),
                why_advanced=(
                    "The stochastic partition is not reproducibly "
                    "controlled, which makes experiment comparison and "
                    "debugging harder. This is not inherently a model "
                    "error."
                ),
                recommendation_basic=(
                    "Add random_state=42 while learning and comparing "
                    "experiments."
                ),
                recommendation_medium=(
                    "Use a fixed random_state for reproducible "
                    "experiments."
                ),
                recommendation_advanced=(
                    "Control experiment seeds explicitly and record "
                    "them with the experiment configuration. Do not "
                    "mistake a single seed for robustness analysis."
                ),
                code_basic=(
                    "train_test_split(\n"
                    "    X, y, test_size=0.2, random_state=42\n"
                    ")"
                ),
                code_medium=(
                    "X_train, X_test, y_train, y_test = "
                    "train_test_split(\n"
                    "    X,\n"
                    "    y,\n"
                    "    test_size=0.2,\n"
                    "    random_state=42,\n"
                    ")"
                ),
                code_advanced=(
                    "SEED = 42\n"
                    "X_train, X_test, y_train, y_test = "
                    "train_test_split(\n"
                    "    X, y, test_size=0.2, random_state=SEED\n"
                    ")"
                ),
                line_number=call_info["line"],
                level=level,
                evidence=expression_text(call),
            )
        )

    return findings


# ---------------------------------------------------------
# Detector 10
# Scaling tree models
# ---------------------------------------------------------


def detect_unnecessary_tree_scaling(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    if not context["tree_models"]:
        return findings

    if not context["scaler_variables"]:
        return findings

    scaler_usage = False

    for call_info in context["calls"]:
        name = call_info["name"]

        for scaler in context["scaler_variables"]:
            if (
                name == f"{scaler}.fit"
                or name == f"{scaler}.transform"
                or name == f"{scaler}.fit_transform"
            ):
                scaler_usage = True
                break

    if not scaler_usage:
        return findings

    tree_names = sorted(
        context["model_variables"].get(variable, variable)
        for variable in context["tree_models"]
    )

    findings.append(
        make_finding(
            finding_id="tree-model-scaling",
            severity="info",
            confidence=0.80,
            category="preprocessing",
            title="Scaling may be unnecessary for this tree model",
            what_happened=(
                "Feature scaling and a tree-based estimator both "
                "appear in the workflow."
            ),
            why_basic=(
                "Decision trees usually do not need features to be "
                "scaled because they make split decisions using "
                "feature thresholds."
            ),
            why_medium=(
                "Most tree-based models are largely invariant to "
                "monotonic feature scaling, so scaling is often "
                "unnecessary."
            ),
            why_advanced=(
                "Threshold-based tree partitioning generally preserves "
                "ordering under affine scaling, so standardization "
                "typically does not improve the tree's split geometry. "
                "There can still be pipeline-specific reasons to keep "
                "a transformer."
            ),
            recommendation_basic=(
                "You can usually train the tree model without "
                "StandardScaler."
            ),
            recommendation_medium=(
                "Remove scaling unless another part of your pipeline "
                "requires it."
            ),
            recommendation_advanced=(
                "Treat this as an optimization/clarity suggestion, not "
                "a correctness error. Retain preprocessing when it is "
                "required by another estimator or shared pipeline."
            ),
            code_basic=(
                "model.fit(X_train, y_train)"
            ),
            code_medium=(
                "# Tree models usually work directly with the "
                "prepared feature values.\n"
                "model.fit(X_train, y_train)"
            ),
            code_advanced=(
                "# Keep only transformations required by the "
                "feature schema or pipeline semantics."
            ),
            level=level,
            evidence=(
                "Tree model(s): "
                + ", ".join(tree_names)
            ),
        )
    )

    return findings


# ---------------------------------------------------------
# Detector 11
# Missing scaling for sensitive models
# ---------------------------------------------------------


def detect_missing_scaling_for_sensitive_models(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    if not context["scale_sensitive_models"]:
        return findings

    if context["scaler_variables"]:
        return findings

    models = sorted(
        context["model_variables"].get(variable, variable)
        for variable in context["scale_sensitive_models"]
    )

    findings.append(
        make_finding(
            finding_id="missing-scaling-sensitive-model",
            severity="warning",
            confidence=0.78,
            category="preprocessing",
            title="Check feature scaling for a scale-sensitive model",
            what_happened=(
                "A scale-sensitive estimator was detected, but no "
                "common feature scaler was found in this code."
            ),
            why_basic=(
                "Some models can be strongly affected when one feature "
                "uses much larger numbers than another."
            ),
            why_medium=(
                "Distance-based and optimization-sensitive algorithms "
                "often behave better when numerical features have "
                "comparable scales."
            ),
            why_advanced=(
                "Feature scale can affect distance geometry, margin "
                "optimization and numerical conditioning. Static "
                "analysis cannot determine whether the current feature "
                "scales already make scaling unnecessary."
            ),
            recommendation_basic=(
                "Check the ranges of your numerical features. For KNN, "
                "SVM and similar models, consider StandardScaler."
            ),
            recommendation_medium=(
                "Inspect feature distributions and scales, then fit the "
                "scaler using X_train only."
            ),
            recommendation_advanced=(
                "Use a leakage-safe Pipeline/ColumnTransformer and "
                "scale only features that require it. Validate whether "
                "scaling improves the selected estimator."
            ),
            code_basic=(
                "scaler = StandardScaler()\n"
                "X_train = scaler.fit_transform(X_train)\n"
                "X_test = scaler.transform(X_test)"
            ),
            code_medium=(
                "scaler = StandardScaler()\n"
                "X_train_scaled = scaler.fit_transform(X_train)\n"
                "X_test_scaled = scaler.transform(X_test)"
            ),
            code_advanced=(
                "pipeline = Pipeline([\n"
                '    ("scaler", StandardScaler()),\n'
                '    ("model", model)\n'
                "])"
            ),
            level=level,
            evidence=(
                "Scale-sensitive model(s): "
                + ", ".join(models)
            ),
        )
    )

    return findings


# ---------------------------------------------------------
# Detector 12
# Metric / model mismatch
# ---------------------------------------------------------


def detect_metric_model_mismatch(
    *,
    tree: ast.AST,
    context: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:

    findings = []

    classification_metrics = {
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "roc_auc_score",
    }

    regression_metrics = {
        "mean_squared_error",
        "mean_absolute_error",
        "r2_score",
    }

    has_classifier = bool(context["classification_models"])
    has_regressor = bool(context["regression_models"])

    for metric in context["metric_calls"]:
        metric_name = metric["name"]

        if has_regressor and metric_name in classification_metrics:
            findings.append(
                make_finding(
                    finding_id=(
                        f"classification-metric-with-regressor-"
                        f"{metric_name}"
                    ),
                    severity="high",
                    confidence=0.94,
                    category="evaluation",
                    title="Classification metric used with a regressor",
                    what_happened=(
                        f"`{metric_name}` appears in code containing "
                        "a regression estimator."
                    ),
                    why_basic=(
                        "This metric is designed for classification, "
                        "but your model predicts continuous values."
                    ),
                    why_medium=(
                        "Classification metrics expect discrete class "
                        "labels or class scores rather than ordinary "
                        "continuous regression predictions."
                    ),
                    why_advanced=(
                        "The evaluation functional does not match the "
                        "estimator's prediction target type. This can "
                        "produce an exception or a conceptually invalid "
                        "evaluation."
                    ),
                    recommendation_basic=(
                        "Use a regression metric such as MAE, MSE or "
                        "R²."
                    ),
                    recommendation_medium=(
                        "Choose MAE, RMSE/MSE or R² according to what "
                        "prediction error matters for your problem."
                    ),
                    recommendation_advanced=(
                        "Choose a loss/evaluation functional aligned "
                        "with the target scale and decision objective, "
                        "for example MAE for absolute-error robustness "
                        "or RMSE when larger errors deserve more weight."
                    ),
                    code_basic=(
                        "print(mean_absolute_error(y_test, y_pred))"
                    ),
                    code_medium=(
                        "mae = mean_absolute_error(y_test, y_pred)\n"
                        "mse = mean_squared_error(y_test, y_pred)\n"
                        "r2 = r2_score(y_test, y_pred)"
                    ),
                    code_advanced=(
                        "# Select regression metrics according to the "
                        "task's loss and target distribution."
                    ),
                    line_number=metric["line"],
                    level=level,
                    evidence=expression_text(metric["node"]),
                )
            )

        if has_classifier and metric_name in regression_metrics:
            findings.append(
                make_finding(
                    finding_id=(
                        f"regression-metric-with-classifier-"
                        f"{metric_name}"
                    ),
                    severity="high",
                    confidence=0.94,
                    category="evaluation",
                    title="Regression metric used with a classifier",
                    what_happened=(
                        f"`{metric_name}` appears in code containing "
                        "a classification estimator."
                    ),
                    why_basic=(
                        "This metric is meant for predicting numbers, "
                        "not ordinary class labels."
                    ),
                    why_medium=(
                        "Regression metrics measure continuous numeric "
                        "prediction error and generally do not represent "
                        "classification quality."
                    ),
                    why_advanced=(
                        "The metric's mathematical objective is not "
                        "aligned with discrete class prediction. Use "
                        "classification metrics appropriate to labels, "
                        "scores, prevalence and error costs."
                    ),
                    recommendation_basic=(
                        "Use classification metrics such as accuracy, "
                        "precision, recall or F1."
                    ),
                    recommendation_medium=(
                        "Choose classification metrics based on which "
                        "classification mistakes matter."
                    ),
                    recommendation_advanced=(
                        "Select threshold-dependent or ranking metrics "
                        "according to the decision objective, class "
                        "prevalence and misclassification costs."
                    ),
                    code_basic=(
                        "print(accuracy_score(y_test, y_pred))"
                    ),
                    code_medium=(
                        "print(classification_report(y_test, y_pred))"
                    ),
                    code_advanced=(
                        "# Evaluate labels and/or probability scores "
                        "with metrics aligned to the decision objective."
                    ),
                    line_number=metric["line"],
                    level=level,
                    evidence=expression_text(metric["node"]),
                )
            )

    return findings


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------


def looks_like_test_data(text: str) -> bool:
    lowered = text.lower()

    test_patterns = {
        "x_test",
        "y_test",
        "test_x",
        "test_y",
        "test_data",
        "test_features",
        "test_labels",
    }

    return any(
        pattern in lowered
        for pattern in test_patterns
    )


def remove_duplicate_findings(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    unique = []
    seen = set()

    for finding in findings:
        key = (
            finding.get("id"),
            finding.get("line_number"),
            finding.get("evidence"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(finding)

    return unique