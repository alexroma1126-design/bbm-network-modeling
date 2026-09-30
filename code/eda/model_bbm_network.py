from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SOURCE = Path("data/generated/bbm_network_modeling_v1.csv")

METRICS_OUTPUT = Path(
    "data/generated/bbm_network_model_metrics_v1.csv"
)
PREDICTIONS_OUTPUT = Path(
    "data/generated/bbm_network_model_predictions_v1.csv"
)
IMPORTANCE_OUTPUT = Path(
    "data/generated/bbm_network_rf_permutation_importance_v1.csv"
)
PARAMS_OUTPUT = Path(
    "data/generated/bbm_network_rf_best_params_v1.json"
)

RANDOM_STATE = 42

TARGET = "delta_sync"
GROUP_COLUMN = "modeling_group_id"

CATEGORICAL_FEATURES = [
    "graph_type",
    "initial_condition_id",
    "scenario",
]

NUMERIC_FEATURES = [
    "n_nodes",
    "gamma",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_one_hot_encoder(*, drop=None):
    """
    Compatibility with both newer and older scikit-learn versions.
    """
    kwargs = {
        "handle_unknown": "ignore",
        "drop": drop,
    }

    try:
        return OneHotEncoder(
            sparse_output=False,
            **kwargs,
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            **kwargs,
        )


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))

    for metric, value in metrics.items():
        print(f"{metric:>5}: {value:.10f}")


# =====================================================================
# 1. Load and validate
# =====================================================================

df = pd.read_csv(SOURCE)

print("=" * 80)
print("BBM NETWORK SUPERVISED MODELING")
print("=" * 80)

require(
    len(df) == 900,
    f"Expected 900 observations, got {len(df)}",
)

require(
    df[GROUP_COLUMN].nunique() == 300,
    "Expected exactly 300 leakage-safe groups",
)

require(
    df[TARGET].notna().all(),
    "Target contains missing values",
)

require(
    df[FEATURES].notna().all().all(),
    "Selected modeling features contain missing values",
)

group_sizes = df.groupby(GROUP_COLUMN).size()

require(
    (group_sizes == 3).all(),
    "Each group must contain exactly three actuator strategies",
)

scenario_count_per_group = (
    df.groupby(GROUP_COLUMN)["scenario"]
    .nunique()
)

require(
    (scenario_count_per_group == 3).all(),
    "Each group must contain all three actuator strategies",
)

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df[GROUP_COLUMN].copy()


# =====================================================================
# 2. Leakage-safe train/test split
# =====================================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

train_group_set = set(groups_train)
test_group_set = set(groups_test)

require(
    train_group_set.isdisjoint(test_group_set),
    "LEAKAGE: train and test share experimental groups",
)

require(
    len(train_group_set) == 240,
    f"Expected 240 train groups, got {len(train_group_set)}",
)

require(
    len(test_group_set) == 60,
    f"Expected 60 test groups, got {len(test_group_set)}",
)

require(
    len(X_train) == 720,
    f"Expected 720 train rows, got {len(X_train)}",
)

require(
    len(X_test) == 180,
    f"Expected 180 test rows, got {len(X_test)}",
)

print("\nLeakage-safe split")
print("------------------")
print(f"Train rows:   {len(X_train)}")
print(f"Test rows:    {len(X_test)}")
print(f"Train groups: {len(train_group_set)}")
print(f"Test groups:  {len(test_group_set)}")
print("Shared groups: 0")


# =====================================================================
# 3. Check design-factor coverage
# =====================================================================

print("\nFactor coverage")
print("---------------")

for feature in FEATURES:
    train_values = sorted(X_train[feature].unique())
    test_values = sorted(X_test[feature].unique())

    print(f"\n{feature}")
    print(f"  train: {train_values}")
    print(f"  test:  {test_values}")


# =====================================================================
# 4. Mean predictor baseline
# =====================================================================

mean_prediction = np.full(
    shape=len(y_test),
    fill_value=float(y_train.mean()),
)

mean_metrics = regression_metrics(
    y_test,
    mean_prediction,
)

print_metrics(
    "Mean predictor baseline",
    mean_metrics,
)


# =====================================================================
# 5. Linear Regression
# =====================================================================

linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            make_one_hot_encoder(drop="first"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES,
        ),
    ],
    remainder="drop",
)

linear_model = Pipeline(
    steps=[
        ("preprocessor", linear_preprocessor),
        ("model", LinearRegression()),
    ]
)

linear_model.fit(
    X_train,
    y_train,
)

linear_prediction = linear_model.predict(X_test)

linear_metrics = regression_metrics(
    y_test,
    linear_prediction,
)

print_metrics(
    "Linear Regression",
    linear_metrics,
)


# =====================================================================
# 6. Random Forest with group-aware tuning
# =====================================================================

forest_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            make_one_hot_encoder(drop=None),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
    ],
    remainder="drop",
)

forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", forest_preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ]
)

parameter_grid = {
    "model__max_depth": [
        None,
        6,
        12,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4,
    ],
    "model__max_features": [
        "sqrt",
        0.7,
    ],
}

group_cv = GroupKFold(n_splits=5)

grid_search = GridSearchCV(
    estimator=forest_pipeline,
    param_grid=parameter_grid,
    scoring="neg_mean_squared_error",
    cv=group_cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
    verbose=1,
)

print("\nRandom Forest group-aware grid search")
print("-------------------------------------")
print(
    f"Candidates: "
    f"{len(parameter_grid['model__max_depth']) * len(parameter_grid['model__min_samples_leaf']) * len(parameter_grid['model__max_features'])}"
)
print("Folds: 5")

grid_search.fit(
    X_train,
    y_train,
    groups=groups_train,
)

forest_model = grid_search.best_estimator_

forest_prediction = forest_model.predict(X_test)

forest_metrics = regression_metrics(
    y_test,
    forest_prediction,
)

print("\nBest Random Forest parameters")
print("-----------------------------")
print(
    json.dumps(
        grid_search.best_params_,
        indent=2,
        sort_keys=True,
    )
)

cv_rmse = float(
    np.sqrt(-grid_search.best_score_)
)

print(
    f"\nBest grouped-CV RMSE: "
    f"{cv_rmse:.10f}"
)

print_metrics(
    "Random Forest",
    forest_metrics,
)


# =====================================================================
# 7. Held-out permutation importance
# =====================================================================

importance = permutation_importance(
    estimator=forest_model,
    X=X_test,
    y=y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

importance_table = pd.DataFrame(
    {
        "feature": FEATURES,
        "mae_increase_mean": importance.importances_mean,
        "mae_increase_std": importance.importances_std,
    }
).sort_values(
    "mae_increase_mean",
    ascending=False,
).reset_index(drop=True)

print("\nRandom Forest permutation importance")
print("------------------------------------")
print(importance_table.to_string(index=False))


# =====================================================================
# 8. Save reproducible outputs
# =====================================================================

metrics_table = pd.DataFrame(
    [
        {
            "model": "mean_baseline",
            **mean_metrics,
        },
        {
            "model": "linear_regression",
            **linear_metrics,
        },
        {
            "model": "random_forest",
            **forest_metrics,
        },
    ]
)

prediction_table = df.iloc[test_idx][
    [
        "experiment_id",
        "baseline_experiment_id",
        GROUP_COLUMN,
        "graph_type",
        "n_nodes",
        "initial_condition_id",
        "gamma",
        "scenario",
        TARGET,
    ]
].copy()

prediction_table["prediction_mean_baseline"] = mean_prediction
prediction_table["prediction_linear_regression"] = linear_prediction
prediction_table["prediction_random_forest"] = forest_prediction

METRICS_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

metrics_table.to_csv(
    METRICS_OUTPUT,
    index=False,
    float_format="%.17g",
)

prediction_table.to_csv(
    PREDICTIONS_OUTPUT,
    index=False,
    float_format="%.17g",
)

importance_table.to_csv(
    IMPORTANCE_OUTPUT,
    index=False,
    float_format="%.17g",
)

with PARAMS_OUTPUT.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        {
            "random_state": RANDOM_STATE,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_groups": len(train_group_set),
            "test_groups": len(test_group_set),
            "best_parameters": grid_search.best_params_,
            "grouped_cv_rmse": cv_rmse,
        },
        handle,
        indent=2,
        sort_keys=True,
    )

print("\nSaved outputs")
print("-------------")
print(METRICS_OUTPUT)
print(PREDICTIONS_OUTPUT)
print(IMPORTANCE_OUTPUT)
print(PARAMS_OUTPUT)


# =====================================================================
# 9. Final checks
# =====================================================================

require(
    np.isfinite(linear_prediction).all(),
    "Linear Regression produced non-finite predictions",
)

require(
    np.isfinite(forest_prediction).all(),
    "Random Forest produced non-finite predictions",
)

require(
    prediction_table[GROUP_COLUMN].nunique() == 60,
    "Prediction table must contain exactly 60 test groups",
)

print("\nVALIDATION: OK")
