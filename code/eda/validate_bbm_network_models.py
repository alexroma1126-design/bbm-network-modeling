from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SOURCE = Path("data/generated/bbm_network_modeling_v1.csv")

FOLD_OUTPUT = Path(
    "data/generated/bbm_network_nested_group_cv_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "data/generated/bbm_network_nested_group_cv_summary_v1.csv"
)

PARAMS_OUTPUT = Path(
    "data/generated/bbm_network_nested_group_cv_params_v1.json"
)

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

RANDOM_STATE = 42


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_one_hot_encoder(*, drop=None):
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


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def linear_pipeline():
    preprocessor = ColumnTransformer(
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def forest_pipeline():
    preprocessor = ColumnTransformer(
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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


# =====================================================================
# Load
# =====================================================================

df = pd.read_csv(SOURCE)

require(len(df) == 900, f"Expected 900 rows, got {len(df)}")
require(
    df[GROUP_COLUMN].nunique() == 300,
    "Expected 300 groups",
)
require(
    (df.groupby(GROUP_COLUMN).size() == 3).all(),
    "Every group must contain exactly three observations",
)

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df[GROUP_COLUMN].copy()

print("=" * 80)
print("BBM NETWORK NESTED GROUP CROSS-VALIDATION")
print("=" * 80)

print("\nRows:", len(df))
print("Groups:", groups.nunique())
print("Features:", FEATURES)
print("Target:", TARGET)


# =====================================================================
# Nested grouped CV
# =====================================================================

outer_cv = GroupKFold(n_splits=5)

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

records = []
parameter_records = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups=groups),
    start=1,
):
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    train_groups = set(groups_train)
    test_groups = set(groups_test)

    require(
        train_groups.isdisjoint(test_groups),
        f"Fold {fold}: group leakage detected",
    )

    require(
        len(train_groups) == 240,
        f"Fold {fold}: expected 240 train groups",
    )

    require(
        len(test_groups) == 60,
        f"Fold {fold}: expected 60 test groups",
    )

    print("\n" + "=" * 80)
    print(f"OUTER FOLD {fold}")
    print("=" * 80)
    print(f"Train rows: {len(train_idx)}")
    print(f"Test rows:  {len(test_idx)}")
    print(f"Train groups: {len(train_groups)}")
    print(f"Test groups:  {len(test_groups)}")

    # -----------------------------------------------------------------
    # Mean baseline
    # -----------------------------------------------------------------

    baseline_pred = np.full(
        len(y_test),
        float(y_train.mean()),
    )

    baseline_metrics = metrics(
        y_test,
        baseline_pred,
    )

    # -----------------------------------------------------------------
    # Linear regression
    # -----------------------------------------------------------------

    linear = linear_pipeline()

    linear.fit(
        X_train,
        y_train,
    )

    linear_pred = linear.predict(X_test)

    linear_metrics = metrics(
        y_test,
        linear_pred,
    )

    # -----------------------------------------------------------------
    # Random Forest: inner grouped CV
    # -----------------------------------------------------------------

    inner_cv = GroupKFold(n_splits=5)

    search = GridSearchCV(
        estimator=forest_pipeline(),
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=0,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    forest = search.best_estimator_

    forest_pred = forest.predict(X_test)

    forest_metrics = metrics(
        y_test,
        forest_pred,
    )

    inner_rmse = float(
        np.sqrt(-search.best_score_)
    )

    print("\nMean baseline")
    print(baseline_metrics)

    print("\nLinear Regression")
    print(linear_metrics)

    print("\nRandom Forest")
    print(forest_metrics)

    print("\nBest RF parameters")
    print(search.best_params_)

    print(
        f"Inner grouped-CV RMSE: "
        f"{inner_rmse:.10f}"
    )

    parameter_records.append(
        {
            "fold": fold,
            "best_params": search.best_params_,
            "inner_cv_rmse": inner_rmse,
        }
    )

    for model_name, model_metrics in [
        ("mean_baseline", baseline_metrics),
        ("linear_regression", linear_metrics),
        ("random_forest", forest_metrics),
    ]:
        records.append(
            {
                "fold": fold,
                "model": model_name,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                **model_metrics,
            }
        )


# =====================================================================
# Aggregate
# =====================================================================

fold_table = pd.DataFrame(records)

summary = (
    fold_table
    .groupby("model")[["MAE", "RMSE", "R2"]]
    .agg(["mean", "std", "min", "max"])
)

print("\n" + "=" * 80)
print("NESTED GROUP-CV SUMMARY")
print("=" * 80)

print(summary.to_string())


# =====================================================================
# RF hyperparameter stability
# =====================================================================

parameter_strings = [
    json.dumps(
        record["best_params"],
        sort_keys=True,
    )
    for record in parameter_records
]

counts = Counter(parameter_strings)

print("\nRandom Forest selected-parameter frequency")
print("------------------------------------------")

for parameters, count in counts.most_common():
    print(f"{count} fold(s): {parameters}")


# =====================================================================
# Save
# =====================================================================

FOLD_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fold_table.to_csv(
    FOLD_OUTPUT,
    index=False,
    float_format="%.17g",
)

summary_flat = summary.copy()
summary_flat.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary_flat.columns
]

summary_flat.reset_index().to_csv(
    SUMMARY_OUTPUT,
    index=False,
    float_format="%.17g",
)

with PARAMS_OUTPUT.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        parameter_records,
        handle,
        indent=2,
        sort_keys=True,
    )

print("\nSaved:")
print(FOLD_OUTPUT)
print(SUMMARY_OUTPUT)
print(PARAMS_OUTPUT)


# =====================================================================
# Final checks
# =====================================================================

require(
    len(fold_table) == 15,
    "Expected 5 folds x 3 models = 15 result rows",
)

require(
    np.isfinite(
        fold_table[["MAE", "RMSE", "R2"]].to_numpy()
    ).all(),
    "Non-finite validation metric detected",
)

print("\nVALIDATION: OK")
