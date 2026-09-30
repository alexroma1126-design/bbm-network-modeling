from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SOURCE = Path("data/generated/bbm_network_modeling_v1.csv")

RESULTS_OUTPUT = Path(
    "data/generated/bbm_network_repeated_group_holdout_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "data/generated/bbm_network_repeated_group_holdout_summary_v1.csv"
)

PARAMS_OUTPUT = Path(
    "data/generated/bbm_network_repeated_group_holdout_params_v1.json"
)


TARGET = "delta_sync"
GROUP = "modeling_group_id"

CATEGORICAL = [
    "graph_type",
    "initial_condition_id",
    "scenario",
]

NUMERIC = [
    "n_nodes",
    "gamma",
]

FEATURES = CATEGORICAL + NUMERIC

SEEDS = [
    11,
    22,
    33,
    42,
    55,
    66,
    77,
    88,
    99,
    123,
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encoder(drop=None):
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

    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                encoder(drop="first"),
                CATEGORICAL,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessing),
            ("model", LinearRegression()),
        ]
    )


def forest_pipeline(seed):

    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                encoder(drop=None),
                CATEGORICAL,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


# ==============================================================
# Load
# ==============================================================

df = pd.read_csv(SOURCE)

require(len(df) == 900, "Expected 900 rows")
require(df[GROUP].nunique() == 300, "Expected 300 groups")

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

expected_levels = {
    col: set(df[col].unique())
    for col in FEATURES
}


# ==============================================================
# Model grid
# ==============================================================

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


print("=" * 80)
print("BBM NETWORK REPEATED GROUPED HOLDOUT")
print("=" * 80)


# ==============================================================
# Repeated group-safe evaluation
# ==============================================================

for repetition, seed in enumerate(SEEDS, start=1):

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_idx, test_idx = next(
        outer_split.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    require(
        set(groups_train).isdisjoint(set(groups_test)),
        "Group leakage detected",
    )

    require(
        groups_train.nunique() == 240,
        "Expected 240 train groups",
    )

    require(
        groups_test.nunique() == 60,
        "Expected 60 test groups",
    )

    # ----------------------------------------------------------
    # Factor coverage
    # ----------------------------------------------------------

    for feature in FEATURES:

        train_levels = set(X_train[feature].unique())
        test_levels = set(X_test[feature].unique())

        require(
            train_levels == expected_levels[feature],
            (
                f"Seed {seed}: training missing levels "
                f"for {feature}"
            ),
        )

        require(
            test_levels == expected_levels[feature],
            (
                f"Seed {seed}: test missing levels "
                f"for {feature}"
            ),
        )

    # ----------------------------------------------------------
    # Mean baseline
    # ----------------------------------------------------------

    baseline_pred = np.full(
        len(y_test),
        float(y_train.mean()),
    )

    baseline_metrics = metrics(
        y_test,
        baseline_pred,
    )

    # ----------------------------------------------------------
    # Linear Regression
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Inner group-safe tuning
    # ----------------------------------------------------------

    inner_cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.20,
        random_state=seed + 1000,
    )

    forest = forest_pipeline(seed)

    search = GridSearchCV(
        estimator=forest,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    forest_pred = search.best_estimator_.predict(X_test)

    forest_metrics = metrics(
        y_test,
        forest_pred,
    )

    inner_rmse = float(
        np.sqrt(-search.best_score_)
    )

    print("\n" + "-" * 80)
    print(
        f"Repetition {repetition:02d} | seed={seed}"
    )

    print(
        f"Baseline: "
        f"MAE={baseline_metrics['MAE']:.6f} "
        f"RMSE={baseline_metrics['RMSE']:.6f} "
        f"R2={baseline_metrics['R2']:.6f}"
    )

    print(
        f"Linear:   "
        f"MAE={linear_metrics['MAE']:.6f} "
        f"RMSE={linear_metrics['RMSE']:.6f} "
        f"R2={linear_metrics['R2']:.6f}"
    )

    print(
        f"RF:       "
        f"MAE={forest_metrics['MAE']:.6f} "
        f"RMSE={forest_metrics['RMSE']:.6f} "
        f"R2={forest_metrics['R2']:.6f}"
    )

    print(
        f"Inner CV RMSE={inner_rmse:.6f}"
    )

    print(
        f"RF params={search.best_params_}"
    )

    for name, result in [
        ("mean_baseline", baseline_metrics),
        ("linear_regression", linear_metrics),
        ("random_forest", forest_metrics),
    ]:

        records.append(
            {
                "repetition": repetition,
                "seed": seed,
                "model": name,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": groups_train.nunique(),
                "test_groups": groups_test.nunique(),
                **result,
            }
        )

    parameter_records.append(
        {
            "repetition": repetition,
            "seed": seed,
            "inner_cv_rmse": inner_rmse,
            "best_params": search.best_params_,
        }
    )


# ==============================================================
# Summary
# ==============================================================

results = pd.DataFrame(records)

summary = (
    results
    .groupby("model")[["MAE", "RMSE", "R2"]]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    )
)


print("\n" + "=" * 80)
print("REPEATED GROUPED HOLDOUT SUMMARY")
print("=" * 80)

print(summary.to_string())


# ==============================================================
# Win counts
# ==============================================================

pivot_rmse = results.pivot(
    index="repetition",
    columns="model",
    values="RMSE",
)

rf_better_linear = int(
    (
        pivot_rmse["random_forest"]
        <
        pivot_rmse["linear_regression"]
    ).sum()
)

rf_better_baseline = int(
    (
        pivot_rmse["random_forest"]
        <
        pivot_rmse["mean_baseline"]
    ).sum()
)

print("\nRandom Forest RMSE wins")
print("-----------------------")

print(
    f"vs Linear Regression: "
    f"{rf_better_linear}/{len(SEEDS)}"
)

print(
    f"vs Mean baseline:      "
    f"{rf_better_baseline}/{len(SEEDS)}"
)


# ==============================================================
# Save
# ==============================================================

RESULTS_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    RESULTS_OUTPUT,
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
) as f:

    json.dump(
        parameter_records,
        f,
        indent=2,
        sort_keys=True,
    )


print("\nSaved:")
print(RESULTS_OUTPUT)
print(SUMMARY_OUTPUT)
print(PARAMS_OUTPUT)

print("\nVALIDATION: OK")
