from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIG_DIR = Path("figures/modeling")
FIG_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = Path(
    "data/generated/bbm_network_model_metrics_v1.csv"
)

predictions_path = Path(
    "data/generated/bbm_network_model_predictions_v1.csv"
)

importance_path = Path(
    "data/generated/bbm_network_rf_permutation_importance_v1.csv"
)

repeated_path = Path(
    "data/generated/bbm_network_repeated_group_holdout_v1.csv"
)

nested_path = Path(
    "data/generated/bbm_network_nested_group_cv_v1.csv"
)


# ============================================================
# Load
# ============================================================

metrics = pd.read_csv(metrics_path)
predictions = pd.read_csv(predictions_path)
importance = pd.read_csv(importance_path)
repeated = pd.read_csv(repeated_path)
nested = pd.read_csv(nested_path)


# ============================================================
# 1. Model performance
# ============================================================

summary = (
    repeated
    .groupby("model")
    [["MAE", "RMSE", "R2"]]
    .agg(["mean", "std"])
)

model_order = [
    "mean_baseline",
    "linear_regression",
    "random_forest",
]

labels = [
    "Mean baseline",
    "Linear regression",
    "Random forest",
]

rmse_mean = [
    summary.loc[m, ("RMSE", "mean")]
    for m in model_order
]

rmse_std = [
    summary.loc[m, ("RMSE", "std")]
    for m in model_order
]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(labels))

ax.bar(
    x,
    rmse_mean,
    yerr=rmse_std,
    capsize=5,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("RMSE")
ax.set_title(
    "BBM network model performance\n"
    "Repeated leakage-safe grouped holdout"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "01_model_performance.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 2. RF predicted vs actual
# ============================================================

actual = predictions["delta_sync"].to_numpy()
predicted = predictions[
    "prediction_random_forest"
].to_numpy()

low = min(actual.min(), predicted.min())
high = max(actual.max(), predicted.max())

padding = 0.05 * (high - low)

low -= padding
high += padding

fig, ax = plt.subplots(figsize=(6.5, 6.5))

ax.scatter(
    actual,
    predicted,
    alpha=0.75,
)

ax.plot(
    [low, high],
    [low, high],
    linestyle="--",
    linewidth=1.5,
)

ax.axhline(
    0.0,
    linewidth=0.8,
    alpha=0.5,
)

ax.axvline(
    0.0,
    linewidth=0.8,
    alpha=0.5,
)

ax.set_xlim(low, high)
ax.set_ylim(low, high)

ax.set_xlabel(
    r"Observed $\Delta E_{\mathrm{sync}}$"
)

ax.set_ylabel(
    r"Predicted $\Delta E_{\mathrm{sync}}$"
)

ax.set_title(
    "Random forest: predicted vs observed\n"
    "Held-out experimental groups"
)

ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "02_rf_predicted_vs_actual.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. Permutation importance
# ============================================================

importance = importance.sort_values(
    "mae_increase_mean",
    ascending=True,
)

display_names = {
    "scenario": "Actuator strategy",
    "initial_condition_id": "Initial condition",
    "n_nodes": "Network size",
    "gamma": r"Coupling $\gamma$",
    "graph_type": "Graph topology",
}

labels_importance = [
    display_names.get(v, v)
    for v in importance["feature"]
]

fig, ax = plt.subplots(figsize=(8, 5))

positions = np.arange(len(importance))

ax.barh(
    positions,
    importance["mae_increase_mean"],
    xerr=importance["mae_increase_std"],
    capsize=4,
)

ax.set_yticks(positions)
ax.set_yticklabels(labels_importance)

ax.set_xlabel(
    "Increase in MAE after permutation"
)

ax.set_title(
    "Random forest permutation importance\n"
    "Held-out test groups"
)

ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03_rf_permutation_importance.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 4. Validation regimes
# ============================================================

interpolation = (
    repeated.loc[
        repeated["model"] == "random_forest",
        "R2",
    ]
    .reset_index(drop=True)
)

extrapolation = (
    nested.loc[
        nested["model"] == "random_forest",
        "R2",
    ]
    .reset_index(drop=True)
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5),
)

# Known factor levels
axes[0].boxplot(
    interpolation,
    showmeans=True,
)

axes[0].scatter(
    [1] * len(interpolation),
    interpolation,
    alpha=0.65,
)

axes[0].set_xticks([1])
axes[0].set_xticklabels(
    ["Repeated grouped\nholdout"]
)

axes[0].set_ylabel(r"$R^2$")
axes[0].set_title("Known factor levels")

axes[0].grid(
    axis="y",
    alpha=0.25,
)

# Unseen initial condition
axes[1].boxplot(
    extrapolation,
    showmeans=True,
)

axes[1].scatter(
    [1] * len(extrapolation),
    extrapolation,
    alpha=0.65,
)

axes[1].axhline(
    0.0,
    linestyle="--",
    linewidth=1,
)

axes[1].set_xticks([1])
axes[1].set_xticklabels(
    ["Leave-one-condition-\nout style"]
)

axes[1].set_title(
    "Unseen initial condition"
)

axes[1].grid(
    axis="y",
    alpha=0.25,
)

fig.suptitle(
    "Random forest generalization depends on the prediction task"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "04_validation_regimes_v2.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

outputs = sorted(FIG_DIR.glob("*.png"))

print("=" * 72)
print("BBM NETWORK MODELING FIGURES")
print("=" * 72)

for path in outputs:
    print(
        f"{path.name:<35} "
        f"{path.stat().st_size:>10} bytes"
    )

if len(outputs) != 4:
    raise RuntimeError(
        f"Expected 4 figures, found {len(outputs)}"
    )

print("\nVALIDATION: OK")
