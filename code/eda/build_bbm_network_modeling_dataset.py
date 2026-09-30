from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("data/generated/bbm_network_eda_v1.csv")
OUTPUT = Path("data/generated/bbm_network_modeling_v1.csv")

PAIR_KEYS = [
    "graph_type",
    "n_nodes",
    "initial_condition_id",
    "gamma",
    "K",
    "T",
    "h",
]

CONTROLLED_SCENARIOS = {
    "single_0",
    "single_middle",
    "two_0_last",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


df = pd.read_csv(SOURCE)

print("=" * 80)
print("BUILDING SUPERVISED BBM NETWORK MODELING DATASET")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Validate source dataset
# ---------------------------------------------------------------------

require(df.shape == (1200, 29), f"Unexpected source shape: {df.shape}")
require(df["experiment_id"].is_unique, "experiment_id is not unique")
require(df.duplicated().sum() == 0, "Duplicated source rows detected")

expected_scenarios = {
    "uncontrolled",
    "single_0",
    "single_middle",
    "two_0_last",
}

require(
    set(df["scenario"].unique()) == expected_scenarios,
    f"Unexpected scenarios: {sorted(df['scenario'].unique())}",
)

scenario_counts = df["scenario"].value_counts().to_dict()

for scenario in expected_scenarios:
    require(
        scenario_counts.get(scenario, 0) == 300,
        f"Expected 300 rows for {scenario}, got {scenario_counts.get(scenario, 0)}",
    )

# ---------------------------------------------------------------------
# 2. Construct uncontrolled baseline table
# ---------------------------------------------------------------------

baseline_columns = PAIR_KEYS + [
    "experiment_id",
    "initial_sync_error",
    "initial_network_l2",
    "initial_max_coefficient",
    "final_sync_error",
]

baseline = (
    df.loc[df["scenario"] == "uncontrolled", baseline_columns]
    .rename(
        columns={
            "experiment_id": "baseline_experiment_id",
            "initial_sync_error": "baseline_initial_sync_error",
            "initial_network_l2": "baseline_initial_network_l2",
            "initial_max_coefficient": "baseline_initial_max_coefficient",
            "final_sync_error": "final_sync_error_uncontrolled",
        }
    )
    .copy()
)

require(len(baseline) == 300, f"Expected 300 baselines, got {len(baseline)}")
require(
    not baseline.duplicated(PAIR_KEYS).any(),
    "More than one uncontrolled baseline exists for a pairing key",
)

# ---------------------------------------------------------------------
# 3. Controlled observations
# ---------------------------------------------------------------------

controlled = df.loc[df["scenario"].isin(CONTROLLED_SCENARIOS)].copy()

require(len(controlled) == 900, f"Expected 900 controlled rows, got {len(controlled)}")
require(
    controlled["mean_actuator_degree"].isna().sum() == 0,
    "Controlled rows contain missing mean_actuator_degree",
)

require(
    np.allclose(
        controlled["squared_control_effort"].to_numpy(),
        0.36,
        rtol=0.0,
        atol=1e-14,
    ),
    "Controlled experiments do not all have squared_control_effort = 0.36",
)

# Every baseline group must contain all three controlled strategies.
group_sizes = controlled.groupby(PAIR_KEYS, dropna=False).size()
group_scenarios = controlled.groupby(PAIR_KEYS, dropna=False)["scenario"].nunique()

require(
    len(group_sizes) == 300,
    f"Expected 300 experimental groups, got {len(group_sizes)}",
)
require(
    (group_sizes == 3).all(),
    "Every baseline group must have exactly three controlled observations",
)
require(
    (group_scenarios == 3).all(),
    "Every baseline group must contain all three controlled scenarios",
)

# ---------------------------------------------------------------------
# 4. Pair each controlled experiment with its uncontrolled baseline
# ---------------------------------------------------------------------

paired = controlled.merge(
    baseline,
    on=PAIR_KEYS,
    how="left",
    validate="many_to_one",
)

require(
    paired["baseline_experiment_id"].notna().all(),
    "At least one controlled experiment has no uncontrolled baseline",
)

# Initial state must be identical between controlled and baseline runs.
for controlled_col, baseline_col in [
    ("initial_sync_error", "baseline_initial_sync_error"),
    ("initial_network_l2", "baseline_initial_network_l2"),
    ("initial_max_coefficient", "baseline_initial_max_coefficient"),
]:
    max_difference = np.max(
        np.abs(
            paired[controlled_col].to_numpy()
            - paired[baseline_col].to_numpy()
        )
    )

    require(
        max_difference <= 1e-14,
        f"Initial-state mismatch in {controlled_col}: {max_difference}",
    )

# ---------------------------------------------------------------------
# 5. Supervised targets
# ---------------------------------------------------------------------

paired["final_sync_error_controlled"] = paired["final_sync_error"]

paired["delta_sync"] = (
    paired["final_sync_error_controlled"]
    - paired["final_sync_error_uncontrolled"]
)

require(
    (paired["final_sync_error_uncontrolled"] > 0).all(),
    "Cannot compute relative delta because a baseline sync error is zero",
)

paired["relative_delta_sync"] = (
    paired["delta_sync"]
    / paired["final_sync_error_uncontrolled"]
)

paired["improved_sync"] = paired["delta_sync"] < 0

# baseline_experiment_id is also our leakage-safe grouping variable.
paired["modeling_group_id"] = paired["baseline_experiment_id"]

# ---------------------------------------------------------------------
# 6. Keep a transparent modeling table
# ---------------------------------------------------------------------

columns = [
    "experiment_id",
    "baseline_experiment_id",
    "modeling_group_id",
    "graph_type",
    "n_nodes",
    "n_edges",
    "density",
    "graph_mean_degree",
    "graph_degree_std",
    "graph_max_degree",
    "initial_condition_id",
    "gamma",
    "K",
    "T",
    "h",
    "scenario",
    "actuator_count",
    "actuator_nodes",
    "mean_actuator_degree",
    "squared_control_effort",
    "initial_sync_error",
    "initial_network_l2",
    "initial_max_coefficient",
    "final_sync_error_controlled",
    "final_sync_error_uncontrolled",
    "delta_sync",
    "relative_delta_sync",
    "improved_sync",
]

modeling = paired[columns].sort_values(
    [
        "graph_type",
        "n_nodes",
        "initial_condition_id",
        "gamma",
        "scenario",
    ]
).reset_index(drop=True)

require(modeling.shape[0] == 900, "Final modeling dataset must have 900 rows")
require(modeling["modeling_group_id"].nunique() == 300, "Expected 300 groups")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
modeling.to_csv(OUTPUT, index=False, float_format="%.17g")

# ---------------------------------------------------------------------
# 7. Audit report
# ---------------------------------------------------------------------

print(f"\nSource: {SOURCE}")
print(f"Output: {OUTPUT}")

print("\nFinal shape:")
print(modeling.shape)

print("\nUnique leakage-safe groups:")
print(modeling["modeling_group_id"].nunique())

print("\nRows per group:")
print(modeling.groupby("modeling_group_id").size().value_counts().sort_index())

print("\nScenario counts:")
print(modeling["scenario"].value_counts().sort_index())

print("\nSquared control effort:")
print(modeling["squared_control_effort"].value_counts().sort_index())

print("\nTarget summary:")
print(
    modeling["delta_sync"]
    .describe()
    .to_string()
)

print("\nTarget by scenario:")
summary = (
    modeling.groupby("scenario")["delta_sync"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)
print(summary.to_string())

print("\nSynchronization improvement counts:")
improvement = (
    modeling.groupby("scenario")["improved_sync"]
    .agg(["sum", "count", "mean"])
)
improvement["percentage"] = 100.0 * improvement["mean"]
print(improvement.to_string())

print("\nModeling features planned for v1:")
print("Categorical: graph_type, initial_condition_id, scenario")
print("Numeric:     n_nodes, gamma")
print("Target:      delta_sync")
print("Group:       modeling_group_id")

print("\nVALIDATION: OK")
