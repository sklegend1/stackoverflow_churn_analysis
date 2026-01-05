import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl
import xgboost as xgb
import pandas as pd
import re

# ======================================================
# 1) Load feature matrix
# ======================================================
fm = pl.read_parquet("../../data/processed/feature_matrix_v2.parquet")

# Convert gap columns from duration to float seconds
gap_cols = [c for c in fm.columns if c.startswith("gap_")]

fm = fm.with_columns([
    pl.col(c).dt.total_seconds().fill_null(0).alias(c)
    for c in gap_cols
])

# ======================================================
# 2) Load ActionTypes.csv → mapping dict
# ======================================================
action_df = pd.read_csv("../../data/raw/ActionTypes.csv")
ACTION_MAP = dict(zip(action_df["ActionTypeId"], action_df["Name"]))

# ======================================================
# 3) Setup feature columns
# ======================================================
drop_cols = ["user_id", "AccountCreationDate", "LastAccessDate"]
feature_cols = [c for c in fm.columns if c not in drop_cols + ["persona"]]

personas = fm.select("persona").unique().to_series().to_list()

results = []

# ======================================================
# 4) Train one-vs-rest model for every persona
# ======================================================
for P in personas:
    print(f"=== Training for persona {P} ===")

    # Binary target definition
    df = fm.with_columns([
        (pl.col("persona") == P).cast(pl.Int8).alias("target")
    ])

    X = df.select(feature_cols).to_pandas()
    y = df.select("target").to_pandas()["target"]

    # Avoid personas with too few samples
    if y.sum() < 20:
        print(f"Skipping persona {P} (too few positive samples)")
        continue

    # Train model
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
    }

    model = xgb.train(params, dtrain, num_boost_round=200)

    importance = model.get_score(importance_type="gain")

    # ==================================================
    # 5) Build importance dataframe WITH action names
    # ==================================================
    rows = []

    for col in feature_cols:
        gain_value = importance.get(col, 0)

        # Extract action_id only if feature name matches "count_action_<id>"
        match = re.match(r"count_action_(\d+)", col)
        if match:
            action_id = int(match.group(1))
            action_name = ACTION_MAP.get(action_id, "Unknown_Action")
        else:
            action_id = None
            action_name = col  # gap stats or other features

        rows.append({
            "persona": P,
            "feature": action_name,
            "action_type_id": action_id,
            # "action_name": action_name,
            "gain": gain_value
        })

    imp_df = pd.DataFrame(rows)

    # Normalize importance inside each persona
    imp_df["gain"] = imp_df["gain"] / (imp_df["gain"].sum() + 1e-12)

    results.append(imp_df)

# ======================================================
# 6) Export final CSV
# ======================================================
final_df = pd.concat(results)

final_df = final_df.sort_values(
    ["persona", "gain"],
    ascending=[True, False]
)

final_df.to_csv("../../data/processed/core_activity_per_persona_named.csv", index=False)

print("Saved → core_activity_per_persona_named.csv")
