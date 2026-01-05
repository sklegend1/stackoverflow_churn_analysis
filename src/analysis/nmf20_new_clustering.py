import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

fm = pl.read_parquet("../../data/processed/feature_matrix_v2.parquet")

# Convert gap durations to seconds
gap_cols = [c for c in fm.columns if c.startswith("gap_")]
fm = fm.with_columns([
    pl.col(c).dt.total_seconds().fill_null(0).alias(c)
    for c in gap_cols
])

drop_cols = ["user_id", "persona", "AccountCreationDate", "LastAccessDate"]

# numeric only
feature_cols = [
    c for c, dt in zip(fm.columns, fm.dtypes)
    if c not in drop_cols and dt.is_numeric()
]

# Replace negative values with zero
fm = fm.with_columns([
    pl.when(pl.col(c) < 0).then(0).otherwise(pl.col(c)).alias(c)
    for c in feature_cols
])

X = fm.select(feature_cols).to_pandas().fillna(0.0)
old_labels = fm.select("persona").to_pandas()["persona"]

# NMF
nmf_model = NMF(
    n_components=20,
    init="nndsvda",
    max_iter=500,
    random_state=42
)

W = nmf_model.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=20, n_init=20, random_state=42)
new_labels = kmeans.fit_predict(W)

# Save comparison
out = pd.DataFrame({
    "user_id": fm.select("user_id").to_pandas()["user_id"],
    "old_persona": old_labels,
    "new_persona": new_labels
})
out.to_csv("../../results/persona_comparison_nmf_vs_old.csv", index=False)

# Similarity scores
ari = adjusted_rand_score(old_labels, new_labels)
nmi = normalized_mutual_info_score(old_labels, new_labels)

print("ARI:", ari)
print("NMI:", nmi)
