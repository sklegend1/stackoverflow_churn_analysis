import polars as pl

df = pl.read_csv("persona_comparison_nmf_vs_old.csv")

old_counts = (
    df.group_by("old_persona")
      .agg(pl.len().alias("old_count"))
      .sort("old_count", descending=True)
      .with_row_index("rank")
)

new_counts = (
    df.group_by("new_persona")
      .agg(pl.len().alias("new_count"))
      .sort("new_count", descending=True)
      .with_row_index("rank")
)

# Join by rank
comparison = (
    old_counts
    .join(new_counts, on="rank", how="outer")
    .select([
        "rank",
        "old_persona", "old_count",
        "new_persona", "new_count"
    ])
)

comparison.write_csv("persona_size_comparison.csv")

print(comparison)
print("Saved → persona_size_comparison.csv")
