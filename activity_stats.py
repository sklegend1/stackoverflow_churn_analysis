import polars as pl

# df = pl.read_parquet("eventlog_v1.parquet")

# stats = df.group_by("ClusterNmf20", "action_type_id").len()

# stats.write_csv("activity_distribution_by_cluster_actiontype.csv")

df = pl.read_parquet("filtered_users_parquet/filtered_users_part_00000.parquet")

stats = df.group_by("ClusterNmf20").len()
print(stats)
stats.write_csv("user_distribution_by_cluster.csv")