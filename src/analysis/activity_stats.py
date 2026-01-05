import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import polars as pl

# df = pl.read_parquet("../../data/processed/eventlog_v1.parquet")

# stats = df.group_by("ClusterNmf20", "action_type_id").len()

# stats.write_csv("../../data/processed/activity_distribution_by_cluster_actiontype.csv")

df = pl.read_parquet("filtered_users_parquet/filtered_users_part_00000.parquet")

stats = df.group_by("ClusterNmf20").len()
print(stats)
stats.write_csv("../../data/processed/user_distribution_by_cluster.csv")