import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import polars as pl

# Mapping PostHistoryTypeId → ActionTypeId
PH_EVENT_MAP = {
    4: 10, 5: 11, 6: 12,
    10: 20, 11: 21, 12: 22, 13: 23,
    14: 24, 15: 25,
    19: 26, 20: 27,
    24: 28,
    66: 29
}

COMMENT_ACTION = 40

start = time.time()

# ---------------------------------------------
# LOAD filtered_users AND NORMALIZE COLUMN NAME
# ---------------------------------------------
users = pl.read_parquet("filtered_users_parquet/filtered_users_part_00000.parquet")

users = users.select([pl.col("Id"),pl.col("DisplayName"), pl.col("LastAccessDate"),pl.col("ClusterNmf20")])
# Rename any possible ID column to "user_id"
if "UserId" in users.columns:
    users = users.rename({"UserId": "user_id"})
elif "Id" in users.columns:
    users = users.rename({"Id": "user_id"})
elif "user_id" not in users.columns:
    raise ValueError("filtered_users must contain user_id/id/userid column")

print("Loaded filtered_users with", users.height, "rows")


# ---------------------------------------------
# LOAD POSTS (lazy)
# ---------------------------------------------
posts = pl.scan_parquet("filtered_posts_parquet/*.parquet").select([
    pl.col("owneruserid").alias("user_id"),
    pl.col("creationdate").alias("timestamp"),
    pl.col("id").alias("post_id"),
    pl.col("posttypeid").alias("post_type_id"),
])

events_create_post = posts.select([
    "user_id",
    "timestamp",
    pl.lit(1).alias("action_type_id"),
    "post_id",
    "post_type_id"
])


# ---------------------------------------------
# POSTHISTORY (lazy)
# ---------------------------------------------
ph = pl.scan_parquet("posthistory_parquet/*.parquet")

ph = ph.filter(pl.col("posthistorytypeid").is_in(list(PH_EVENT_MAP.keys())))

mapped_actions = (
    pl.when(pl.col("posthistorytypeid") == 4).then(10)
    .when(pl.col("posthistorytypeid") == 5).then(11)
    .when(pl.col("posthistorytypeid") == 6).then(12)
    .when(pl.col("posthistorytypeid") == 10).then(20)
    .when(pl.col("posthistorytypeid") == 11).then(21)
    .when(pl.col("posthistorytypeid") == 12).then(22)
    .when(pl.col("posthistorytypeid") == 13).then(23)
    .when(pl.col("posthistorytypeid") == 14).then(24)
    .when(pl.col("posthistorytypeid") == 15).then(25)
    .when(pl.col("posthistorytypeid") == 19).then(26)
    .when(pl.col("posthistorytypeid") == 20).then(27)
    .when(pl.col("posthistorytypeid") == 24).then(28)
    .when(pl.col("posthistorytypeid") == 66).then(29)
    .otherwise(None)
)

events_ph = (
    ph
    .with_columns([
        pl.col("userid").alias("user_id"),
        pl.col("creationdate").alias("timestamp"),
        mapped_actions.alias("action_type_id"),
        pl.col("postid").alias("post_id"),
        pl.lit(None, dtype=pl.Int32).alias("post_type_id")
    ])
    .select(["user_id", "timestamp", "action_type_id", "post_id", "post_type_id"])
)

# JOIN post_type_id from posts
events_ph = events_ph.join(
    posts.select(["post_id", "post_type_id"]),
    on="post_id",
    how="left"
).with_columns([
    pl.coalesce(["post_type_id", "post_type_id_right"]).alias("post_type_id")
]).drop("post_type_id_right")


# ---------------------------------------------
# COMMENTS (lazy)
# ---------------------------------------------
comments = pl.scan_parquet("comments_parquet/*.parquet")

events_comments = comments.select([
    pl.col("userid").alias("user_id"),
    pl.col("creationdate").alias("timestamp"),
    pl.lit(COMMENT_ACTION).alias("action_type_id"),
    pl.col("postid").alias("post_id"),
    pl.lit(None, dtype=pl.Int32).alias("post_type_id"),
])

events_comments = events_comments.join(
    posts.select(["post_id", "post_type_id"]),
    on="post_id",
    how="left"
).with_columns([
    pl.coalesce(["post_type_id", "post_type_id_right"]).alias("post_type_id")
]).drop("post_type_id_right")


# ---------------------------------------------
# FILTER ALL EVENTS TO ONLY filtered_users
# ---------------------------------------------
events_create_post = events_create_post.join(users.lazy(), on="user_id", how="inner")
events_ph = events_ph.join(users.lazy(), on="user_id", how="inner")
events_comments = events_comments.join(users.lazy(), on="user_id", how="inner")


# ---------------------------------------------
# CONCAT ALL (same schema now)
# ---------------------------------------------
eventlog_lazy = pl.concat([events_create_post, events_ph, events_comments])

eventlog_lazy = eventlog_lazy.sort(["user_id", "timestamp"])

eventlog_lazy = (
    eventlog_lazy.with_columns([
        (pl.col("timestamp")
            .diff()
            .over("user_id")/1000000
            
        ).alias("gap_time_actions")
    ])
)

eventlog = eventlog_lazy.collect(engine="streaming")

print("Created eventlog_v1 with", eventlog.height, "rows.")
eventlog.write_parquet("../../data/processed/eventlog_v1.parquet")

print(f"Completed in {time.time()-start:.2f} seconds")
