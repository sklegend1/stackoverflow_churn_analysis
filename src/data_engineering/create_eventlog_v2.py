import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import polars as pl

# ---------- Action type mappings ----------

PH_ACTION_MAP = {
    # 1: 10,   # Initial Title
    # 2: 11,   # Initial Body
    # 3: 12,   # Initial Tags
    4: 13,   # Edit Title
    5: 14,   # Edit Body
    6: 15,   # Edit Tags
    7: 16,   # Rollback Title
    8: 17,   # Rollback Body
    9: 18,   # Rollback Tags
    10: 30,  # Close Post
    11: 31,  # Reopen Post
    12: 32,  # Delete
    13: 33,  # Undelete
    14: 34,  # Lock
    15: 35,  # Unlock
    16: 40,  # Community Owned
    17: 41,  # Migrated (legacy)
    18: 42,  # Question Merged
    19: 43,  # Question Protected
    20: 44,  # Question Unprotected
    21: 45,  # Post Disassociated
    22: 46,  # Question Unmerged
    24: 47,  # Suggested Edit Applied
    25: 48,  # Post Tweeted
    31: 49,  # Comment discussion moved to chat
    33: 50,  # Post notice added
    34: 51,  # Post notice removed
    35: 52,  # Post migrated away
    36: 53,  # Post migrated here
    37: 54,  # Post merge source
    38: 55,  # Post merge destination
    50: 56,  # Bumped by Community User
    52: 57,  # Question became hot network question
    53: 58,  # Question removed from hot network
    66: 59,  # Created from Ask Wizard
}

CREATE_QUESTION_ACTION = 1
CREATE_ANSWER_ACTION = 2
COMMENT_ACTION = 70

start = time.time()

# ---------- Load base tables (lazy) ----------

# filtered posts: only posts belonging to the 92k users
posts = pl.scan_parquet("../../data/processed/filtered_posts_parquet/*.parquet")

# filtered users with persona + last access
users = (
    pl.scan_parquet("../../data/processed/filtered_users_parquet/*.parquet")
    .select([
        pl.col("Id").alias("user_id"),
        pl.col("ClusterNmf20").alias("persona"),
        pl.col("CreationDate").alias("AccountCreationDate"),
        pl.col("LastAccessDate"),
    ])
)

# full posthistory (we will filter by user later)
ph = pl.scan_parquet("../../data/processed/posthistory_parquet/*.parquet")

# full comments (we will filter by user later)
comments = pl.scan_parquet("../../data/processed/comments_parquet/*.parquet")

# ---------- Events: Create Post (from posts) ----------

events_create_post = (
    posts
    .join(users, left_on="owneruserid", right_on="user_id", how="inner")
    .with_columns([
        pl.col("owneruserid").alias("user_id"),
        pl.col("creationdate").alias("timestamp"),
        pl.col("id").alias("post_id"),
        pl.col("posttypeid").alias("post_type_id"),
        # map posttypeid 1=question, 2=answer
        pl.when(pl.col("posttypeid") == 1)
          .then(pl.lit(CREATE_QUESTION_ACTION))
          .when(pl.col("posttypeid") == 2)
          .then(pl.lit(CREATE_ANSWER_ACTION))
          .otherwise(pl.lit(None))
          .alias("action_type_id"),
    ])
    .select([
        "user_id",
        "timestamp",
        "action_type_id",
        "post_id",
        "post_type_id",
        "persona",
        "LastAccessDate",
        "AccountCreationDate",
    ])
)

# ---------- Events: PostHistory (all types, actor-based) ----------
PH_VALID = list(PH_ACTION_MAP.keys())
# Keep only posthistory rows where the actor is one of our filtered users
ph_user = (
    ph
    .join(users, left_on="userid", right_on="user_id", how="inner")
    .filter(pl.col("posthistorytypeid").is_in(PH_VALID))
)

# Map posthistorytypeid to action_type_id
ph_mapped = ph_user.with_columns([
    pl.col("userid").alias("user_id"),
    pl.col("creationdate").alias("timestamp"),
    pl.col("postid").alias("post_id"),
    pl.col("posthistorytypeid")
    .replace(PH_ACTION_MAP)
    .cast(pl.Int32)
    .alias("action_type_id"),
])

# Join with posts to get post_type_id (question/answer/etc.)
events_ph = (
    ph_mapped
    .join(
        posts.select([
            pl.col("id").alias("post_id"),
            pl.col("posttypeid").alias("post_type_id"),
        ]),
        on="post_id",
        how="left",
    )
    .select([
        "user_id",
        "timestamp",
        "action_type_id",
        "post_id",
        "post_type_id",
        "persona",
        "LastAccessDate",
        "AccountCreationDate",
    ])
)

# ---------- Events: Comments ----------

# Comments written by our filtered users
comments_user = (
    comments
    .join(users, left_on="userid", right_on="user_id", how="inner")
)

events_comments = (
    comments_user
    .join(
        posts.select([
            pl.col("id").alias("post_id"),
            pl.col("posttypeid").alias("post_type_id"),
        ]),
        left_on="postid",
        right_on="post_id",
        how="left",
    )
    .with_columns([
        pl.col("userid").alias("user_id"),
        pl.col("creationdate").alias("timestamp"),
        pl.lit(COMMENT_ACTION).alias("action_type_id"),
        pl.col("postid").alias("post_id"),  # ← این خط مهم است
    ])
    .select([
        "user_id",
        "timestamp",
        "action_type_id",
        "post_id",
        "post_type_id",
        "persona",
        "LastAccessDate",
        "AccountCreationDate",
    ])
)

# ---------- Build unified eventlog ----------

eventlog_lazy = pl.concat(
    [events_create_post, events_ph, events_comments],
    how="diagonal_relaxed"
)

# Sort by user and time
eventlog_lazy = eventlog_lazy.sort(["user_id", "timestamp"])

# Compute gap_time_actions (in seconds) per user
eventlog_lazy = eventlog_lazy.with_columns([
    (pl.col("timestamp")
      .diff()
      .over("user_id")/(1000000*60*60*24))
      .alias("gap_time_actions_days"),
      (pl.col("timestamp")
      .diff()
      .over("user_id")/(1000000))
      .alias("gap_time_actions_seconds"),
])

# Add primary key id (row index)
eventlog_lazy = eventlog_lazy.with_row_index("id")

print("Collecting eventlog with streaming engine...")
eventlog = eventlog_lazy.collect(engine="streaming")

print("Created eventlog_v2 with", eventlog.height, "rows.")
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

print("Writing eventlog_v2.parquet...")
eventlog.write_parquet("../../data/processed/eventlog_v2.parquet")
print("Done.")

