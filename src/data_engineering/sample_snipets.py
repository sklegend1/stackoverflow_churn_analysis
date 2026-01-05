import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

Tp_sec = alpha * p95_gap_sec
window_start = cutoff - pl.duration(seconds=int(Tp_sec))

win = core_p.filter(
    (pl.col("timestamp") >= window_start) &
    (pl.col("timestamp") <= cutoff)
)

feat = (
    win.group_by(["user_id", "action_type_id"]).agg(pl.len().alias("count"))
       .pivot(values="count", index="user_id", on="action_type_id")
       .fill_null(0)
)
