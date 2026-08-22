# Reproduction Guide — Running This Methodology on a New Dataset

**Audience:** a team that has their own user-activity data and wants the same churn system.

This is the transferable methodology, stripped of Stack Overflow specifics. Each stage names the
script in this repo that implements it, so you can read the reference implementation.

**Read `PROJECT_RECAP.md` first** for what the results were and which documented claims to distrust.

---

## Stage 0 — Check that your data can support this at all

The method needs exactly three things. If any is missing, stop and fix that first.

| Requirement | Why | Failure mode if absent |
|---|---|---|
| **An event log**: `(user_id, timestamp, action_type)` at event granularity | The whole method is built on inter-event gaps | Aggregated monthly counts cannot produce `Tp`, and the LSTM has no sequence to read |
| **≥3–4 years of history** | You need multiple sliding cutoffs, and you need to *see* drift before you can model it | You'll tune to one regime and silently overfit to it |
| **User segment labels you own** | Every threshold is per-segment | See the warning below — this is the trap this project hit |

### ⚠️ The segment-label trap — the single most important lesson

This project inherited 20 personas from prior NMF clustering, and **the original model was lost.**
Re-deriving the segments from the surviving data achieved **7.47% agreement** (random = 5%). Best
supervised classifier: **46–50%**, against a 60% production bar. Documented verdict:
*"persona classification is intractable with available features."*

Two features the clustering had used could not be recovered: **comment sentiment** (needed the raw
comment text) and **age demographics** (never stored).

**So, for your dataset — decide segmentation up front and own it end to end:**

- **Option A (recommended): derive your own segments, and version the model.** Save the fitted
  clustering artifact, the exact feature matrix, and the feature list. Anyone can then assign new
  users. This is what the project wishes it had.
- **Option B: use segments you already have** — plan tier, acquisition channel, role, region ×
  behavior. Free, stable, explainable, and no reproduction risk.
- **Option C: skip segments entirely.** Fit one global `Tp`/`Ap`. Weaker but honest, and a fine
  baseline to beat.

Do **not** take Option D: inherit undocumented labels from someone else's model.

**If you do cluster:** avoid features you can't recompute later (anything requiring NLP over text
you don't retain, anything manually collected). Watch class imbalance — here it ran 552 to 18,131
users, and every small segment underperformed. And beware proxy dominance: geography carried
**60–80%** of this project's classifier importance while behavioral ratios contributed **1.5%**,
which meant the segmentation was largely a geography model wearing a behavior costume.

---

## Stage 1 — Build the event log

*Reference: `src/data_engineering/create_eventlog_v2.py`*

Target schema, one row per event:

```
user_id | timestamp | action_type_id | segment | gap_time_actions_seconds | gap_time_actions_days
```

Decisions to make explicitly, because each one bit this project:

1. **Who is the actor?** Distinguish "user who performed the action" from "user who owns the
   object." Here, `posthistory.UserId` was the actor but `posts.OwnerUserId` was the owner — mixing
   them would have attributed edits to the wrong person.
2. **Drop event types with no attributable actor.** Votes were excluded entirely for this reason.
   An event you can't attribute is noise, not signal.
3. **Unify all sources into one `action_type_id` space** before computing anything. Three separate
   sources here (posts, edit history, comments) became one integer namespace.
4. **Compute gaps per user, after sorting:**
   ```python
   df = df.sort(["user_id", "timestamp"])
   df = df.with_columns((pl.col("timestamp").diff().over("user_id") / 1_000_000)
                        .alias("gap_time_actions_seconds"))
   ```

**Scale advice, if you're on one machine:** this ran 40.6M events with no Spark. Polars lazy
(`scan_parquet` → `collect(engine="streaming")`) for ETL, DuckDB for analytical queries, Parquet for
storage, staged files between steps. Convert to pandas only at the sklearn/XGBoost boundary.
Spark was evaluated and rejected as too heavy for exploratory work at this size.

---

## Stage 2 — Find each segment's core actions

*Reference: `src/data_engineering/create_core_eventlog_splited.py`*

Not all actions signal engagement equally, and they differ per segment.

1. Build a per-user feature matrix of action counts (`create_feature_matrix.py`).
2. Train a **one-vs-rest classifier per segment** (XGBoost here) predicting segment membership.
3. Extract **Gain** feature importance → this is `gain` for each `(segment, action_type)`.
4. Keep the **top-K actions per segment** as its *core activities*. K = 7 in this implementation.
5. Filter the event log to core actions only → the **core event log**. Recompute gaps on this
   filtered log (`gap_time_key_actions_seconds`) — gaps between *key* actions, not all actions.

> **Terminology warning:** `gain` here means *XGBoost Gain feature importance*. Some docs in this
> repo sloppily call it "reputation points." It is not. Keep this straight or your client will build
> the wrong thing.

**Also do your train/eval split here, at the user level, before any modeling.** This implementation
uses a deterministic LCG hash of `user_id` (SEED=42, 80/20) so the split is reproducible and works
inside a Polars lazy query. Splitting by user — not by event and not by time — is what prevents
leakage.

---

## Stage 3 — Derive the window `Tp` from data, not from a meeting

*Reference: `src/models/xgboost/tp_ap_xgboost.py`*

```
Tp = α × p95(gap_time_key_actions)     per segment
```

Take the 95th percentile of inter-key-action gaps for the segment — its natural rhythm — and scale
it by a searched coefficient α.

Measured here: α ∈ [5.5, 19.0] → **Tp ∈ [61, 268] days**. The reference code searches
`np.arange(3, 10, 1)`; the enhanced run used a wider grid with 0.5 steps.

Sanity-clamp the result (this code skips any Tp outside 60–270 days). A segment whose Tp comes out
at 5 days or 5 years means the gap distribution is pathological — investigate rather than accept.

**Why p95 and not a round number:** it adapts per segment automatically and it's defensible in a
review. "180 days because it felt right" is not.

---

## Stage 4 — Define the label, with a grace period

```
churn = 1  if the user has no activity in [cutoff + grace, cutoff + grace + horizon]
```

```python
GRACE_DAYS  = 90     # 180 in the docs' example — tune to your product's rhythm
HORIZON     = 365
```

The grace period is the point people miss: it absorbs vacations and busy stretches so you don't
label temporary quiet as churn. Set it from your own return-time distribution — e.g. the point
beyond which historical return probability drops below ~5%.

**Eligibility filter — do not skip this.** Only score users with ≥1 core event in
`[cutoff − Tp, cutoff]`. Without it you pad your metrics with users who were never active, and every
number you report is inflated.

---

## Stage 5 — Compute weighted activity, and make it time-aware

```python
# Baseline
WA_simple = sum(gain for core events in [cutoff - Tp, cutoff])

# Time-aware components
mid            = (cutoff - Tp) + Tp/2
activity_trend = clip((recent_gain + 1) / (old_gain + 1), 0.5, 2.0)   # 2nd half vs 1st half
recency_weight = exp(-days_since_last / 30)                           # 30-day decay constant

WA_trend    = WA_simple × activity_trend
WA_recency  = WA_simple × recency_weight
WA_enhanced = WA_simple × activity_trend × recency_weight
```

Evaluate all four on **your** data. On this dataset the winner was **WA_recency** (mean F1 0.5804 vs
0.5614 for simple) — but the spread across all four was only 0.561–0.580, so don't expect a
transformation. Yours may rank differently; recency mattered more than trend *here*.

The `+1` in the trend ratio is there to avoid division by zero; the `clip(0.5, 2.0)` stops a user
with one recent event from getting an unbounded multiplier.

---

## Stage 6 — Derive `Ap`, then evaluate across time

**Ap** — two documented routes, pick one:
- **Percentile:** sweep percentiles 10–90 of the segment's WA distribution, keep the best F1. Simple,
  no model.
- **Classifier + PR curve:** train a classifier on WA, then choose the threshold from the
  precision–recall curve at your chosen operating point. Used here with targets recall 0.8 /
  precision 0.65.

**Then evaluate with sliding cutoffs — this is non-negotiable.** One cutoff tells you nothing about
stability. Run the full pipeline at every year boundary:

```python
CUTOFFS = ["2016-01-01", "2017-01-01", ..., "2023-01-01"]
```

**Report median F1 across cutoffs, and report the standard deviation.** Median because one anomalous
year shouldn't dominate; std because it *is* the drift measurement.

**Track `rate_ratio = predicted_churn_rate / actual_churn_rate` alongside F1.** This project ran at
**1.9** — predicting nearly twice the true churn rate — while F1 looked acceptable. F1 alone hid it
for a long time. Make it a first-class metric from day one.

**Expected shape of results at this stage:** F1 ≈ 0.55–0.60, recall high (~0.82), precision mediocre
(~0.44), rate ratio ~2, and **noticeably worse F1 on your oldest cutoffs.** That last part is drift,
and it's Stage 7.

---

## Stage 7 — Diagnose drift before you try to fix it

Plot median F1 against cutoff year. If it slopes, you have drift. Here it swung **±0.18** — 0.43 in
2017 vs 0.61 in 2023, a 42% relative gap — and over-prediction was systematically worse in older
years. That asymmetry is the diagnostic: a random error wouldn't correlate with time.

**Then resist these three temptations, in this order. All were tried here. All failed.**

| Attempted fix | Result | Why it fails |
|---|---|---|
| **Probability calibration** (Platt, Isotonic, Beta) | **Ap collapsed to 0.01**, below minimum possible WA. Rule broken. | Calibration assumes the feature→label mapping is stable and only output scale is wrong |
| **Temperature scaling** | Over-prediction got *worse*: rate ratio 2.68 vs 2.0 | Same wrong assumption |
| **Quantile matching** | Rate fixed (1.1) but F1 fell to 0.48 | Traded the metric you care about for the one you were monitoring |
| **Per-year threshold multipliers** | **+0.0015 F1** (0.5603 → 0.5618) | Corrects the output after the time information was already destroyed |

**The lesson, stated plainly:** under drift, **the feature itself changes meaning over time.** No
post-hoc correction of a time-blind feature can recover that. If you're reaching for per-year
multipliers, that's the signal to change model class instead.

*(Era-based thresholds — separate parameters per regime — reached the same 0.5618 with marginally
better std. Also not worth the complexity.)*

---

## Stage 8 — Move to a sequence model

*Reference: `src/models/lstm/lstm_temporal_ap_tp.py`*

Stop compressing the time series into a scalar. Feed the sequence.

```
Event sequence (last N=50 events, each = [gain, time_delta])
        │
   LSTM(hidden=128, layers=2, dropout 0.2–0.3)
        │  → hidden (128)
        ├─ + Global features (4): days_since_last, era, month, year
        └─ + Segment embedding (32)
        │
   concat 164 → FC 256 → FC 128
        ├──► churn probability
        ├──► predicted Ap
        └──► predicted Tp
```

```python
loss = 2.0 * BCE(churn_pred, churn_true) \
     + 1.0 * MSE(ap_pred,  ap_reference) \
     + 0.1 * MSE(tp_pred,  tp_reference)
```

**The three design choices that actually matter — copy these, not the hyperparameters:**

1. **`time_delta` inside every event.** This is why drift becomes learnable rather than an error
   term. Without it you've built a fancier version of the same time-blind feature.
2. **Regime features as globals** (`era`, `year`, `month`). Give the model the calendar explicitly.
   `era` here was `pre_covid (2016–19) / covid (2020–21) / post_llm (2022–24)` — define your own
   regimes from your own domain history.
3. **Multi-task heads predicting Ap and Tp, supervised by your Stage-6 results.** Your interpretable
   phase becomes the teacher for the deep phase. This regularizes the model *and* keeps the earlier
   work alive as an explanation layer. It's the best idea in the project — don't drop it to save
   time.

Also: **segment as an embedding, not as separate models.** 32 dims. Small segments borrow strength
from large ones — which is why the LSTM covered 20/20 segments where the rule covered 17/20, and why
the two best-performing segments were ones the rule couldn't handle at all.

**Training recipe:** temporal split (train on earlier years, validate on a later one — *not* random),
ReduceLROnPlateau, early stopping patience 10, gradient clipping 1.0, batch 256, lr 0.001. Here:
55,979 samples, 300,131 parameters, ~15 min on GPU, 1.2 MB model.

**Expected outcome if it worked:** F1 that *improves* toward recent years instead of degrading, std
dropping to ~0.03, and rate ratio approaching 1.0. Here: F1 0.6151 (2016) → 0.9011 (2023),
std 0.031, rate ratio 1.07 versus ~2.0 before.

**Expected failure mode:** your oldest year stays weakest, because it has the least training data.
2016 sat at F1 0.615 here while everything from 2019 on cleared 0.78. That's acceptable; just say so
rather than hiding it.

---

## Stage 9 — Validate honestly

- **Evaluate every year separately, and every segment separately.** Aggregate numbers hide the
  failures that matter. Persona 14 here averaged F1 0.6675 with std 0.19 while the headline said 0.78.
- **Report median + std + rate_ratio + per-segment coverage.** Four numbers, not one.
- **Never evaluate on a cutoff you tuned on.** Here 2023 was held out entirely from training
  (trained 2017–2022) — which is exactly why the 2023 result is the credible one.
- **Compare against the rule-based baseline you already built.** "+45.8% over baseline" is a much
  stronger claim than "F1 = 0.90," and you can only make it because you built the baseline first.

---

## Stage 10 — Deploy as a hybrid

| Layer | Component | Job |
|---|---|---|
| Prediction | sequence model | production scoring, early warning |
| Explanation & control | rule-based system | why-explanation, sanity check, drift monitoring, stakeholder reporting |

Keep both. The rule tells a stakeholder *"this user's weighted activity is 0.18, below the 0.29
threshold for their segment"* — the LSTM cannot.

**Rollout:** shadow (2–3 mo, log both, rule still decides) → A/B 50/50 (2–3 mo, measure retention
and false-alarm complaints) → gradual 50/75/90/100% with the rule as fallback. Alert on any >5% F1
degradation.

**Ongoing:** monitor F1 *and* rate_ratio, retrain on a fixed cadence (pick quarterly or annual and
commit — this repo's docs say both), and expect a new drift regime eventually. Nothing here
immunizes you against the next platform shift; the retraining pipeline is the immune system.

---

## The compressed version — 10 rules

1. Define churn behaviorally before you model it. On most platforms there is no churn event.
2. Segment first. One global threshold is wrong for heterogeneous users.
3. **Own your segmentation model.** Never inherit undocumented labels — re-derivation scored 7.47% here.
4. Derive windows from data (`α × p95(gap)`), not from meetings.
5. Use a grace period, and filter to eligible users, or your metrics lie to you.
6. Slide the cutoff across every year. Report median **and** std.
7. Track `rate_ratio` from day one. F1 hid a 2× over-prediction here.
8. When you see drift, do not calibrate. Calibration assumes stationarity — that's the thing you
   just lost.
9. If you reach for per-year multipliers, change model class instead. They bought +0.0015 F1.
10. Build the interpretable model first anyway — it becomes your baseline, your explanation layer,
    and the supervision signal for the deep model.
