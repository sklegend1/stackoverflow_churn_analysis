# Project Recap — Behavioral Churn Prediction on Stack Overflow

**Purpose of this file:** a single verified narrative of what this project did, in the order you
should present it. Every number here was re-checked against the raw CSVs in `results/`, not just
copied from the summary docs. Where the docs disagree with each other, the disagreement is
flagged and resolved from the data.

---

## The one-sentence version

> On a platform with no "cancel subscription" event, we had to *define* churn before we could
> predict it. We built an interpretable, per-segment rule (`WA < Ap` over a window `Tp`), pushed it
> to its ceiling (F1 ≈ 0.57), proved that ceiling was caused by **temporal drift** rather than by
> weak modeling, and then broke through it with a multi-task LSTM over raw activity sequences
> (median F1 **0.62 → 0.90** across 2016–2023, **+45.8%** average over the XGBoost baseline).

---

## Act 1 — The problem is definitional before it is predictive

Stack Overflow has no churn event. Users don't cancel; they fade. So:

1. **Churn had to be defined behaviorally**, independent of account deletion or deactivation.
2. **One global definition is wrong**, because a user who answers 40 questions a month and a user
   who asks two questions a year are not "inactive" at the same threshold.

That gives the central formula of the whole project:

```
WA(u, Tp) < Ap   →   churn signal
```

| Symbol | Meaning | How it's obtained |
|---|---|---|
| `Tp` | observation window, per persona | derived from data: `Tp = α × p95(gap between key actions)` |
| `Ap` | minimum effective activity, per persona | derived from data: threshold chosen on the precision–recall curve |
| `WA` | weighted activity in the window | sum of `gain` over that persona's *core* actions |

**Everything is per-persona.** 20 personas, from prior NMF clustering of 92,289 users
(H. Mohamadi's thesis). Archetype letters used in the docs: **H = Helper, C = Contributor,
T = Teacher, I = Inquirer** (+ a `SPECIALIST` label in one doc). These labels are *outcomes* of
the clustering, not inputs to it.

**Presenting tip:** lead with this. It's the intellectually strongest part of the project and it's
what makes the work transferable — the client's dataset will have the same problem.

---

## Act 2 — Data engineering: 40.6M events, no cluster

| Fact | Value |
|---|---|
| Users | 92,289 (the clustered cohort) |
| Events | 40.6M |
| Distinct action types | 26 |
| Span | 2016–2023 |
| Infrastructure | one machine, no Spark |

Spark was **deliberately rejected** as too heavy for exploratory preprocessing. The stack instead:

```
Stack Overflow XML  →  MySQL (Docker, port 3307)  →  DuckDB (ATTACH TYPE mysql)
                    →  sharded Parquet  →  Polars lazy scan  →  eventlog_v2.parquet
```

Key engineering decisions worth a slide each:

- **Actor consistency.** In `posthistory`, `UserId` *is* the actor. In `posts`, `OwnerUserId` is the
  content owner, not necessarily the actor. So PostHistory drove the event log and `posts` was used
  only for enrichment. **`votes` was excluded entirely** — no valid actor could be attributed.
- **Three event sources unified** into one `action_type_id` space: Posts (create question / create
  answer), PostHistory (edit, rollback, close, reopen, …), Comments (add comment).
- **Gap time is the core derived feature**, computed per user:
  ```python
  pl.col("timestamp").diff().over("user_id")   # → gap_time_actions_seconds / _days
  ```
- **Streaming, not in-memory:** `scan_parquet(...).collect(engine="streaming")` throughout.

Output: `data/processed/eventlog_v2.parquet` — event-based, persona-aware, sorted by
`(user_id, timestamp)`, gap-enriched.

---

## Act 3 — Interpretability-first: build the rule, honestly

The order of derivation matters, and it's the part you'll be teaching:

**Step 1 — Which actions matter for each persona?**
Train a per-persona XGBoost one-vs-rest classifier, extract **Gain** feature importance, keep the
**top-K action types** as that persona's *core activities*. Result: `core_activity_per_persona_named.csv`
(persona × action → gain). In the code, K = 7.

**Step 2 — How long should the window be?**
```
Tp = α × p95(gap_time_key_actions)
```
Searched α per persona. **Measured result:** α ∈ [5.5, 19.0], giving **Tp ∈ [61, 268] days**.

**Step 3 — What counts as "enough" activity?**
```
WA(u, Tp) = Σ gain(e)  for core events e in [cutoff − Tp, cutoff]
```
Train an XGBoost classifier on WA, then pick `Ap` from the precision–recall curve.
**Measured result:** Ap ∈ [0.159, 0.457].

**Step 4 — Label from the future, with a grace period.**
```
churn = 1  if no activity in [cutoff + grace, cutoff + grace + 365d]
```
`GRACE_DAYS` = 90 in the code, 180 in the docs' worked example — a tunable, not a finding.

**Step 5 — Evaluate with sliding cutoffs, report the median.**
Cutoffs at `2016-01-01 … 2023-01-01`. Median F1, not mean, because it's robust to one bad year.
Only users with ≥1 core event in `[cutoff − Tp, cutoff]` are eligible — this prevents inflating the
metrics with users who were never there.

### What it achieved

| Metric | Naive grid search | Final rule | Change |
|---|---|---|---|
| Average F1 | 0.353 | **0.576** | **+63%** |
| Average Precision | 0.289 | 0.430 | +49% |
| Average Recall | 0.460 | 0.816 | +77% |
| Personas covered | 13/20 | 17/20 | — |

### What it did NOT achieve — say this out loud

- **Rate ratio ≈ 1.9.** The rule predicted roughly **twice** the true churn rate. Recall 0.82 was
  bought with precision 0.44.
- **17/20 personas, not 20.** Verified from the CSVs: **personas 9, 14 and 18 have no rule-based
  parameters at all.** Note this for later — 18 and 9 turn out to be the LSTM's *two best* personas.
- Some docs claim "20 of 20 coverage." **That claim is not supported by the data. Do not use it.**

---

## Act 4 — The wall: temporal drift

This is the pivot of the whole story.

### The evidence

The same rule, same parameters, evaluated at different cutoffs:

| Year | Median F1 |
|---|---|
| 2017 | 0.43 |
| … | … |
| 2023 | 0.61 |

**F1 swings ±0.18 across years — a 42% relative gap between 2017 and 2023.** And the
over-prediction was worse in older years. That last detail is the tell: the error wasn't random, it
was **structural**.

### Why

1. **COVID (2020–21)** — remote-work surge changed engagement volume and rhythm.
2. **The LLM shift (2022+)** — ChatGPT launched **November 2022**; Copilot went mainstream.
   Developers substituted AI assistants for Stack Overflow. Question-asking fell.
3. **Platform evolution** — policy, features, community norms, demographics.

Net effect: **"normal activity" means something different in 2017 than in 2023. A threshold tuned
for 2023 is too strict for 2017.**

### Three fixes attempted, in increasing order of desperation

**Fix 1 — Better features.** Make WA time-aware:

```python
activity_trend  = clip((recent_gain + 1) / (old_gain + 1), 0.5, 2.0)   # 2nd half vs 1st half of window
recency_weight  = exp(-days_since_last / 30)                            # 30-day decay constant

WA_simple   = sum(gain)
WA_trend    = sum(gain) × activity_trend
WA_recency  = sum(gain) × recency_weight
WA_enhanced = sum(gain) × activity_trend × recency_weight
```

**⚠️ Your three docs name three different winners here. Resolved from the raw CSVs**
(`results/best_params/tp_ap_xgb_enhanced_best_params_WA_*.csv`, all written in the same run,
2025-12-21 15:51, n=17 personas each):

| Variant | Median of median_f1 | Mean of median_f1 |
|---|---|---|
| WA_simple | 0.5634 | 0.5614 |
| WA_trend | 0.5727 | 0.5712 |
| **WA_recency** | **0.5747** | **0.5804** |
| WA_enhanced | 0.5654 | 0.5740 |

→ **WA_recency wins, on both median and mean.** This is corroborated independently: the LSTM
training script loads `tp_ap_xgb_enhanced_best_params_WA_recency.csv` as its reference Ap/Tp.
*Present WA_recency. Ignore the docs that claim WA_trend (0.576) or WA_enhanced.*

Notice how small the spread is: 0.561 → 0.580 across all four variants. **Feature engineering had
plateaued.**

**Fix 2 — Calibration. Failed outright.**

| Method | Outcome |
|---|---|
| Platt / Isotonic / Beta | **Ap collapsed to 0.01** — below the minimum possible WA. Rule broken. |
| Temperature scaling | Made over-prediction *worse*: rate ratio 2.68 vs 2.0 baseline |
| Quantile matching | Fixed the rate (ratio 1.1) but F1 dropped to 0.48 |
| Ratio scaling | F1 0.555, ratio 1.95 — no real gain |

**Why it failed, and this is the key insight:** calibration assumes the feature→label relationship
is stable and only the output scale is off. Here **the feature itself changes meaning over time.**
Calibration is the wrong tool for drift.

**Fix 3 — Year multipliers. A band-aid, and measurably so.**

```
Adjusted_Ap = Ap_base × Year_Multiplier
```
(2017: 0.850, 2018: 0.941, 2019: 0.911, 2020: 0.938, 2021: 0.961, 2022: 0.976, 2023: 1.000)

| Approach | Median F1 | F1 Std |
|---|---|---|
| Baseline | 0.5603 | 0.1307 |
| Year multipliers | **0.5618** | 0.1302 |
| Era-based thresholds | 0.5618 | 0.1300 |

**+0.0015 F1.** That is noise against a ±0.18 year-to-year swing. And after adjustment the 2017→2023
spread was still 0.460 → 0.635. Some docs describe this as "maintains consistent performance across
all years" — **their own table contradicts that. Don't repeat it.**

### The diagnosis

> The rule-based system is a **fixed-threshold** system. Drift is not a calibration error; it is the
> non-stationarity of the data itself. To fix it at the root you need a model that **learns time
> directly.**

That sentence is the bridge to Act 5. It's also the most valuable thing you can teach the client.

---

## Act 5 — The LSTM: stop engineering the feature, feed the sequence

The insight: `WA` is a hand-designed scalar compression of a time series. Every drift patch was an
attempt to re-inject the time information that compression threw away. So don't compress.

### Architecture

```
Event sequence (last 50 events, each = [gain, time_delta])
        │
        ▼
  LSTM(hidden=128, layers=2, dropout 0.2–0.3)
        │
        ▼  hidden state (128)
        ├──── + Global features (4):  days_since_last, era, month, year
        └──── + Persona embedding (32)
        ▼
    concat = 164  →  FC 256  →  FC 128
        │
        ├──► Head 1: churn probability (0–1)
        ├──► Head 2: predicted Ap
        └──► Head 3: predicted Tp (days)
```

**Multi-task loss** — this is the elegant part. The model doesn't just classify; it *learns the rule's
own parameters*, per user, per moment in time:

```
loss = 2.0 × BCE(churn) + 1.0 × MSE(Ap) + 0.1 × MSE(Tp)
```

Ap/Tp targets come from the Act-3 rule-based results (`..._WA_recency.csv`). So the interpretable
phase wasn't discarded — **it became the supervision signal for the deep phase.** That's the single
best slide in the deck.

### How time enters the model

Three separate channels, which is why drift becomes learnable rather than an error term:
1. `time_delta` inside every event of the sequence (local rhythm)
2. `days_since_last` (current staleness)
3. `era` / `month` / `year` as global features (regime)

### Training

| | |
|---|---|
| Parameters | **300,131** (lightweight) |
| Training samples | **55,979**, from 2017–2022 |
| Temporal split | train 2017–2021, validate 2022 |
| Val loss | 1.06 → 0.95 |
| Optimizer schedule | ReduceLROnPlateau, early stopping patience 10, grad clip 1.0 |
| Model size | **1.2 MB** |
| Training time | ~15 min on GPU |
| Inference | ~100 users/s GPU, ~20 users/s CPU |

### Results — verified directly from `results/evaluations/lstm_eval_2016_to_2023.csv`

| Year | Personas | Users | Median F1 | Mean F1 | Std F1 | Rate ratio |
|---|---|---|---|---|---|---|
| 2016 | 20 | 12,944 | 0.6151 | 0.6129 | 0.0924 | 1.14 |
| 2017 | 20 | 14,859 | 0.6811 | 0.6730 | 0.0935 | 1.05 |
| 2018 | 20 | 15,895 | 0.7174 | 0.7353 | 0.0630 | 0.96 |
| 2019 | 20 | 16,487 | 0.7855 | 0.7991 | 0.0467 | 1.07 |
| 2020 | 20 | 16,865 | 0.8186 | 0.8273 | 0.0421 | 1.12 |
| 2021 | 20 | 17,154 | 0.8542 | 0.8593 | 0.0346 | 1.10 |
| 2022 | 20 | 17,353 | 0.8773 | 0.8844 | 0.0325 | 1.08 |
| 2023 | 20 | 17,442 | **0.9011** | **0.9050** | **0.0308** | **1.07** |

Reproduced exactly. **Average F1 2016–2023 = 0.7813. Average rate ratio = 1.07.**

### Head-to-head

| | Rule-based / XGBoost | LSTM |
|---|---|---|
| Median F1 2021 | 0.5809 | **0.8542** (+47.1%) |
| Median F1 2023 | 0.3855 – 0.6454 * | **0.9011** |
| F1 std | ~0.09–0.12 | **0.031–0.092** |
| Rate ratio | ~1.9–2.0 | **1.07** |
| Personas covered | 17/20 | **20/20** |
| Trend across years | degrades on old years | **improves monotonically** |

\* the docs quote both 0.3855 and 0.6454 for XGBoost 2023 depending on variant; the honest framing
is "**+39.6% to +134%** depending on which XGBoost configuration you compare against." Use +39.6%
if you want the conservative, unimpeachable number. Average improvement across all years: **+45.8%**.

### Per-persona highlights (2023)

| Persona | F1 | Precision | Recall | Users |
|---|---|---|---|---|
| 18 | **0.9712** | 0.9560 | 0.9870 | 207 |
| 9 | 0.9404 | | | 1,052 |
| 12 | 0.9391 | | | |
| 19 | 0.9336 | | | |
| 16 | 0.9295 | | | |

**Every one of the 20 personas exceeded F1 0.85 in 2023.** And — the punchline — **personas 18 and
9, the two best LSTM performers, are two of the three personas the rule-based system could not
handle at all.**

Weakest across all years: persona 14 (avg F1 0.6675, std 0.1941), persona 17 (0.7347).

---

## Act 6 — Why it worked (the part that transfers)

Not "deep learning is better." The actual reasons, in order:

1. **Time is in the input, so drift is a signal instead of an error.** Same reason year multipliers
   failed: they tried to correct *after* the information was destroyed.
2. **No feature engineering to plateau on.** The four WA variants spanned 0.561–0.580 — the ceiling
   wasn't the model, it was the compression.
3. **Multi-task learning regularizes.** Forcing the model to also predict Ap and Tp makes it learn
   *why* someone churns, not just *whether*.
4. **Per-user, per-moment thresholds** replace per-persona constants. The rule had 17 Ap values;
   the LSTM effectively has one per user per timestep.
5. **Persona as a 32-dim embedding**, not 20 separate models — small personas borrow strength from
   large ones. That's why 20/20 coverage was achievable where the rule got 17/20.

---

## Act 7 — Limitations. Present these; they are your credibility.

**The persona-reproduction problem — this is the big one.**
The original NMF model is **not available**. Attempts to re-derive the 20 personas from the
available data:

| Approach | Accuracy vs original personas |
|---|---|
| NMF re-clustering (29 features) | **7.47%** (random = 5%) |
| XGBoost classifier, 19 features | 24.23% |
| XGBoost classifier, 33 features | **46.08%** (best) |
| Neural net, 33 features | 28.36% |
| Production classifier (guide's v4) | 50.0% |

Verdict in the docs: **"persona classification is intractable with available features"** against a
≥60% production bar. Two causes: `comment_sentiment` (needs comment text + NLP — called "the
smoking gun") and age demographics are simply **not in the data**. Average cluster purity 15.4%;
two re-derived clusters swallowed ~58,000 users (63%) at <5% purity.

**What this means practically — and it's the crucial caveat for the client:** the churn methodology
is sound and transferable, but it **assumes you already own your segment labels.** The project
worked around this with a **held-out validation** design (80/20 stratified split of the 92,289
users, `random_state=42` → 73,831 train / 18,458 holdout) using *known* labels, rather than
predicting personas for new users.

**Other honest limitations:**
- **2016 is the weak year** (F1 0.615) — least training data from that era.
- **Persona 14 is unstable** (std 0.19); persona 17 inconsistent.
- **Interpretability drops.** The rule could explain *why*. The LSTM currently cannot.
- **GPU needed for training** (inference is CPU-viable).
- **Drift will recur.** Nothing here immunizes against GPT-6-era behavior change; it requires
  retraining.
- **Class imbalance** in the personas: 552 to 18,131 users.
- **Region extraction is keyword-matching on a free-text field** and carries 60–80% of the persona
  classifier's feature importance. On a new dataset with different location conventions that fails
  silently.

---

## Act 8 — Deployment: hybrid, not replacement

The recommended architecture keeps both systems, with different jobs:

| Layer | System | Role |
|---|---|---|
| Prediction engine | **LSTM** | production scoring, early-warning signal |
| Explanation & control | **Rule-based** | why-explanation, sanity check, monitoring, stakeholder-facing |

Framing from the project's own discussion, which is the right note to close on:

> The LSTM is not the failed replacement of the rule-based system, but the natural result of the
> **maturation of the problem analysis.** Each phase exposed the previous phase's limits and
> justified the next.

**Rollout plan:**
1. **Shadow mode (2–3 months).** LSTM scores everything, rule-based still decides. Log both. Monitor
   agreement rate, real outcomes, latency. Gate: LSTM precision and F1 beat the rule on live data.
2. **A/B test (2–3 months).** 50/50. Measure retention after warning, false-alarm complaints, TPR.
3. **Gradual rollout.** 50% → 75% → 90% → 100%, one week each. Alert on any >5% degradation. Keep
   the rule-based path as emergency fallback.

**Operational musts:** monitor F1 **and** rate_ratio (F1 alone hid the 2× over-prediction for a long
time). Retrain periodically — the docs say "quarterly" in one place and "annually" in another; pick
one and state it.

---

## Numbers you can quote without fear

Every one of these was re-verified against the raw artifacts:

- 92,289 users · 40.6M events · 26 action types · 20 personas · 2016–2023
- Rule-based: **F1 0.353 → 0.576** (+63%), recall 0.82, precision 0.44, **17/20 personas**,
  rate ratio **1.9**, Tp ∈ [61, 268] days, α ∈ [5.5, 19.0], Ap ∈ [0.159, 0.457]
- Best WA variant: **WA_recency**, mean F1 **0.5804**
- Drift: **±0.18 F1** across years; year multipliers bought **+0.0015**
- LSTM: **300,131 params**, **1.2 MB**, 55,979 training samples, val loss 1.06 → 0.95
- LSTM median F1: **0.6151 (2016) → 0.9011 (2023)**, average **0.7813**
- LSTM vs XGBoost: **+45.8%** average, **+47.1%** in 2021, **+39.6%** in 2023 (conservative)
- LSTM std F1 **0.031** in 2023 vs ~0.09–0.12 · rate ratio **1.07** vs **~2.0**
- **20/20 personas covered**, all above F1 0.85 in 2023

## Claims in the docs that the data does NOT support — do not present these

| Claim | Reality |
|---|---|
| "WA_trend is the winner (0.576)" | CSVs say WA_recency, 0.5804 mean / 0.5747 median |
| "WA_enhanced is the final variant" | Persian narrative only; not what the numbers or the LSTM script use |
| "20 of 20 personas covered" (rule-based) | 17/20. Personas 9, 14, 18 have no parameters |
| "Temporal adjustment maintains consistent performance across all years" | Post-adjustment spread is still 0.460 → 0.635 |
| "+66% F1 improvement" | +63% (0.353 → 0.576). Same doc says both |
| "gain = reputation points" | `gain` is XGBoost Gain feature importance. Different thing entirely |
| "30-day half-life" | It's a 30-day decay *constant* — `exp(-d/30)` |
