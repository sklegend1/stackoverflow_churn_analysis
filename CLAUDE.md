# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **behavioral churn prediction on Stack Overflow users**. Churn is defined behaviorally, not contractually: a user is a churn signal when their weighted activity `WA(u, Tp)` in an observation window drops below a persona-specific threshold `Ap`:

```
WA(u, Tp) < Ap  →  Churn Signal
```

- **Tp** — observation window length, per persona; derived as `Tp = alpha * p95(gap between core actions)`
- **Ap** — effective-activity threshold, per persona
- **WA** — weighted sum of a persona's "core" actions in the window, weighted by `gain`
- **gain** — XGBoost **Gain feature importance** of an `(persona, action_type)` pair. *Not* reputation points; some docs get this wrong.
- **persona** — one of 20 user clusters from prior NMF clustering (`ClusterNmf20`); the original clustering model is lost and could not be re-derived (see Limitations below)

Three modeling tracks exist in parallel (this is experimental code — expect many versioned variants of the same idea):
1. **Rule-based / XGBoost** — data-driven estimation of `Tp`/`Ap` per persona (`src/models/rule_based`, `src/models/xgboost`).
2. **Temporal-drift improvements** — time-aware `WA` variants to survive distribution shift across years (COVID / LLM eras). Four variants exist (`WA_simple`, `WA_trend`, `WA_recency`, `WA_enhanced`); **`WA_recency` is the measured winner** and the one downstream code loads.
3. **LSTM** — end-to-end sequence model that predicts churn plus dynamic `Ap`/`Tp` directly, removing fixed thresholds (`src/models/lstm`).

## Start here when reading docs

Three root-level English documents are the reliable entry points, and they supersede the Persian prose:

- **`PROJECT_RECAP.md`** — verified narrative of what the project did, with every number re-checked against the raw CSVs in `results/`. Ends with a table of **claims the Persian docs make that the data does not support** (e.g. "WA_trend is the winner", "20/20 personas covered by the rule", "gain = reputation points"). Read this before quoting any figure from `docs/` or `All Mds/`.
- **`REPRODUCTION_GUIDE.md`** — the methodology as 10 transferable stages, each naming the script that implements it. Best map from concept → code.
- **`WORKFLOW.md`** (Persian) — the two pipelines as ordered steps.

Everything under `docs/`, `docs_detailed/`, and `All Mds/` is largely **Persian** narrative (`All Mds/Part_0_executive_overview.md` … `part_5_*.md` is the design narrative), heavily duplicated across the three folders, and **contains figures contradicted by the CSVs**. Treat it as background, not as ground truth.

## Running things

There is **no build system, no test framework, and no linter** (no `setup.py`, `pyproject.toml`, `Makefile`, or `pytest`), and the working tree is **not a git repo** (a `.gitignore` exists but there is no `.git`). Everything is standalone scripts.

```bash
pip install -r requirements.txt
```

`requirements.txt` is **UTF-16** encoded (looks space-separated if read as UTF-8; `pip` handles it). `torch` is pinned to a CUDA 12.1 build (`torch==2.5.1+cu121`) — LSTM *training* wants a GPU, inference is CPU-viable.

### Recommended: use `app.py`

```bash
streamlit run app.py
```

Launch from project root. This is the more accurate orchestrator — it uses `create_core_eventlog_splited.py` for the XGBoost track, which downstream scripts require.

### Alternative: `run_pipeline.py`

```bash
python run_pipeline.py
```

Also launched from root. **Warning:** step 2 runs the non-split `create_core_eventlog.py` for both tracks, but downstream scripts need the split outputs (`core_eventlog_train_v1.parquet`, `core_eventlog_eval_v1.parquet`). Only use this if you manually fix step 2 to call `_splited`.

The Docker service is **only** the local MySQL used during raw-data ingestion (maps host port `3307` → container `3306`); it is not needed once the parquet artifacts exist:

```bash
docker-compose up -d
```

### ⚠️ Script execution convention (most important thing to know)

Scripts use **hardcoded relative paths** and only work when the **current working directory is the script's own directory**. `run_pipeline.py` and `app.py` both enforce this by launching each script with `cwd=<script dir>`. When running a script by hand, `cd` into its folder first:

```bash
cd src/models/xgboost && python tp_ap_xgboost.py
```

Path depth depends on where the script lives:
- `src/data_engineering/*`, `src/analysis/*`, `src/evaluation/*` (2 levels deep) → `../../data/...`
- `src/models/{xgboost,lstm,rule_based}/*` (3 levels deep) → `../../../data/...`

Every script also does `sys.path.append(...'../../'|'../../../')` at the top for the same reason. When adding a script, match the depth of its neighbors.

**Exception — bare shard paths that don't resolve.** Several scripts read the raw shard directories with *no* prefix at all, e.g. `pl.scan_parquet("filtered_users_parquet/*.parquet")` in `create_eventlog_v2.py`, `create_eventlog_v1.py`, `nmf_clustering_clean.py`, `extract_enhanced_persona_features.py`, `persona_classifier_statistics_based.py`, `activity_stats.py`, `test_new_users_pipeline.py`, and the `persona_clustering_*` / `persona_predictor_production` scripts. Those directories actually live in `data/processed/` (`data/processed/filtered_users_parquet/`, …), so these scripts fail under the documented `cwd=<script dir>` convention. They were originally run from a directory where the shards sat alongside the script. If you need to re-run one, either fix the path to `../../data/processed/<name>_parquet/*.parquet` or run it from a directory containing the shards — don't assume the orchestrators handle it.

### ⚠️ `run_pipeline.py` step 2 is stale — prefer `app.py`

`run_pipeline.py` runs `create_core_eventlog.py` (non-split, writes only `core_eventlog_v1.parquet`) for **both** tracks, but the downstream scripts need the split outputs:

| Script | Reads |
|---|---|
| `models/xgboost/tp_ap_xgboost.py` | `core_eventlog_train_v1.parquet` |
| `models/lstm/lstm_temporal_ap_tp.py` | `core_eventlog_train_v1.parquet` |
| `evaluation/simple_eval.py` | `core_eventlog_eval_v1.parquet` |
| `evaluation/lstm_eval_multi_year.py` | `core_eventlog_eval_v1.parquet` |

Only `create_core_eventlog_splited.py` writes those two files. **In practice: always run `create_core_eventlog_splited.py`.** `app.py`'s XGBoost track does; its LSTM track still points at the non-split version. Some of `app.py`'s "Open output" paths are also stale (e.g. LSTM step 1 claims `feature_matrix.csv`, step 4 claims `lstm_eval_multi_year.csv`, but the scripts write `feature_matrix_v2.parquet` and `lstm_eval_2016_to_2023_only_stats.csv`) — the *run* is correct, only the file link is wrong.

## Data flow (big picture)

The pipeline is a chain of parquet artifacts under `data/processed/`. Understanding it requires reading several `src/data_engineering` scripts together:

```
Stack Overflow XML → MySQL (Docker) → DuckDB export → data/processed/*_parquet/ shards
   scripts: import_*.py (MySQL→MySQL, filtered to the 92,289 target users),
            *_to_parquet.py (DuckDB ATTACH TYPE mysql → chunked COPY to parquet shards)

*_parquet/ (filtered_posts, filtered_users, posthistory, comments)
   → create_eventlog_v2.py
        Unifies create-post + posthistory + comment events into one long-format
        event log. Encodes action_type_id (CREATE_QUESTION=1, CREATE_ANSWER=2,
        COMMENT=70, posthistory types remapped to 13–59). Computes per-user gaps
        via pl.col("timestamp").diff().over("user_id").
        NOTE: posthistory.UserId is the actor; posts.OwnerUserId is only the owner.
        Votes are excluded entirely — no attributable actor.
   → data/processed/eventlog_v2.parquet   ← the base table everything derives from
                                            (~40.6M events, 26 action types, 2016–2023)

eventlog_v2.parquet
   ├→ create_feature_matrix.py    → feature_matrix_v2.parquet (per-user aggregates; XGBoost/NMF input)
   ├→ nmf_clustering_clean.py     → 20-persona NMF clusters + nmf_cluster_statistics_20.csv
   └→ create_core_eventlog[_splited].py
        Keeps only each persona's top-7 core actions (by `gain` from
        core_activity_per_persona_named.csv) and recomputes gaps on the filtered
        log as gap_time_key_actions_seconds/_days. The _splited variant does a
        deterministic 80/20 per-user split (LCG hash of user_id, SEED=42) into:
          core_eventlog_train_v1.parquet   (models train on this)
          core_eventlog_eval_v1.parquet    (evaluation reads this)
```

Then, per track: **train** (`src/models/**`) → writes to `results/best_params/` and/or `checkpoints/`; **evaluate** (`src/evaluation/**`) → writes to `results/evaluations/`.

**XGBoost / rule-based track.** `tp_ap_xgboost.py` computes `p95(gap_time_key_actions)` per persona, grid-searches `alpha` (`np.arange(3, 10, 1)`) and cutoff years (2016–2020), **skipping any `Tp` outside 60–270 days**, labels churn from the future with a grace period, and writes `results/best_params/tp_ap_xgb_joint_search_sliding.csv`. `tp_ap_xgboost_enhanced.py` is the four-`WA`-variant version and writes one `results/best_params/tp_ap_xgb_enhanced_best_params_WA_<variant>.csv` per variant (plus a `WA_trend` copy under the un-suffixed legacy name, purely for backward compatibility — that legacy file is *not* the winner).

**LSTM track.** `lstm_temporal_ap_tp.py` consumes event sequences (last `SEQUENCE_LENGTH=50` events, each `[gain, time_delta]`) plus a 32-dim persona embedding, `nmf_cluster_statistics_20.csv`, and `results/best_params/tp_ap_xgb_enhanced_best_params_WA_recency.csv` as the **reference Ap/Tp supervision targets**. Multi-task loss `2.0*BCE(churn) + 1.0*MSE(Ap) + 0.1*MSE(Tp)`. It writes **both** `checkpoints/best_temporal_lstm_model.pt` and `checkpoints/lstm_model_metadata.pkl`; every LSTM eval script loads both, so the `.pkl` is not optional.

## Directory map

- `src/data_engineering/` — ingestion (`import_*`, `*_to_parquet`) and event-log/feature construction.
- `src/analysis/` — NMF clustering, persona classifiers, temporal-drift analysis, ad-hoc counts.
- `src/models/{rule_based,xgboost,lstm}/` — the three modeling tracks; filenames suffixed `_v2`, `_enhanced`, `_improved`, `_improved2`, `_optimized`, `_clean` are experiment iterations, not dead code.
- `src/evaluation/` — `simple_eval*`, `lstm_eval*`, `pragmatic_real_eval*`; note the grace-period and temporal variants.
- `src/visualization/` — LSTM architecture diagrams (graphviz).
- `data/processed/` — all parquet/CSV artifacts including the raw `*_parquet/` shard dirs (parquet is git-ignored; large, regenerated by the pipeline).
- `data/raw/` — `ActionTypes.csv` and the `ActionTypeId*.docx` files, which document the `action_type_id` encoding.
- `checkpoints/` — trained models (`*.pt` torch, `*.pkl`/`*.joblib` sklearn/xgboost) plus `lstm_model_metadata.pkl`.
- `results/{best_params,calibration,evaluations,figures}/` — model outputs and metrics; the CSVs here are the authority when docs disagree.
- `scripts/` — one-off doc/formatting utilities (Persian report generation, `organize_project_v*.py`), unrelated to the ML pipeline.
- `mysql-data/`, `xml/`, `xml.rar`, `processed-data/` — raw ingestion state / stray artifacts; not needed once parquet exists.

## Conventions & gotchas

- **Polars is the primary data library** (usually lazy: `scan_parquet` + `collect(engine="streaming")` for large tables); pandas is used only at the sklearn/XGBoost boundary. Prefer Polars for new data-engineering code. Spark was deliberately rejected — 40.6M events run on one machine.
- **Configuration is edited in place, not passed in.** No CLI args, no config files. Cutoffs, grace periods, alpha grids, and even *which params CSV to read* are literals near the top of each script, often with several alternatives left commented out (`simple_eval.py` has half a dozen commented `pd.read_csv` lines). Before running anything, read the first ~80 lines and check what's currently uncommented.
- **Grace period and cutoff constants are inconsistent between scripts, by design of the experiments.** `tp_ap_xgboost.py` uses `90 * grace_coef` days (coef 2 → 180) over sliding cutoffs 2016–2020; `simple_eval.py` uses `grace_days = 120` at a **single** hardcoded cutoff `2020-01-01`; `lstm_temporal_ap_tp.py` uses `GRACE_DAYS = 90`. Don't assume a number you read in one script applies in another, and don't "fix" one to match another without checking which results were produced with which.
- **Artifacts are versioned by filename suffix** rather than overwritten. When a script's output feeds another, check the exact filename. Several eval outputs differ only by suffix (`ap_tp_simple_eval.csv` vs `_adjusted` vs `_optimized` vs `_temporal_<year>_WA_recency`; `lstm_eval_2016_to_2023.csv` vs `_only_stats` vs `_full_features`).
- **Eligibility filter matters.** Evaluation only scores users with ≥1 core event in `[cutoff − Tp, cutoff]`. Removing it inflates every metric.
- **Track `rate_ratio` (predicted churn rate / actual), not just F1.** The rule-based system ran at ~1.9 — nearly 2× over-prediction — while F1 looked acceptable. This is the metric that caught it.
- Some `import_*.py` scripts contain **hardcoded remote MySQL credentials** and are historical ingestion steps — not part of the reproducible pipeline, and should not be re-run casually.
- `requirements.txt` includes the dummy `sklearn==0.0` package alongside the real `scikit-learn`; harmless, but don't treat it as the real dependency.

## Known limitations worth knowing before you build on this

Documented in `PROJECT_RECAP.md` (Act 7) and `REPRODUCTION_GUIDE.md` (Stage 0), and they constrain what new work is meaningful:

- **The personas are inherited and unreproducible.** The original NMF model is gone. Re-clustering reached **7.47%** agreement (random = 5%); the best supervised classifier reached **46–50%** against a 60% bar. Two inputs are simply absent from the data: comment sentiment (needs raw comment text) and age demographics. Everything in `src/analysis/persona_*` and `src/models/xgboost/*persona*` is that failed reconstruction effort. Work that assumes personas can be assigned to *new* users is not supported; the project used a held-out split of the 92k users with *known* labels instead.
- **Region features are keyword matching on a free-text field** and carry 60–80% of the persona classifier's importance — i.e. that classifier is largely a geography model.
- **The rule-based track covers 17/20 personas, not 20.** Personas 9, 14, 18 have no parameters. (Ironically 18 and 9 are the LSTM's two best.) Docs claiming 20/20 for the rule are wrong.
- **Drift is the central finding, and calibration does not fix it.** Platt/Isotonic/Beta collapsed `Ap` to 0.01; temperature scaling made over-prediction worse; per-year multipliers bought +0.0015 F1 against a ±0.18 year-to-year swing. The `src/models/rule_based/calibrate_*` and `adjust_ap_*` scripts are that dead end, kept for the record.
