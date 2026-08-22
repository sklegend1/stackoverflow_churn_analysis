# Repository Instructions

## Start Here

- Read [PROJECT_RECAP.md](PROJECT_RECAP.md) before quoting results or comparing model quality. Its figures were re-checked against `results/`.
- Use [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) as the best concept-to-script map.
- Use [CLAUDE.md](CLAUDE.md) for deeper pipeline notes, path pitfalls, and modeling limitations.
- Treat `docs/`, `docs_detailed/`, and `All Mds/` as background only. They are duplicated, partly stale, and sometimes contradict verified outputs.

## Working Model

- This is experimental research code, not a packaged application. Similarly named `_v2`, `_enhanced`, `_improved`, and `_optimized` scripts are separate experiments, not interchangeable entrypoints.
- There is no build system, no test suite, no linter, no formatter, no typechecker, and no CI.
- `src/models/xgboost/test_new_users_pipeline.py` is an experiment, not a unit test.
- Prefer minimal, local changes. Do not normalize cutoff, grace-period, or artifact-version differences across scripts unless the task is explicitly about that experiment design.

## Validation

- Install dependencies with `pip install -r requirements.txt`.
- `requirements.txt` is UTF-16. Do not rewrite it because it looks malformed in UTF-8.
- For a focused code-only check, use `python -m py_compile <changed-script.py>`.
- Full script execution is usually a large data or training run. Avoid using it as a default validation step.

## Execution Rules

- Run each standalone script from its own directory because relative paths and `sys.path` setup depend on the current working directory.
- Examples:
  - `python create_feature_matrix.py` with cwd `src/data_engineering`
  - `python tp_ap_xgboost.py` with cwd `src/models/xgboost`
  - `python simple_eval.py` with cwd `src/evaluation`
- Root launchers exist, but neither is fully authoritative:
  - `streamlit run app.py`
  - `python run_pipeline.py`

## Pipeline Pitfalls

- Prefer `src/data_engineering/create_core_eventlog_splited.py`. The filename typo is real.
- `create_core_eventlog_splited.py` writes the train/eval artifacts that downstream training and evaluation expect:
  - `core_eventlog_train_v1.parquet`
  - `core_eventlog_eval_v1.parquet`
- `create_core_eventlog.py` writes only `core_eventlog_v1.parquet` and is not a drop-in substitute.
- `run_pipeline.py` still uses the non-split core-log builder.
- `app.py` fixes that only for its XGBoost path; its LSTM path and some displayed output filenames remain stale.
- Several scripts still use bare shard paths like `filtered_users_parquet/*.parquet`. Under the normal cwd convention those paths are broken; the shards actually live under `data/processed/`.
- Do not casually run `import_*.py`. They are historical ingestion steps, and some contain hardcoded remote credentials.
- Docker is only for historical XML/MySQL ingestion. Do not start it for work based on existing Parquet artifacts.

## Data And Modeling Constraints

- Prefer Polars lazy scans and `collect(engine="streaming")` for ETL. Convert to pandas only at sklearn or XGBoost boundaries.
- `gain` means normalized XGBoost Gain feature importance, not reputation points.
- Personas are inherited `ClusterNmf20` labels. The original NMF model is lost, and persona reconstruction for new users is not reliable.
- The rule-based track covers 17 of 20 personas. Personas 9, 14, and 18 have no rule parameters.
- The measured temporal winner is `WA_recency`. Do not present `WA_trend` or `WA_enhanced` as the winner.
- Preserve the evaluation eligibility rule: score only users with at least one core event in `[cutoff - Tp, cutoff]`.
- Report `rate_ratio` alongside F1. F1 alone hid major over-prediction in this project.

## Output Discipline

- Before citing any metric from narrative docs, confirm it against [PROJECT_RECAP.md](PROJECT_RECAP.md) or the CSVs in `results/`.
- When a task depends on an exact artifact, check the currently uncommented input/output filenames near the top of the script before running it.
