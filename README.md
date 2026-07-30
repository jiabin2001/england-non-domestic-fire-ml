# England Other Building Fire Spread ML

Reproducible code and outputs for the dissertation study:

> **Do random train–test splits overestimate fire-spread prediction? A temporally validated machine-learning study of non-domestic building fires in England**

## Scope and interpretation

The official data source is the **Other Building Fires Dataset** published by the UK government for England. “Non-domestic building fires” is used here as an analytical term for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings. It can also include hotels, hostels, care homes and student halls that are not ordinary private dwellings; it must not be described as containing only buildings with no accommodation function.

This is a retrospective incident-level prediction study using completed Fire and Rescue Service records. It is not a real-time deployment tool, causal-inference study, annual building ignition-risk model or fire-physics simulation.

- Official entry page: [Fire statistics incident level datasets](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets)
- Official field and quality guidance: [Other building fires dataset guidance](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance)
- Current ODS used: [Other Building Fires Dataset](https://assets.publishing.service.gov.uk/media/6a5e7a52c7c34404041b4663/Other_building_fires_dataset.ods)

The raw ODS is immutable. Its file size, download timestamp, SHA-256, page-update date, sheet inventory and data-sheet choice are in `data/raw/source_metadata.json` and `reports/methods_receipt.md`.

## Main design

- Main cohort: primary other-building fires, 2010/11–2023/24, exact duplicates removed, late calls excluded, and only unambiguously mapped target categories retained.
- Binary target: `LARGER_FIRE`, derived strictly from observed `SPREAD_OF_FIRE` strings. `Roofs/ Roof spaces` is excluded in the main analysis and positive in the mandatory sensitivity analysis.
- Models: prior Dummy baseline, Logistic Regression, Random Forest and XGBoost. The latter three are the only main machine-learning models.
- Block A: structural and context information.
- Block B: retrospective incident information. Investigation and estimated-delay limitations mean this is not strictly a dispatch-time model.
- Block C: first-arrival prognostic information. It is reported separately and cannot support pre-incident or pre-arrival risk claims.
- Validation: a temporally ordered primary design and a target-stratified random comparator with exactly matching train/validation/test sample sizes.
- Primary metric: PR-AUC. The validation-F1 threshold is an analytical operating point, not a business-optimal FRS decision rule.

## Environment

The executed environment was Python 3.12 with dependencies pinned in `pyproject.toml`. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

If network access is unavailable and neither cached data file exists, manually download the official ODS to:

```text
data/raw/Other_building_fires_dataset.ods
```

Do not modify the file. Once Parquet exists at `data/interim/other_building_fires_raw.parquet`, the ODS is not read again.

## Full reproducible run

After activating the environment, the single command below rebuilds the audit, cohort, models, robustness analyses, figures and reports from the cached Parquet, or performs the one-time ODS import if Parquet is absent:

```powershell
python scripts/06_build_report.py
```

The explicit stage sequence is:

```powershell
python scripts/01_acquire_data.py
python scripts/02_audit_data.py
python scripts/03_build_cohort.py
python scripts/04_run_models.py
python scripts/05_run_temporal_robustness.py
python scripts/06_build_report.py
```

`06_build_report.py` is an end-to-end orchestrator, so running it after the preceding five scripts repeats the analysis. Use either the single-command route or the staged route.

Run tests with:

```powershell
python -m pytest -q
```

## Key outputs

- `reports/day1_feasibility.md`: blocking feasibility checks and locked time window.
- `reports/hyperparameter_plan.md`: post-baseline, pre-test compact search plan.
- `reports/final_analysis_report.md`: results organised around RQ1–RQ3.
- `reports/methods_receipt.md`: exact source, cohort, target, features, split, parameters, thresholds, seeds, software and manifest.
- `outputs/tables/`: all requested audit, performance, stability, subgroup and sensitivity tables.
- `outputs/figures/`: eight figures in both PNG and PDF.
- `outputs/models/`: validation-selected fitted pipelines.
- `outputs/metrics/locked_model_config.json`: proof that configuration was locked before the temporal test phase.

No external GIS, weather, demographic, socioeconomic or commercial-building data are used. No neural network, deep learning, SMOTE comparison, stacking, Bayesian optimisation or large-scale Optuna search is included.

