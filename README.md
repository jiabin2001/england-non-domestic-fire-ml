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

The publisher URL is mutable: a checksum can verify a retained file but cannot recover it after replacement. Before dissertation deposit, archive the exact ODS together with `data/interim/other_building_fires_raw.parquet` and `data/processed/analysis_cohort.parquet` in durable institution-controlled storage. `outputs/metrics/data_archive_manifest.json` records their sizes and SHA-256 values; the data files remain excluded from Git to avoid treating source-data preservation as source-code versioning.

## Main design

- Main cohort: primary other-building fires, 2010/11–2023/24, exact duplicates removed, late calls excluded, and only unambiguously mapped target categories retained.
- Binary target: `LARGER_FIRE`, derived strictly from observed `SPREAD_OF_FIRE` strings. The main estimand uses the unambiguous room→floor→whole-building ordering and excludes `Roofs/ Roof spaces`. Official sources differ on whether roofs belong in the larger-fire grouping, so a roof-positive sensitivity tests the alternative published convention.
- Models: prior Dummy baseline, Logistic Regression, Random Forest and XGBoost. The latter three are the only main machine-learning models.
- Block A: structural and context information.
- Block B: retrospective incident information. Investigation and estimated-delay limitations mean this is not strictly a dispatch-time model.
- Block C: first-arrival prognostic information. It is reported separately and cannot support pre-incident or pre-arrival risk claims.
- Validation: a temporally ordered primary design and a target-stratified random comparator with exactly matching train/validation/test sample sizes.
- XGBoost execution device: explicitly fixed in `config/analysis.yaml`; `auto` remains available only as a visibly warned fallback because CPU/GPU training can produce different fitted artifacts.
- Primary metric: average precision (AP), calculated with scikit-learn's non-interpolated `average_precision_score`. Legacy `pr_auc` column and file stems retain their existing names for compatibility but store this AP value, not trapezoidal area under an interpolated precision–recall curve. Because AP's no-information baseline is approximately the positive prevalence, random–temporal comparisons are reported with prevalence context, ROC-AUC, Brier score and fixed-model bootstrap intervals. The validation-F1 threshold is an analytical operating point, not a business-optimal FRS decision rule.
- Interpretation: original-field grouped permutation importance is reported only for the validation-selected Temporal Block B XGBoost pipeline. It measures model dependence on the fixed temporal test set, not causal effects.

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

After activating the environment, the single command below rebuilds the audit, cohort, models, test predictions, bootstrap intervals, grouped permutation importance, prevalence-context table, robustness analyses, ten figures and reports from the cached Parquet, or performs the one-time ODS import if Parquet is absent:

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

To extend only the configured random split-assignment sensitivity analysis, without rerunning core model selection or the fixed-model bootstrap, use:

```powershell
python scripts/05_run_random_split_stability.py
python -c "from fireml.reporting import build_report; build_report()"
```

The targeted stability script recomputes all configured split seeds but skips the other robustness analyses, core model selection and fixed-model bootstrap.

Run tests with:

```powershell
python -m pytest -q
```

## Key outputs

- `reports/day1_feasibility.md`: feasibility checks and the defined main time window.
- `reports/hyperparameter_plan.md`: compact validation search plan.
- `reports/final_analysis_report.md`: results organised around RQ1–RQ3.
- `reports/methods_receipt.md`: exact source, cohort, target, features, split, parameters, thresholds, uncertainty/interpretability settings, seeds, software and manifest.
- `outputs/tables/bootstrap_confidence_intervals.csv`: 100,000-repeat partially paired fixed-model AP intervals; shared holdout records are resampled jointly so the observed overlap covariance is represented.
- `outputs/tables/grouped_permutation_importance.csv`: original-field Temporal Block B model-dependence estimates, summarised by mean AP decrease and permutation sample SD. Thirty repeats are not used to infer tail percentiles.
- `outputs/tables/pr_auc_prevalence_context.csv`: AP baseline, absolute lift and auxiliary normalized AP alongside ROC-AUC and Brier score. The legacy filename is retained for compatibility.
- `outputs/tables/random_seed_stability.csv`: seed-level results for 20 random split assignments with estimator seed and selected hyperparameters fixed.
- `outputs/tables/random_seed_stability_summary.csv`: median, IQR, range and sign-count receipt for the split-assignment AP sensitivity analysis.
- `outputs/tables/`: all requested audit, performance, stability, subgroup and sensitivity tables.
- `outputs/figures/`: ten figures in both PNG and PDF.
- `outputs/models/`: validation-selected fitted pipelines.
- `outputs/metrics/model_selection.json`: validation-selected hyperparameters, model families and analytical thresholds required for reproducibility.
- `outputs/metrics/data_archive_manifest.json`: checksums and sizes for the raw ODS and two reproducibility Parquet files that must be deposited separately.

No external GIS, weather, demographic, socioeconomic or commercial-building data are used. No neural network, deep learning, SMOTE comparison, stacking, Bayesian optimisation or large-scale Optuna search is included.
