# England Other Building Fire Spread ML

Research code and saved outputs for the dissertation study:

> **Do random train–test splits overestimate fire-spread prediction? A temporally validated machine-learning study of non-domestic building fires in England**

**Revision status:** the current code and documentation corrections do not rerun the study. Numerical tables, fitted models, predictions and historical figures remain unchanged. The reviewed snapshot [`c67f6fc`](https://github.com/jiabin2001/england-non-domestic-fire-ml/tree/c67f6fcb52834abe2695bfc05ba5bcf6a9e40498) contains those artifacts; it is not established as their training commit. Read the [no-rerun revision note](reports/no_rerun_revision.md) for the corrected interpretation, artifact boundary and remaining experiments.

## Scope and interpretation

The official data source is the **Other Building Fires Dataset** published by the UK government for England. “Non-domestic building fires” is used here as an analytical term for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings. It can also include hotels, hostels, care homes and student halls that are not ordinary private dwellings; it must not be described as containing only buildings with no accommodation function.

This is a retrospective incident-level prediction study using completed Fire and Rescue Service records. It is not a real-time deployment tool, causal-inference study, annual building ignition-risk model or fire-physics simulation.

- Official entry page: [Fire statistics incident level datasets](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets)
- Official field and quality guidance: [Other building fires dataset guidance](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance)
- Current ODS used: [Other Building Fires Dataset](https://assets.publishing.service.gov.uk/media/6a5e7a52c7c34404041b4663/Other_building_fires_dataset.ods)

The study treats the retained raw ODS as immutable. Its recorded file size, download timestamp, SHA-256, page-update date, sheet inventory and data-sheet choice are in `data/raw/source_metadata.json` and `reports/methods_receipt.md`. The source metadata is committed; the ODS and the raw/cohort Parquet files are excluded from Git and absent from a clean checkout.

The publisher URL is mutable: a checksum can verify a retained file but cannot recover it after replacement. A separately downloadable archive of the exact historical data is not supplied by this repository. Before dissertation deposit, archive the exact ODS, its `source_metadata.json`, `data/interim/other_building_fires_raw.parquet` and `data/processed/analysis_cohort.parquet` in durable institution-controlled storage. `outputs/metrics/data_archive_manifest.json` records historical data-file sizes and SHA-256 values; its `exists` fields describe the original recording environment, not availability in a fresh checkout.

## Main design

- Main cohort: primary other-building fires, 2010/11–2023/24, exact duplicates removed, late calls excluded, and only unambiguously mapped target categories retained.
- Binary target: `LARGER_FIRE`, derived strictly from observed `SPREAD_OF_FIRE` strings. The main estimand uses the unambiguous room→floor→whole-building ordering and excludes `Roofs/ Roof spaces`. Official sources differ on whether roofs belong in the larger-fire grouping, so a roof-positive sensitivity tests the alternative published convention.
- Models: prior Dummy baseline, Logistic Regression, Random Forest and XGBoost. The latter three are the only main machine-learning models.
- Block A: structural and context information.
- Block B: retrospective incident information. Investigation and estimated-delay limitations mean this is not strictly a dispatch-time model.
- Block C: retrospective incident information plus arrival-state information. It inherits Block B's investigation fields, which need not be available when crews arrive; its high AP does not establish a deployable arrival-time predictor.
- `OCCUPIED_TIME`, retained in B and C, can include occupancy of buildings to which the fire has spread. It is a potential outcome proxy rather than a reliably initial-state field. No field-removal ablation has been performed in this revision.
- Validation: a temporally ordered primary design and a target-stratified random comparator with exactly matching train/validation/test sample sizes.
- Annual analysis: 2020/21–2021/22 also supplied model-selection labels, so their expanding-window results are descriptive development-period results. The 2022/23–2023/24 annual results occur after that selection period. The four rows are not four independent test folds.
- XGBoost execution device: explicitly fixed in `config/analysis.yaml`; `auto` remains available only as a visibly warned fallback because CPU/GPU training can produce different fitted artifacts.
- Primary metric: average precision (AP), calculated with scikit-learn's non-interpolated `average_precision_score`. Legacy `pr_auc` column and file stems retain their existing names for compatibility but store this AP value, not trapezoidal area under an interpolated precision–recall curve. Because AP's no-information baseline is approximately the positive prevalence, random–temporal comparisons are reported with prevalence context, ROC-AUC, Brier score and fixed-model bootstrap intervals. The validation-F1 threshold is an analytical operating point, not a business-optimal FRS decision rule.
- Interpretation: original-field grouped permutation importance is reported only for the validation-selected Temporal Block B XGBoost pipeline. It measures model dependence on the fixed temporal test set, not causal effects.

## Environment

The saved environment receipt records Python 3.12 and dependencies pinned in `pyproject.toml`. This revision preserves that receipt and does not independently establish the original training environment. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The acquisition script does not download data automatically. A full analysis requires either the retained raw Parquet with the committed `data/raw/source_metadata.json`, or the source ODS for a one-time import. If neither data file is available, obtain the ODS separately and place it at:

```text
data/raw/Other_building_fires_dataset.ods
```

Do not modify the source file. Compare it against the saved SHA-256 before calling a run a reproduction of the historical data. A newer file from the same URL constitutes a new data version. Once the raw Parquet and source metadata exist, the ordinary cached path does not re-import the ODS. The processed cohort alone is insufficient for the full audit and alternative-cohort analyses.

The recorded XGBoost device is `cuda`. A run intended to reproduce those settings needs compatible NVIDIA hardware and the recorded dependencies; matching them does not guarantee bit-for-bit output equality. Setting `xgboost_device: cpu` is supported for a separately recorded CPU experiment, but it is not an exact rerun of the saved CUDA models. Synthetic unit tests and saved-artifact reporting do not require CUDA.

## Full analysis or saved-artifact reporting

The following command is a **full rerun**. It rebuilds the audit, cohort, models, predictions, bootstrap intervals, permutation analysis, robustness analyses and reports, and overwrites their existing paths. Use a separate checkout if retaining the saved experiment:

```powershell
python scripts/06_build_report.py
```

To render reports from existing saved outputs without acquiring data, fitting models, bootstrapping or running permutation analysis, use:

```powershell
python scripts/07_render_report.py
```

The reporting command reads the committed source metadata, saved tables, receipts and per-record predictions; it does not require the external ODS or raw/cohort Parquet files. It recomputes report summaries, calibration/confusion tables and figures, so it is a reporting calculation rather than a zero-computation edit. It preserves `runtime_environment.json` and the historical data archive manifest, and records the current rendering environment separately in `report_environment.json`. Missing saved inputs must be restored; this command does not recreate them by training. It was not run for this documentation revision.

Scripts `01`–`05` expose individual acquisition, audit, cohort, model and robustness stages. They are not a complete replacement for `06`: the full orchestrator also runs uncertainty and permutation analyses. Running `06` after them repeats their work.

To extend only the configured random split-assignment sensitivity analysis, without rerunning core model selection or the fixed-model bootstrap, use:

```powershell
python scripts/05_run_random_split_stability.py
python scripts/07_render_report.py
```

The targeted stability script recomputes all configured split seeds but skips the other robustness analyses, core model selection and fixed-model bootstrap.

The targeted stability command fits models for the configured assignments and changes results; it is not part of a no-rerun update.

## Tests

After installing the test dependencies, the default suite uses synthetic data and temporary files. It does not need source data, saved models or CUDA:

```powershell
python -m pytest -q
```

Check the committed study artifacts separately:

```powershell
python -m pytest -q -m artifact
```

Real-data integration checks require the exact external ODS and both Parquet files at the paths above, alongside the committed source metadata. They are intentionally excluded from the default suite and CI:

```powershell
python -m pytest -q -m integration
```

Neither the unit suite nor saved-artifact consistency checks establish that corrected code reproduces historical model performance; that requires an explicit research rerun.

## Key outputs

- `reports/day1_feasibility.md`: feasibility checks and the defined main time window.
- `reports/no_rerun_revision.md`: corrections, unchanged historical artifacts and experiments still needed.
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

The unchanged historical PNG/PDF captions and `outputs/tables/feature_policy.csv` may retain earlier Block C and field-risk wording. The revision note and corrected Markdown interpretation take precedence; the preserved files are not evidence that the corrected code has been run.

No external GIS, weather, demographic, socioeconomic or commercial-building data are used. No neural network, deep learning, SMOTE comparison, stacking, Bayesian optimisation or large-scale Optuna search is included.
