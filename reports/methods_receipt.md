# Methods receipt

This receipt describes the analysis settings and saved outputs used to generate the report. Training and rendering environments are recorded separately.

## Source

- Official dataset: Other Building Fires Dataset
- Source page: https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets
- Guidance: https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance
- Download URL: https://assets.publishing.service.gov.uk/media/6a5e7a52c7c34404041b4663/Other_building_fires_dataset.ods
- Download timestamp (raw-file mtime, if recorded): 2026-09-20T20:46:44.568462+00:00
- Metadata recorded: 2026-09-20T20:55:40.884511+00:00
- Official page update: 2026-07-22
- File size: 45066461 bytes
- SHA-256: `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`
- ODS sheets: [{"sheet_name": "Cover_sheet", "rows": 14, "columns": 1}, {"sheet_name": "Datasheet", "rows": 490412, "columns": 42}, {"sheet_name": "config", "rows": 6, "columns": 2}]
- Data sheet: Datasheet
- Data archive manifest: `[{"role": "official source ODS", "path": "data/raw/Other_building_fires_dataset.ods", "exists": true, "size_bytes": 45066461, "sha256": "560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595"}, {"role": "one-time imported raw Parquet", "path": "data/interim/other_building_fires_raw.parquet", "exists": true, "size_bytes": 3279580, "sha256": "504a3520301a7d013d4ae93b0f753887839f1564d22003f4936059fbba2c8245"}, {"role": "main analysis cohort Parquet", "path": "data/processed/analysis_cohort.parquet", "exists": true, "size_bytes": 3932124, "sha256": "d595b7a5a161646dbc1068383c8137c497b7eb7d6df64c1b737799a1d9d583f3"}]`
- The source ODS and reproducibility Parquet files are excluded from Git and require a separate durable institution-controlled deposit. Checksums verify retained files but cannot recover them after a publisher URL is replaced; this manifest is not itself an archive.

## Cohort and target

- Main years: 2010/11, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24
- Exclusions, in order: outside main years; complete duplicate rows; `LATE_CALL=yes`; main-target exclusions (`Roofs/ Roof spaces` and any unmappable/missing category).
- Final rows: 209171; positives: 53804; prevalence: 0.25722495.
- Exact target mapping: `{"Whole Building/ Affecting more than 2 floors": 1, "Limited to room of origin": 0, "No fire damage": 0, "Roofs/ Roof spaces": null, "Limited to floor of origin (not whole building)": 1, "Limited to item 1st ignited": 0, "Limited to 2 floors": 1}`.
- Main-estimand rationale: the room→floor→whole-building ordering maps six categories unambiguously; `Roofs/ Roof spaces` is not assigned because it cannot be located unambiguously on that ordering.
- Roof-definition sensitivity: the source definitions documented for this study differ on roof spaces. The main target excludes roofs; the saved sensitivity maps them to 1. Consult the sensitivity table for its observed prevalence.

## Feature blocks

- Block A — structural and context: `['DAY_OF_WEEK', 'DAY_NIGHT', 'BUILDING_TYPE', 'FSO_APPLY', 'OCCUPIED_NORMAL']`
- Block B — retrospective incident information: `['DAY_OF_WEEK', 'DAY_NIGHT', 'BUILDING_TYPE', 'FSO_APPLY', 'OCCUPIED_NORMAL', 'OCCUPIED_TIME', 'ALARM_SYSTEM', 'SAFETY_SYSTEM', 'IGNITION_TO_DISCOVERY', 'DISCOVERY_TO_CALL', 'ACCIDENTAL_OR_DELIBERATE', 'CAUSE_OF_FIRE', 'IGNITION_POWER', 'SOURCE_OF_IGNITION', 'FIRE_START_LOCATION', 'ITEM_IGNITED']`
- Block C — retrospective incident information plus arrival-state information: `['DAY_OF_WEEK', 'DAY_NIGHT', 'BUILDING_TYPE', 'FSO_APPLY', 'OCCUPIED_NORMAL', 'OCCUPIED_TIME', 'ALARM_SYSTEM', 'SAFETY_SYSTEM', 'IGNITION_TO_DISCOVERY', 'DISCOVERY_TO_CALL', 'ACCIDENTAL_OR_DELIBERATE', 'CAUSE_OF_FIRE', 'IGNITION_POWER', 'SOURCE_OF_IGNITION', 'FIRE_START_LOCATION', 'ITEM_IGNITED', 'FIRE_SIZE_ON_ARRIVAL', 'OTHER_PROPERTY_AFFECTED_ON_ARRIVAL', 'RESPONSE_TIME']`
- Block C inherits retrospective cause/ignition fields from Block B; availability of the complete set at arrival has not been established. Its arrival-state information is close to the outcome. A single-field baseline comparison changes both information and model family and cannot isolate the effect of additional features.
- `OCCUPIED_TIME` may include occupants in buildings to which the fire spread, potentially encoding already-realised spread. Its removal is evaluated in the post-review fixed-configuration sensitivity, which does not identify the presence or amount of leakage.
- All retained predictors are treated as categorical/banded fields. Missing/blank values become `Missing/Unknown`; one-hot encoding uses `handle_unknown='ignore'`. No rare-category merger is applied.
- `RESPONSE_TIME` is used; its redundant code field is not used.
- Leakage blacklist: `['SPREAD_OF_FIRE', 'ITEM_CAUSING_SPREAD', 'RAPID_FIRE_GROWTH', 'FIRE_DAMAGE_EXTENT', 'FIRE_DAMAGE_EXTENT_CODE', 'TOTAL_DAMAGE_EXTENT', 'TOTAL_DAMAGE_EXTENT_CODE', 'OTHER_PROPERTY_AFFECTED_CLOSE', 'TIME_AT_SCENE', 'TIME_AT_SCENE_CODE', 'TIME_AT_SCENE _CODE', 'VEHICLES', 'VEHICLES_CODE', 'PERSONNEL', 'PERSONNEL_CODE', 'FATALITY_CASUALTY', 'RESCUES', 'EVACUATIONS', 'EVACUATIONS_CODE']`
- Always excluded from predictors: `['FINANCIAL_YEAR', 'FRS_TERRITORY', 'LATE_CALL', 'LARGER_FIRE']` plus `E_CODE_TERRITORY`.
- `FRS_TERRITORY` is excluded as a scope/geographic-generalisation decision, not classified as outcome leakage; no territory-inclusive sensitivity was estimated in this run.

## Validation

- Temporal train: 2010/11, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20
- Temporal validation: 2020/21, 2021/22
- Temporal test: 2022/23, 2023/24
- Random comparator: stratified sampling with exactly matching train/validation/test counts.
- Primary seed: 20260801; stability seeds: [20260801, 20260802, 20260803, 20260804, 20260805, 20260806, 20260807, 20260808, 20260809, 20260810, 20260811, 20260812, 20260813, 20260814, 20260815, 20260816, 20260817, 20260818, 20260819, 20260820].
- Selection metric: validation average precision (`average_precision_score`). Threshold: validation F1 maximum, an analytical operating point rather than an operational optimum.
- Legacy output columns and file stems named `pr_auc` store this non-interpolated AP value; no trapezoidal precision–recall curve area is calculated.
- Threshold provenance: the selected train-fitted pipeline and its validation-derived threshold are evaluated on test without a train+validation refit or test-set retuning.
- Annual models reuse the family and settings selected using 2020/21–2021/22. The 2020/21–2021/22 folds are development-period descriptive results, not independent temporal validation: model selection had access to outcomes from their period even though each model fits only earlier years. The 2022/23–2023/24 folds occur after that selection window and are later-year evaluations under the fixed selected settings. These folds are not a nested annual model-selection procedure.
- Expanding-window thresholded metrics use fixed threshold 0.5 and are not directly comparable with main-table thresholded metrics. Annual AP, prevalence-relative summaries and ROC-AUC are the intended temporal-stability comparisons.

## Hyperparameters and thresholds

- Full candidate ranges and results: `reports/hyperparameter_plan.md` and `outputs/tables/hyperparameter_search_results.csv`.
- Selected parameters: `{"temporal": {"logistic_regression": {"C": 0.1}, "random_forest": {"n_estimators": 200, "max_depth": null, "min_samples_leaf": 5, "max_features": "sqrt"}, "xgboost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8}}, "random": {"logistic_regression": {"C": 1.0}, "random_forest": {"n_estimators": 200, "max_depth": null, "min_samples_leaf": 5, "max_features": "sqrt"}, "xgboost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8}}}`
- Families with identical random and temporal selected hyperparameters: `['random_forest', 'xgboost']`.
- Families with differing selected hyperparameters, including their design-specific settings: `{"logistic_regression": {"random": {"C": 1.0}, "temporal": {"C": 0.1}}}`
- Validation-selected family by block: `{"temporal": {"A": "xgboost", "B": "xgboost", "C": "xgboost"}, "random": {"A": "xgboost", "B": "xgboost", "C": "xgboost"}}`
- Validation thresholds: `{"temporal": {"A": {"dummy": 0.2499106767878746, "logistic_regression": 0.2867622375488281, "random_forest": 0.2722170425235016, "xgboost": 0.33036476373672485}, "B": {"dummy": 0.2499106767878746, "logistic_regression": 0.2742215096950531, "random_forest": 0.3519588449756025, "xgboost": 0.2836891710758209}, "C": {"dummy": 0.2499106767878746, "logistic_regression": 0.5661454796791077, "random_forest": 0.4307341088371494, "xgboost": 0.3571653366088867}}, "random": {"A": {"dummy": 0.25722577773877503, "logistic_regression": 0.272072970867157, "random_forest": 0.29698505349039045, "xgboost": 0.29469117522239685}, "B": {"dummy": 0.25722577773877503, "logistic_regression": 0.30903926491737366, "random_forest": 0.3399175365666454, "xgboost": 0.34557846188545227}, "C": {"dummy": 0.25722577773877503, "logistic_regression": 0.4314391613006592, "random_forest": 0.47731288810430045, "xgboost": 0.5401079654693604}}}`
- XGBoost device: `cuda`; tree method: `hist`. The configured device is `cuda` and must match the recorded selection for a result-reproducing rerun.

## Fixed-model uncertainty and prevalence context

- Average-precision intervals use 100,000 partially paired, class-and-membership-stratified bootstrap repeats with seed 20260731.
- Random and temporal test sets overlap by 3,252 records (12.597815% of random test and 12.597815% of temporal test), based on `SOURCE_ROW_ID` and cross-checked against cohort index.
- Shared records are resampled jointly in both holdouts; random-only and temporal-only records are resampled independently. Outcome class and observed overlap membership counts remain fixed.
- Percentile limits are the 2.5th and 97.5th percentiles. They condition on fixed splits, fitted models and selected settings; repeated end-to-end selection is outside their scope.
- The repeat count controls Monte Carlo error in these fixed-model percentile limits; it does not address split-assignment or model-selection uncertainty.
- AP baseline, absolute lift and normalized AP are prevalence-context diagnostics. Normalized AP is auxiliary and does not replace the primary AP definition.

## Split-assignment stability

- The 20 saved split seeds are checked against `config/analysis.yaml`. Complete seed-level results are in `outputs/tables/random_seed_stability.csv`; seed-selection history is not reconstructed during report rendering.
- With estimator seed 20260801 and selected random-design hyperparameters fixed, the median random-minus-temporal AP difference was +0.013670 (IQR +0.009221 to +0.016894; range -0.001135 to +0.028112). Positive differences occurred for 19 of 20 assignments.
- The primary fixed-split bootstrap interval width was 0.030232; the across-split point range width was 0.029247. They describe dependent, different uncertainty sources and are not combined into a joint interval.
- This is an empirical split-assignment sensitivity analysis, not a Monte Carlo bootstrap or a repetition of end-to-end family/hyperparameter selection. It does not require a second bootstrap; the separate fixed-model bootstrap remains configured at 100,000 repeats.

## Grouped permutation importance

- The saved permutation results evaluate the validation-selected temporal Block B XGBoost pipeline on the saved 2022/23–2023/24 test indices.
- Each of the 16 original Block B fields is permuted as a whole before the complete fitted preprocessing-and-model pipeline. This automatically groups all one-hot columns derived from that field.
- Each field uses 30 repeats with seed 20260821; importance is baseline AP minus permuted AP, with negative values retained.
- Because 30 repeats do not resolve tail quantiles well, permutation variability is summarised by the sample standard deviation rather than empirical 2.5th/97.5th percentiles.
- Importance measures model dependence, not a causal effect, and may be shared across correlated or overlapping fields.

## Post-review diagnostic protocol

- Four additional fits: `[{"columns": ["DAY_OF_WEEK", "DAY_NIGHT", "BUILDING_TYPE", "FSO_APPLY", "OCCUPIED_NORMAL", "ALARM_SYSTEM", "SAFETY_SYSTEM", "IGNITION_TO_DISCOVERY", "DISCOVERY_TO_CALL", "ACCIDENTAL_OR_DELIBERATE", "CAUSE_OF_FIRE", "IGNITION_POWER", "SOURCE_OF_IGNITION", "FIRE_START_LOCATION", "ITEM_IGNITED"], "design": "random", "experiment_id": "b_minus_occupied_random", "model": "xgboost", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "n_estimators": 200, "subsample": 0.8}, "purpose": "Fixed-configuration removal sensitivity for OCCUPIED_TIME", "reference_block": "B", "reference_key": "random_B"}, {"columns": ["DAY_OF_WEEK", "DAY_NIGHT", "BUILDING_TYPE", "FSO_APPLY", "OCCUPIED_NORMAL", "ALARM_SYSTEM", "SAFETY_SYSTEM", "IGNITION_TO_DISCOVERY", "DISCOVERY_TO_CALL", "ACCIDENTAL_OR_DELIBERATE", "CAUSE_OF_FIRE", "IGNITION_POWER", "SOURCE_OF_IGNITION", "FIRE_START_LOCATION", "ITEM_IGNITED"], "design": "temporal", "experiment_id": "b_minus_occupied_temporal", "model": "xgboost", "parameters": {"colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "n_estimators": 200, "subsample": 0.8}, "purpose": "Fixed-configuration removal sensitivity for OCCUPIED_TIME", "reference_block": "B", "reference_key": "temporal_B"}, {"columns": ["BUILDING_TYPE"], "design": "temporal", "experiment_id": "building_type_temporal", "model": "logistic_regression", "parameters": {"C": 1.0}, "purpose": "Single-field categorical baseline with fixed C=1.0 and no search; comparison changes information and model family, not a pure feature increment", "reference_block": "B", "reference_key": "temporal_B"}, {"columns": ["FIRE_SIZE_ON_ARRIVAL"], "design": "temporal", "experiment_id": "arrival_size_temporal", "model": "logistic_regression", "parameters": {"C": 1.0}, "purpose": "Single-field categorical baseline with fixed C=1.0 and no search; comparison changes information and model family, not a pure feature increment", "reference_block": "C", "reference_key": "temporal_C"}]`.
- The primary run's cohort, ordered split assignments, source identifiers and labels are verified before fitting; reference predictions must agree with their recorded performance.
- Each pipeline fits preprocessing and the estimator on training records only. Validation chooses its F1 threshold; the same fitted pipeline is evaluated on test without a train+validation refit.
- Bootstrap repeats: 10,000; seed: 20260921. Paired intervals compare diagnostic minus full-reference AP; the occupancy-removal split contrast is random minus temporal with partially paired resampling.
- Average precision is sklearn's non-interpolated average_precision_score. Intervals are two-sided 95% percentile bootstrap intervals conditional on the fitted models and observed class counts. Same-test model comparisons align SOURCE_ROW_ID and resample positive and negative records separately, using the same sampled records for both models; differences are new minus reference. Random-minus-temporal comparisons additionally fix overlap membership: shared records are sampled jointly within outcome class, and design-only records are sampled independently. Thus test sizes, prevalences, and overlap counts stay fixed. AP calculations group tied scores before accumulating precision; bootstrap multiplicities are exact observation weights, not score bins. Difference intervals come directly from paired differences, not from comparing separate AP intervals. No model refitting, hyperparameter selection, threshold selection, year-level resampling, or multiplicity adjustment is included. These exploratory post-review analyses use previously examined test periods and do not create a new untouched or prospective validation set.
- Occupancy removal is a fixed-configuration sensitivity, not a leakage test. The single-field comparisons change both model family and information set. The new diagnostics use already-inspected test periods and are exploratory.
- Input/code hashes, complete fit settings, actual devices, software versions and result hashes: `outputs/diagnostics/experiment_receipt.json` and `outputs/diagnostics/uncertainty_receipt.json`. Predictions, fitted models and validation/test performance are retained alongside them.

## Training environment

The following versions and device settings were recorded during core model training.

```json
{
  "record_type": "model_training_environment",
  "python": "3.12.14",
  "platform": "Windows-11-10.0.26200-SP0",
  "pandas": "3.0.5",
  "numpy": "2.5.1",
  "scikit_learn": "1.9.0",
  "xgboost": "3.3.0",
  "xgboost_device": "cuda",
  "n_jobs": 4,
  "packages": {
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "pyarrow": "25.0.0",
    "odfpy": "1.4.1",
    "scikit-learn": "1.9.0",
    "xgboost": "3.3.0",
    "scipy": "1.18.0",
    "matplotlib": "3.11.1",
    "joblib": "1.5.3",
    "PyYAML": "6.0.3",
    "pytest": "9.1.1"
  }
}
```

## Current report-render environment

These versions describe report rendering. The training environment is recorded above.

```json
{
  "record_type": "report_render_environment",
  "rendered_at_utc": "2026-09-20T21:38:33.454636+00:00",
  "renderer_source_sha256": "ee1c6c091b0da2affa4ce53cee9d8b0d6e07b36d8b5d8cf8eaaf0656afefbc6d",
  "python": "3.12.14",
  "platform": "Windows-11-10.0.26200-SP0",
  "packages": {
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "pyarrow": "25.0.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "matplotlib": "3.11.1",
    "PyYAML": "6.0.3"
  },
  "scope": "Rendering saved artifacts only. Recomputes confusion/calibration and table summaries from saved predictions/results; no data acquisition, model fitting, bootstrap or permutation analysis. These versions are not training provenance.",
  "input_receipts": {
    "outputs/metrics/runtime_environment.json": {
      "sha256": "1e04b706574253219fa22da2e996d1c7f4403c79a176665c1f1332eed6921ea9",
      "role": "analysis input receipt"
    },
    "outputs/metrics/data_archive_manifest.json": {
      "sha256": "52d807c20c8db00a4a7fd56f339f22ef421362ca6fb0274db72cfa2a477565de",
      "role": "analysis input receipt"
    },
    "outputs/metrics/model_selection.json": {
      "sha256": "7efdcd267f2a16bd254069ac1d7b9b91a418fac62b04f0204bf15c3c44a81c28",
      "role": "analysis input receipt"
    }
  }
}
```

## Output manifest

- `outputs/diagnostics/config_snapshot.json`
- `outputs/diagnostics/experiment_receipt.json`
- `outputs/diagnostics/fit_log.jsonl`
- `outputs/diagnostics/models/arrival_size_temporal.joblib`
- `outputs/diagnostics/models/b_minus_occupied_random.joblib`
- `outputs/diagnostics/models/b_minus_occupied_temporal.joblib`
- `outputs/diagnostics/models/building_type_temporal.joblib`
- `outputs/diagnostics/performance.csv`
- `outputs/diagnostics/predictions/arrival_size_temporal_test.parquet`
- `outputs/diagnostics/predictions/arrival_size_temporal_validation.parquet`
- `outputs/diagnostics/predictions/b_minus_occupied_random_test.parquet`
- `outputs/diagnostics/predictions/b_minus_occupied_random_validation.parquet`
- `outputs/diagnostics/predictions/b_minus_occupied_temporal_test.parquet`
- `outputs/diagnostics/predictions/b_minus_occupied_temporal_validation.parquet`
- `outputs/diagnostics/predictions/building_type_temporal_test.parquet`
- `outputs/diagnostics/predictions/building_type_temporal_validation.parquet`
- `outputs/diagnostics/uncertainty.csv`
- `outputs/diagnostics/uncertainty_receipt.json`
- `outputs/figures/01_study_workflow.pdf`
- `outputs/figures/01_study_workflow.png`
- `outputs/figures/02_annual_count_prevalence.pdf`
- `outputs/figures/02_annual_count_prevalence.png`
- `outputs/figures/03_random_vs_temporal_pr_auc.pdf`
- `outputs/figures/03_random_vs_temporal_pr_auc.png`
- `outputs/figures/04_expanding_window_performance.pdf`
- `outputs/figures/04_expanding_window_performance.png`
- `outputs/figures/05_information_block_comparison.pdf`
- `outputs/figures/05_information_block_comparison.png`
- `outputs/figures/06_best_temporal_confusion_matrix.pdf`
- `outputs/figures/06_best_temporal_confusion_matrix.png`
- `outputs/figures/07_best_temporal_calibration.pdf`
- `outputs/figures/07_best_temporal_calibration.png`
- `outputs/figures/08_building_type_subgroups.pdf`
- `outputs/figures/08_building_type_subgroups.png`
- `outputs/figures/09_bootstrap_pr_auc_ci.pdf`
- `outputs/figures/09_bootstrap_pr_auc_ci.png`
- `outputs/figures/10_grouped_permutation_importance.pdf`
- `outputs/figures/10_grouped_permutation_importance.png`
- `outputs/figures/11_occupancy_ablation.pdf`
- `outputs/figures/11_occupancy_ablation.png`
- `outputs/figures/12_simple_baselines.pdf`
- `outputs/figures/12_simple_baselines.png`
- `outputs/metrics/audit_receipt.json`
- `outputs/metrics/cohort_receipt.json`
- `outputs/metrics/data_archive_manifest.json`
- `outputs/metrics/execution.json`
- `outputs/metrics/model_selection.json`
- `outputs/metrics/output_manifest.json`
- `outputs/metrics/predictions_random_block_A.parquet`
- `outputs/metrics/predictions_random_block_B.parquet`
- `outputs/metrics/predictions_random_block_C.parquet`
- `outputs/metrics/predictions_temporal_block_A.parquet`
- `outputs/metrics/predictions_temporal_block_B.parquet`
- `outputs/metrics/predictions_temporal_block_C.parquet`
- `outputs/metrics/report_environment.json`
- `outputs/metrics/runtime_environment.json`
- `outputs/models/best_random_block_A_xgboost.joblib`
- `outputs/models/best_random_block_B_xgboost.joblib`
- `outputs/models/best_random_block_C_xgboost.joblib`
- `outputs/models/best_temporal_block_A_xgboost.joblib`
- `outputs/models/best_temporal_block_B_xgboost.joblib`
- `outputs/models/best_temporal_block_C_xgboost.joblib`
- `outputs/tables/annual_incident_prevalence.csv`
- `outputs/tables/annual_major_building_type_proportions.csv`
- `outputs/tables/baseline_validation_performance.csv`
- `outputs/tables/best_temporal_calibration_curve.csv`
- `outputs/tables/best_temporal_confusion_matrix.csv`
- `outputs/tables/block_comparison.csv`
- `outputs/tables/bootstrap_confidence_intervals.csv`
- `outputs/tables/building_type_subgroup_performance.csv`
- `outputs/tables/category_lifecycle.csv`
- `outputs/tables/category_support_by_split.csv`
- `outputs/tables/category_support_by_year.csv`
- `outputs/tables/cohort_flow.csv`
- `outputs/tables/column_inventory.csv`
- `outputs/tables/cross_block_split_difference.csv`
- `outputs/tables/development_validation_performance.csv`
- `outputs/tables/drift_summary.csv`
- `outputs/tables/expanding_window_performance.csv`
- `outputs/tables/feature_policy.csv`
- `outputs/tables/grouped_permutation_importance.csv`
- `outputs/tables/hyperparameter_search_results.csv`
- `outputs/tables/hyperparameter_stability_summary.csv`
- `outputs/tables/missingness_by_year.csv`
- `outputs/tables/ods_sheet_inventory.csv`
- `outputs/tables/pr_auc_prevalence_context.csv`
- `outputs/tables/random_seed_stability.csv`
- `outputs/tables/random_seed_stability_summary.csv`
- `outputs/tables/random_to_temporal_difference.csv`
- `outputs/tables/random_validation_performance.csv`
- `outputs/tables/sensitivity_analysis_results.csv`
- `outputs/tables/split_assignments.csv`
- `outputs/tables/spread_of_fire_categories.csv`
- `outputs/tables/target_mapping.csv`
- `outputs/tables/temporal_validation_performance.csv`
- `outputs/tables/year_coverage.csv`
- `reports/day1_feasibility.md`
- `reports/final_analysis_report.md`
- `reports/hyperparameter_plan.md`
- `reports/methods_receipt.md`