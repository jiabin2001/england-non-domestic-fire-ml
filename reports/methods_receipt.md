# Methods receipt

## Source

- Official dataset: Other Building Fires Dataset
- Source page: https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets
- Guidance: https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance
- Download URL: https://assets.publishing.service.gov.uk/media/6a5e7a52c7c34404041b4663/Other_building_fires_dataset.ods
- Download timestamp (raw-file mtime): 2026-07-30T19:13:49.508579+00:00
- Metadata recorded: 2026-07-30T19:21:56.590944+00:00
- Official page update: 2026-07-22
- File size: 45066461 bytes
- SHA-256: `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`
- ODS sheets: [{"sheet_name": "Cover_sheet", "rows": 14, "columns": 1}, {"sheet_name": "Datasheet", "rows": 490412, "columns": 42}, {"sheet_name": "config", "rows": 6, "columns": 2}]
- Data sheet: Datasheet

## Cohort and target

- Main years: 2010/11, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24
- Exclusions, in order: outside main years; complete duplicate rows; `LATE_CALL=yes`; main-target exclusions (`Roofs/ Roof spaces` and any unmappable/missing category).
- Final rows: 209171; positives: 53804; prevalence: 0.25722495.
- Exact target mapping: `{"Whole Building/ Affecting more than 2 floors": 1, "Limited to room of origin": 0, "No fire damage": 0, "Roofs/ Roof spaces": null, "Limited to floor of origin (not whole building)": 1, "Limited to item 1st ignited": 0, "Limited to 2 floors": 1}`.
- Roof sensitivity: map `Roofs/ Roof spaces` to 1.

## Feature blocks

- Block A — structural and context: `['DAY_OF_WEEK', 'DAY_NIGHT', 'BUILDING_TYPE', 'FSO_APPLY', 'OCCUPIED_NORMAL']`
- Block B — retrospective incident information: `['DAY_OF_WEEK', 'DAY_NIGHT', 'BUILDING_TYPE', 'FSO_APPLY', 'OCCUPIED_NORMAL', 'OCCUPIED_TIME', 'ALARM_SYSTEM', 'SAFETY_SYSTEM', 'IGNITION_TO_DISCOVERY', 'DISCOVERY_TO_CALL', 'ACCIDENTAL_OR_DELIBERATE', 'CAUSE_OF_FIRE', 'IGNITION_POWER', 'SOURCE_OF_IGNITION', 'FIRE_START_LOCATION', 'ITEM_IGNITED']`
- Block C — first-arrival prognostic: `['DAY_OF_WEEK', 'DAY_NIGHT', 'BUILDING_TYPE', 'FSO_APPLY', 'OCCUPIED_NORMAL', 'OCCUPIED_TIME', 'ALARM_SYSTEM', 'SAFETY_SYSTEM', 'IGNITION_TO_DISCOVERY', 'DISCOVERY_TO_CALL', 'ACCIDENTAL_OR_DELIBERATE', 'CAUSE_OF_FIRE', 'IGNITION_POWER', 'SOURCE_OF_IGNITION', 'FIRE_START_LOCATION', 'ITEM_IGNITED', 'FIRE_SIZE_ON_ARRIVAL', 'OTHER_PROPERTY_AFFECTED_ON_ARRIVAL', 'RESPONSE_TIME']`
- All retained predictors are treated as categorical/banded fields. Missing/blank values become `Missing/Unknown`; one-hot encoding uses `handle_unknown='ignore'`. No rare-category merger was required because the largest field has 82 disclosed categories and sparse one-hot encoding remained tractable.
- `RESPONSE_TIME` is used; its redundant code field is not used.
- Leakage blacklist: `['SPREAD_OF_FIRE', 'ITEM_CAUSING_SPREAD', 'RAPID_FIRE_GROWTH', 'FIRE_DAMAGE_EXTENT', 'FIRE_DAMAGE_EXTENT_CODE', 'TOTAL_DAMAGE_EXTENT', 'TOTAL_DAMAGE_EXTENT_CODE', 'OTHER_PROPERTY_AFFECTED_CLOSE', 'TIME_AT_SCENE', 'TIME_AT_SCENE_CODE', 'TIME_AT_SCENE _CODE', 'VEHICLES', 'VEHICLES_CODE', 'PERSONNEL', 'PERSONNEL_CODE', 'FATALITY_CASUALTY', 'RESCUES', 'EVACUATIONS', 'EVACUATIONS_CODE']`
- Always excluded from predictors: `['FINANCIAL_YEAR', 'FRS_TERRITORY', 'LATE_CALL', 'LARGER_FIRE']` plus `E_CODE_TERRITORY`.

## Validation

- Temporal train: 2010/11, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20
- Temporal validation: 2020/21, 2021/22
- Temporal test: 2022/23, 2023/24
- Random comparator: stratified sampling with exactly matching train/validation/test counts.
- Primary seed: 20260801; stability seeds: [20260801, 20260802, 20260803].
- Selection metric: validation PR-AUC. Threshold: validation F1 maximum, an analytical operating point rather than an operational optimum.
- Holdout evaluation count in this analysis run: 1.

## Within-run evaluation records

Model configurations and thresholds were programmatically recorded before holdout evaluation within each analysis run. This is an internal procedural safeguard, not an externally timestamped preregistration.

- Pre-test record: `outputs/metrics/pre_test_model_config.json` (`internal_within_run_pre_test_configuration`), generated 2026-07-30T22:50:52.696824+00:00.
- Post-test record: `outputs/metrics/post_test_evaluation_receipt.json` (`internal_within_run_post_test_evaluation_receipt`), generated 2026-07-30T22:56:39.695483+00:00.
- The post-test record references pre-test SHA-256 `fdd3b98035488c4461f0b023ef50b99cb0e78f0adf8e2aece55794b914e290a8`. These files provide an auditable within-run order record; they do not independently verify researcher history outside the run.

## Hyperparameters and thresholds

- Full candidate ranges and results: `reports/hyperparameter_plan.md` and `outputs/tables/hyperparameter_search_results.csv`.
- Selected parameters: `{"temporal": {"logistic_regression": {"C": 0.1}, "random_forest": {"n_estimators": 200, "max_depth": null, "min_samples_leaf": 5, "max_features": "sqrt"}, "xgboost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8}}, "random": {"logistic_regression": {"C": 1.0}, "random_forest": {"n_estimators": 200, "max_depth": null, "min_samples_leaf": 5, "max_features": "sqrt"}, "xgboost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8}}}`
- Validation-selected family by block: `{"temporal": {"A": "xgboost", "B": "xgboost", "C": "xgboost"}, "random": {"A": "xgboost", "B": "xgboost", "C": "xgboost"}}`
- Locked thresholds: `{"temporal": {"A": {"dummy": 0.2499106767878746, "logistic_regression": 0.2867622375488281, "random_forest": 0.2722170425235016, "xgboost": 0.33036476373672485}, "B": {"dummy": 0.2499106767878746, "logistic_regression": 0.2742215096950531, "random_forest": 0.3519588449756026, "xgboost": 0.2836891710758209}, "C": {"dummy": 0.2499106767878746, "logistic_regression": 0.5661454796791077, "random_forest": 0.4307341088371494, "xgboost": 0.3571653366088867}}, "random": {"A": {"dummy": 0.25722577773877503, "logistic_regression": 0.272072970867157, "random_forest": 0.29698505349039045, "xgboost": 0.29469117522239685}, "B": {"dummy": 0.25722577773877503, "logistic_regression": 0.30903926491737366, "random_forest": 0.3399175365666454, "xgboost": 0.34557846188545227}, "C": {"dummy": 0.25722577773877503, "logistic_regression": 0.4314391613006592, "random_forest": 0.4773128881043004, "xgboost": 0.5401079654693604}}}`
- XGBoost device: `cuda`; tree method: `hist`.

## Fixed-model uncertainty and prevalence context

- PR-AUC intervals use 2,000 stratified bootstrap repeats with seed 20260811; positives and negatives are resampled separately with replacement so both class counts remain fixed.
- Random and temporal test sets overlap by 3,252 records (12.597815% of random test and 12.597815% of temporal test), based on `SOURCE_ROW_ID` and cross-checked against cohort index.
- The random-minus-temporal contrast separately resamples these partially overlapping holdouts as an approximate-independent comparison. It is not a paired bootstrap, the holdouts are not fully independent, and covariance induced by overlapping records is not explicitly modelled.
- Percentile limits are the 2.5th and 97.5th percentiles. They condition on fixed data splits, fitted models and selected hyperparameters; retraining and repeated model selection are outside their scope.
- PR-AUC baseline, absolute lift and normalized PR-AUC are prevalence-context diagnostics. Normalized PR-AUC is auxiliary and does not replace the primary PR-AUC definition.

## Grouped permutation importance

- The validation-selected Temporal Block B XGBoost pipeline is evaluated on the exact saved 2022/23–2023/24 test indices.
- Each of the 16 original Block B fields is permuted as a whole before the complete fitted preprocessing-and-model pipeline. This automatically groups all one-hot columns derived from that field.
- Each field uses 30 repeats with seed 20260821; importance is baseline PR-AUC minus permuted PR-AUC, with negative values retained.
- `permutation_p02_5` and `permutation_p97_5` are the 2.5th and 97.5th percentiles across random permutations. They describe permutation variability and are not confidence-interval limits.
- Importance measures model dependence, not a causal effect, and may be shared across correlated or overlapping fields.

## Software

```json
{
  "python": "3.12.13",
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

## Output manifest

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
- `outputs/metrics/audit_receipt.json`
- `outputs/metrics/cohort_receipt.json`
- `outputs/metrics/output_manifest.json`
- `outputs/metrics/post_test_evaluation_receipt.json`
- `outputs/metrics/pre_test_model_config.json`
- `outputs/metrics/predictions_random_block_A.parquet`
- `outputs/metrics/predictions_random_block_B.parquet`
- `outputs/metrics/predictions_random_block_C.parquet`
- `outputs/metrics/predictions_temporal_block_A.parquet`
- `outputs/metrics/predictions_temporal_block_B.parquet`
- `outputs/metrics/predictions_temporal_block_C.parquet`
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