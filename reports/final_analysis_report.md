# Final analysis report

## Study definition

This retrospective prediction study uses the official **Other Building Fires Dataset** for England. “Non-domestic building fires” is the dissertation's analytical wording for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings and can include hotels, hostels, care homes and student halls; it is not limited to buildings without accommodation functions.

The analysis predicts incident-level final fire spread among already-recorded primary fires. It is not a causal study, annual building fire-risk model, fire-physics simulation or real-time FRS deployment tool. Block B contains retrospectively recorded incident information; Block C is a separately interpreted first-arrival prognostic model.

## Data and cohort

The official ODS was updated 22 July 2026. The raw-file SHA-256 is `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`. The audit found 245,207 logical incident rows across 2010/11–2025/26. The locked main period is 2010/11–2023/24; 2024/25 is excluded because Suffolk submissions are incomplete for part of that year, and 2025/26 is excluded because it spans the IRS-to-FaRDaP transition.

After year restriction, 20 exact duplicates, 3,063 late calls and 6,081 `Roofs/ Roof spaces` target records were removed sequentially. The main cohort contains 209,171 incidents, 53,804 larger fires (25.7%).

This roof exclusion is an estimand decision, not a claim that the official statistics use the same binary definition. The study's main target follows the unambiguous room→floor→whole-building ordering; `Roofs/ Roof spaces` cannot be placed unambiguously on that scale. Official FIRE0304 instead counts roofs/roof spaces as a larger fire. The explicitly reported official-definition sensitivity maps it positive and tests the consequence of that convention.

## Validation and modelling

Temporal train/validation/test years are 2010/11–2019/20, 2020/21–2021/22 and 2022/23–2023/24. The stratified random comparator has exactly the same 159,533/23,824/25,814 sample sizes. All imputing and encoding were pipeline-fitted on development data only. Compact hyperparameter selection used validation PR-AUC; analytical classification thresholds maximised validation F1. Each threshold was selected from validation probabilities produced by a train-fitted model, then held fixed while the selected model was refitted on train+validation. Because refitting can shift the probability distribution, threshold-dependent test metrics are descriptive operating-point summaries; threshold-free PR-AUC remains primary, and no test-set retuning occurred. Configurations and thresholds were programmatically recorded before the single within-run holdout-evaluation phase.

Random Forest, XGBoost selected identical hyperparameters under random and temporal development designs. In particular, the primary XGBoost RQ1 contrast is not confounded by comparing different XGBoost configurations; Logistic Regression selected different regularisation strengths (`C=1.0` random versus `C=0.1` temporal).

The main model comparison is Block B, the retrospective incident-information model:

| model | n | positive_count | positive_prevalence | pr_auc | roc_auc | f1 | brier_score |
|---|---|---|---|---|---|---|---|
| XGBoost | 25814 | 6850 | 0.265 | 0.642 | 0.840 | 0.638 | 0.136 |
| Random Forest | 25814 | 6850 | 0.265 | 0.635 | 0.836 | 0.634 | 0.138 |
| Logistic Regression | 25814 | 6850 | 0.265 | 0.629 | 0.831 | 0.631 | 0.139 |

## Research questions

### RQ1 — Random versus temporal validation

For the validation-selected Block B XGBoost, random holdout PR-AUC was 0.658 (95% stratified bootstrap CI 0.646–0.669) and temporal holdout PR-AUC was 0.642 (0.631–0.653). The random-minus-temporal point difference was +0.016, with a 95% approximate-independent bootstrap interval of -0.001 to +0.031. The holdouts overlap by 3,252 records, 12.6% of the random test set and 12.6% of the temporal test set, calculated by `SOURCE_ROW_ID` and cross-checked against cohort index. They are therefore neither paired nor fully independent. The bootstrap resamples the two holdouts separately as an approximation and does not explicitly model covariance induced by this overlap. The interval included zero, so the +0.016 point difference was not clearly larger than test-sample resampling variation; evidence is insufficient to claim more than a possible modest overestimation in this fixed comparison.

These intervals condition on the fixed splits, fitted models and selected hyperparameters. They represent test-sample uncertainty only and do not include variability from retraining or repeating model and hyperparameter selection.

PR-AUC's no-information baseline is approximately the positive prevalence. The random and temporal XGBoost holdouts had prevalences of 0.257 and 0.265, respectively, so their PR-AUC values should not be compared mechanically without that context:

| design | positive_prevalence | pr_auc | pr_auc_absolute_lift | normalized_pr_auc | roc_auc | brier_score |
|---|---|---|---|---|---|---|
| random | 0.257 | 0.658 | 0.401 | 0.539 | 0.856 | 0.128 |
| temporal | 0.265 | 0.642 | 0.377 | 0.513 | 0.840 | 0.136 |

Normalized PR-AUC is shown only as an auxiliary prevalence-relative summary, not as a uniquely accepted primary metric. The +0.016 raw PR-AUC difference is therefore interpreted jointly with prevalence, ROC-AUC, Brier score and the bootstrap interval. PR-AUC 0.642 is an area-under-curve measure, not “64.2% accuracy.”

### RQ2 — Best later-year model

XGBoost had the highest temporal Block B PR-AUC (0.642), followed by Random Forest (0.635) and Logistic Regression (0.629). The margins are small relative to the much larger gain from adding information, so the result supports XGBoost within this prespecified comparison rather than a universal algorithm ranking.

Grouped permutation of each original Block B field on the exact 2022/23–2023/24 temporal test set gave the following five largest mean PR-AUC decreases:

| feature | mean_pr_auc_decrease | std_pr_auc_decrease | permutation_p02_5 | permutation_p97_5 |
|---|---|---|---|---|
| BUILDING_TYPE | 0.109 | 0.003 | 0.103 | 0.114 |
| ITEM_IGNITED | 0.034 | 0.003 | 0.027 | 0.041 |
| ALARM_SYSTEM | 0.031 | 0.003 | 0.025 | 0.036 |
| FIRE_START_LOCATION | 0.026 | 0.002 | 0.023 | 0.029 |
| IGNITION_TO_DISCOVERY | 0.022 | 0.002 | 0.018 | 0.025 |

The `permutation_p02_5`–`permutation_p97_5` range is the 2.5th–97.5th percentile range across 30 random permutations; it describes permutation randomness and is not a 95% confidence interval. This analysis measures the fitted model's dependence on each recorded field, not a causal effect. High importance does not mean that a variable causes greater fire spread. Correlated or overlapping fields can share importance; in particular, `CAUSE_OF_FIRE`, `SOURCE_OF_IGNITION` and `ITEM_IGNITED` may encode overlapping information. Results apply only to this fitted pipeline, feature set and temporal test set, and negative values are retained rather than truncated.

### RQ3 — First-arrival information

For validation-selected families, temporal PR-AUC rose from 0.642 in Block B to 0.982 in Block C, an absolute gain of 0.340. `FIRE_SIZE_ON_ARRIVAL` is temporally prior to final `SPREAD_OF_FIRE`, but it is a highly proximal state variable. The Block C result is therefore first-arrival prognosis, not pre-incident building risk and not evidence of deployability before crews arrive.

## Temporal stability and sensitivity

| test_year | positive_prevalence | pr_auc | pr_auc_absolute_lift | normalized_pr_auc | roc_auc |
|---|---|---|---|---|---|
| 2020/21 | 0.325 | 0.687 | 0.362 | 0.537 | 0.840 |
| 2021/22 | 0.272 | 0.643 | 0.371 | 0.510 | 0.840 |
| 2022/23 | 0.286 | 0.655 | 0.369 | 0.517 | 0.838 |
| 2023/24 | 0.244 | 0.628 | 0.384 | 0.508 | 0.842 |

Across these 4 later-year folds, raw PR-AUC ranged from 0.628 to 0.687 (range 0.060) and had the same rank ordering as prevalence (Spearman 1.000). In contrast, ROC-AUC varied by only 0.004, PR-AUC absolute lift by 0.022, and normalized PR-AUC by 0.029. This pattern supports relatively stable later-year discrimination and shows that the apparent raw PR-AUC decline substantially tracks the changing prevalence baseline. With only four annual folds, it does not establish that prevalence explains all variation or identify why prevalence changed.

Expanding-window F1, precision, recall and balanced accuracy use a fixed descriptive threshold of 0.5 and are not directly comparable with the main table's validation-F1 operating point. The 2020/21–2021/22 validation window overlaps the COVID-disrupted period, and 2020/21 has the highest expanding-window prevalence (0.325); this may affect selected settings and thresholds. No policy or COVID attribution is made.

| analysis | test_period | n | positive_prevalence | pr_auc | roc_auc | f1 |
|---|---|---|---|---|---|---|
| main_temporal_definition | 2022/23-2023/24 | 25814 | 0.265 | 0.642 | 0.840 | 0.638 |
| roofs_roof_spaces_positive | 2022/23-2023/24 | 26577 | 0.286 | 0.659 | 0.840 | 0.653 |
| include_late_calls | 2022/23-2023/24 | 26108 | 0.263 | 0.640 | 0.840 | 0.639 |
| include_2024_25_exclude_suffolk | 2024/25 | 12553 | 0.241 | 0.630 | 0.846 | 0.623 |

The official FIRE0304-aligned roof-positive definition increased temporal Block B PR-AUC to 0.659. Reintroducing late calls produced 0.640. A separate 2024/25 check excluding Suffolk produced 0.630; it remains secondary because the main time window was locked before modelling. Across three prespecified random seeds, PR-AUC ranged from 0.651 to 0.658.

## Limitations

- The public file has no incident identifier or exact date/month, limiting dependence checks and finer temporal validation.
- Incident fields may reflect officer judgement; cause/ignition fields may be revised after investigation, and delay fields may be estimated.
- Block B is retrospective and not strictly dispatch-time information.
- Block C's exceptional performance is dominated by proximity to the final outcome and must remain a separate prognostic scenario.
- PR-AUC is prevalence-sensitive; cross-split and subgroup comparisons require their respective positive prevalences.
- Hyperparameters and analytical thresholds were selected using 2020/21–2021/22, a validation window that overlaps the COVID-disrupted period and includes an unusually high-prevalence first year.
- Validation-selected thresholds were transferred to models refitted on train+validation; any probability shift makes threshold-dependent test metrics descriptive rather than re-optimised operating points.
- Subgroup and permutation results are descriptive model diagnostics, not evidence of differential or variable-level causal effects.
- Temporal performance differences do not by themselves identify why distributions changed.
- The publisher URL can be replaced in future. Checksums verify retained files but cannot recover them; the ODS and reproducibility Parquet files require a separate durable institutional deposit.

## Reproducibility

All tables, figures, fitted selected pipelines, split assignments, within-run pre/post-test records, software versions and exact method decisions are saved under `outputs/` and `reports/`. From an existing ODS or Parquet cache, run `python scripts/06_build_report.py` after installing the pinned project environment.
