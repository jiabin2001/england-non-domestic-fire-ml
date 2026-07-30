# Final analysis report

## Study definition

This retrospective prediction study uses the official **Other Building Fires Dataset** for England. “Non-domestic building fires” is the dissertation's analytical wording for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings and can include hotels, hostels, care homes and student halls; it is not limited to buildings without accommodation functions.

The analysis predicts incident-level final fire spread among already-recorded primary fires. It is not a causal study, annual building fire-risk model, fire-physics simulation or real-time FRS deployment tool. Block B contains retrospectively recorded incident information; Block C is a separately interpreted first-arrival prognostic model.

## Data and cohort

The official ODS was updated 22 July 2026. The raw-file SHA-256 is `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`. The audit found 245,207 logical incident rows across 2010/11–2025/26. The locked main period is 2010/11–2023/24; 2024/25 is excluded because Suffolk submissions are incomplete for part of that year, and 2025/26 is excluded because it spans the IRS-to-FaRDaP transition.

After year restriction, 20 exact duplicates, 3,063 late calls and 6,081 `Roofs/ Roof spaces` target records were removed sequentially. The main cohort contains 209,171 incidents, 53,804 larger fires (25.7%).

## Validation and modelling

Temporal train/validation/test years are 2010/11–2019/20, 2020/21–2021/22 and 2022/23–2023/24. The stratified random comparator has exactly the same 159,533/23,824/25,814 sample sizes. All imputing and encoding were pipeline-fitted on development data only. Compact hyperparameter selection used validation PR-AUC; analytical classification thresholds maximised validation F1. The temporal holdout was evaluated once after configuration locking.

The main model comparison is Block B, the retrospective incident-information model:

| model | n | positive_count | positive_prevalence | pr_auc | roc_auc | f1 | brier_score |
|---|---|---|---|---|---|---|---|
| XGBoost | 25814 | 6850 | 0.265 | 0.642 | 0.840 | 0.638 | 0.136 |
| Random Forest | 25814 | 6850 | 0.265 | 0.635 | 0.836 | 0.634 | 0.138 |
| Logistic Regression | 25814 | 6850 | 0.265 | 0.629 | 0.831 | 0.631 | 0.139 |

## Research questions

### RQ1 — Random versus temporal validation

For the validation-selected Block B XGBoost, random holdout PR-AUC was 0.658 and temporal holdout PR-AUC was 0.642, a random-minus-temporal difference of +0.016. Logistic Regression and Random Forest also had higher random than temporal Block B PR-AUC. Thus random splitting modestly overestimated later-year discrimination in the core retrospective scenario, although the magnitude depends on information block and should not be generalised to every deployment definition.

### RQ2 — Best later-year model

XGBoost had the highest temporal Block B PR-AUC (0.642), followed by Random Forest (0.635) and Logistic Regression (0.629). The margins are small relative to the much larger gain from adding information, so the result supports XGBoost within this prespecified comparison rather than a universal algorithm ranking.

### RQ3 — First-arrival information

For validation-selected families, temporal PR-AUC rose from 0.642 in Block B to 0.982 in Block C, an absolute gain of 0.340. `FIRE_SIZE_ON_ARRIVAL` is temporally prior to final `SPREAD_OF_FIRE`, but it is a highly proximal state variable. The Block C result is therefore first-arrival prognosis, not pre-incident building risk and not evidence of deployability before crews arrive.

## Temporal stability and sensitivity

Expanding-window annual PR-AUC ranged from 0.628 to 0.687; 2023/24 was 0.628. No policy or COVID attribution is made because this design establishes performance variation, not its cause.

| analysis | test_period | n | positive_prevalence | pr_auc | roc_auc | f1 |
|---|---|---|---|---|---|---|
| main_temporal_definition | 2022/23-2023/24 | 25814 | 0.265 | 0.642 | 0.840 | 0.638 |
| roofs_roof_spaces_positive | 2022/23-2023/24 | 26577 | 0.286 | 0.659 | 0.840 | 0.653 |
| include_late_calls | 2022/23-2023/24 | 26108 | 0.263 | 0.640 | 0.840 | 0.639 |
| include_2024_25_exclude_suffolk | 2024/25 | 12553 | 0.241 | 0.630 | 0.846 | 0.623 |

The mandatory roof-positive definition increased temporal Block B PR-AUC to 0.659. Reintroducing late calls produced 0.640. A separate 2024/25 check excluding Suffolk produced 0.630; it remains secondary because the main time window was locked before modelling. Across three prespecified random seeds, PR-AUC ranged from 0.651 to 0.658.

## Limitations

- The public file has no incident identifier or exact date/month, limiting dependence checks and finer temporal validation.
- Incident fields may reflect officer judgement; cause/ignition fields may be revised after investigation, and delay fields may be estimated.
- Block B is retrospective and not strictly dispatch-time information.
- Block C's exceptional performance is dominated by proximity to the final outcome and must remain a separate prognostic scenario.
- Subgroup PR-AUC is prevalence-sensitive; subgroup comparisons are descriptive and not evidence of differential causal effects.
- Temporal performance differences do not by themselves identify why distributions changed.

## Reproducibility

All tables, figures, fitted selected pipelines, split assignments, lock receipt, software versions and exact method decisions are saved under `outputs/` and `reports/`. From an existing ODS or Parquet cache, run `python scripts/06_build_report.py` after installing the pinned project environment.
