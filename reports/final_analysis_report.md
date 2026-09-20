# Final analysis report

> Documentation correction without a research rerun: the numerical values below are retained from the artifacts in reviewed snapshot `c67f6fcb52834abe2695bfc05ba5bcf6a9e40498`. That snapshot is not established as the training commit. The interpretation of Block C, occupancy information and annual validation is corrected here; no new fit, ablation, statistical estimate or figure has been produced. See [no-rerun revision note](no_rerun_revision.md).

## Study definition

This retrospective prediction study uses the official **Other Building Fires Dataset** for England. “Non-domestic building fires” is the dissertation's analytical wording for the official *other building fires* category. The category includes commercial, industrial, public and institutional buildings and can include hotels, hostels, care homes and student halls; it is not limited to buildings without accommodation functions.

The analysis predicts incident-level final fire spread among already-recorded primary fires. It is not a causal study, annual building fire-risk model, fire-physics simulation or real-time FRS deployment tool. Block B contains retrospectively recorded incident information; Block C combines retrospective incident information with arrival-state information. Because C inherits B's investigation fields, its inputs are not established as available at first arrival.

## Data and cohort

The official ODS was updated 22 July 2026. The raw-file SHA-256 is `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`. The audit found 245,207 logical incident rows across 2010/11–2025/26. The defined main period is 2010/11–2023/24. Excluded years are 2024/25: Suffolk FRS did not submit all incidents from September 2024 to March 2025 before the data cut; 2025/26: IRS-to-FaRDaP collection transition from November 2025 may introduce a system discontinuity.

After year restriction, 20 exact duplicates, 3,063 late calls and 6,081 roof-space or otherwise unmappable target records were removed sequentially. The main cohort contains 209,171 incidents, 53,804 larger fires (25.7%).

The main target follows the unambiguous room→floor→whole-building ordering and excludes `Roofs/ Roof spaces`, which cannot be placed uniquely on that scale. Official sources are inconsistent: the current Fire statistics definitions omit roofs from the larger-fire list, while FIRE0304-linked detailed releases include them. The roof-positive sensitivity therefore tests the alternative published convention rather than treating either source as uniquely authoritative.

## Validation and modelling

Temporal train/validation/test years are 2010/11–2019/20, 2020/21–2021/22 and 2022/23–2023/24. The stratified random comparator has exactly the same 159,533/23,824/25,814 sample sizes. All imputing and encoding were pipeline-fitted on training data only. Compact hyperparameter and family selection used validation average precision (AP), calculated with scikit-learn's non-interpolated `average_precision_score`; analytical classification thresholds maximised validation F1. The selected train-fitted pipeline and its validation-derived threshold were then evaluated once on the corresponding holdout, with no train+validation refit or test-set retuning.

Random Forest and XGBoost selected identical hyperparameters under random and temporal development designs. Selected settings differed for Logistic Regression: random `{"C": 1.0}` versus temporal `{"C": 0.1}`. The primary XGBoost RQ1 contrast therefore uses the same selected configuration in both designs.

The original model-selection period also supplies the first two expanding-window evaluation years. Their model families and hyperparameters were therefore selected using labels from those years, despite each annual fit using earlier training records. Results for 2020/21–2021/22 are descriptive development-period analyses; only the 2022/23–2023/24 annual rows follow the selection period. This correction does not change the primary train/validation/test boundary.

The main model comparison is Block B, the retrospective incident-information model:

| model | n | positive_count | positive_prevalence | average_precision | roc_auc | f1 | brier_score |
|---|---|---|---|---|---|---|---|
| XGBoost | 25814 | 6850 | 0.265 | 0.640 | 0.840 | 0.638 | 0.137 |
| Random Forest | 25814 | 6850 | 0.265 | 0.634 | 0.835 | 0.632 | 0.138 |
| Logistic Regression | 25814 | 6850 | 0.265 | 0.628 | 0.831 | 0.630 | 0.140 |

## Research questions

### RQ1 — Random versus temporal validation

For the validation-selected Block B XGBoost, random holdout AP was 0.657 (95% bootstrap CI 0.645–0.668) and temporal holdout AP was 0.640 (0.629–0.652). The random-minus-temporal point difference was +0.016, with a 95% partially paired bootstrap interval of +0.0013 to +0.0316. The holdouts overlap by 3,252 records (12.6% of each holdout); those records were resampled jointly, while design-specific records were resampled independently within outcome and membership strata. The interval lay entirely above zero.

Across-assignment stability was assessed using 20 consecutive split seeds. The original three were retained, and the 17 additions were fixed as a set before their results were inspected. With the estimator seed held fixed, they produced a median random-minus-temporal AP difference of +0.014 (IQR +0.009 to +0.017; range -0.001 to +0.028); 19 of 20 differences were positive. The median supports a small typical random-split optimism effect in this retrospective Block B task, but the direction was not uniform across assignments.

The fixed-split bootstrap interval and across-split point range address different uncertainty sources. The former excludes zero only conditional on the primary splits, fitted models and selected settings; the negative minimum across the alternative assignments shows that the sign is not invariant to split assignment. Their widths were similar (approximately 0.030 and 0.029), but the quantities are dependent and neither is a joint interval or a measure of total uncertainty across test sampling and split assignment.

For the primary split, all three Block B model families favoured random splitting in AP (Logistic Regression +0.011, Random Forest +0.018, XGBoost +0.016), and the corresponding ROC-AUC differences were also positive. The exact magnitude of random-split optimism varies with the split and should not be treated as universal or operationally important without a decision-specific cost analysis. The repeated splits are an empirical sensitivity analysis under fixed model settings, not a second bootstrap interval or 20 independent datasets.

The direction is not universal across information blocks, even for the same XGBoost family:

| block | random_ap | temporal_ap | ap_difference | absolute_lift_difference | normalized_ap_difference | roc_auc_difference |
|---|---|---|---|---|---|---|
| A | 0.565 | 0.560 | 0.005 | 0.013 | 0.014 | 0.013 |
| B | 0.657 | 0.640 | 0.016 | 0.025 | 0.027 | 0.015 |
| C | 0.974 | 0.982 | -0.008 | -0.000 | -0.011 | -0.002 |

Block A showed a smaller random advantage (+0.005); Block C reversed in raw AP (-0.008). For Block C, the absolute AP-lift difference after subtracting each holdout's prevalence was effectively zero (-0.0000), while normalized AP slightly favoured the temporal holdout (-0.011). This confines the inferentially supported positive finding to the retrospective Block B specification and shows that split effects depend on the information set. It does not by itself establish that temporal stability of any particular field caused the pattern.

The expanding-window models trained on all years available before each evaluation year achieved a sample-size-weighted mean annual AP of 0.642 in 2022/23–2023/24, compared with 0.640 for the main train-through-2019/20 model on the combined two-year holdout. The +0.001 gap is a descriptive comparison: the annual models use different training sets and a weighted mean of annual AP is not the pooled two-year AP, so it does not isolate the effect of training recency.

AP's no-information baseline is approximately the positive prevalence. The random and temporal XGBoost holdouts had prevalences of 0.257 and 0.265, respectively, so their AP values are interpreted with prevalence context:

| design | positive_prevalence | average_precision | ap_absolute_lift | normalized_ap | roc_auc | brier_score |
|---|---|---|---|---|---|---|
| random | 0.257 | 0.657 | 0.399 | 0.538 | 0.855 | 0.129 |
| temporal | 0.265 | 0.640 | 0.375 | 0.510 | 0.840 | 0.137 |

Normalized AP is an auxiliary prevalence-relative summary, not a replacement primary metric. AP 0.640 is a ranking summary, not “64.0% accuracy.”

### RQ2 — Best later-year model

XGBoost had the highest temporal Block B AP (0.640), followed by Random Forest (0.634) and Logistic Regression (0.628). The margins are small and no pairwise model-difference interval was estimated, so XGBoost is described only as the highest-performing evaluated family.

Grouped permutation of each original Block B field on the exact 2022/23–2023/24 temporal test set gave the following five largest mean AP decreases:

| feature | mean_pr_auc_decrease | std_pr_auc_decrease |
|---|---|---|
| BUILDING_TYPE | 0.112 | 0.003 |
| ITEM_IGNITED | 0.033 | 0.003 |
| ALARM_SYSTEM | 0.031 | 0.003 |
| FIRE_START_LOCATION | 0.027 | 0.002 |
| IGNITION_TO_DISCOVERY | 0.021 | 0.002 |

With only 30 permutations, the table reports the mean and sample standard deviation; empirical 2.5th and 97.5th percentiles are too coarsely resolved to interpret. This analysis measures the fitted model's dependence on each recorded field, not a causal effect. High importance does not mean that a variable causes greater fire spread. Correlated or overlapping fields can share importance; in particular, `CAUSE_OF_FIRE`, `SOURCE_OF_IGNITION` and `ITEM_IGNITED` may encode overlapping information. Results apply only to this fitted pipeline, feature set and temporal test set, and negative values are retained rather than truncated.

### Structural/context information

On the same temporal holdout, Block A's 5 structural/context fields achieved AP 0.560, an absolute lift of 0.294 above prevalence. That is 78.5% of Block B's 0.375 lift using 16 fields. This is a descriptive nested-block comparison, not an operational-utility estimate, because some Block A fields are retrospectively recorded.

### RQ3 — Arrival-state information added to retrospective incident records

For validation-selected families, temporal AP rose from 0.640 in Block B to 0.982 in Block C, an absolute gain of 0.342. `FIRE_SIZE_ON_ARRIVAL` is temporally prior to final `SPREAD_OF_FIRE`, but it is a highly proximal state variable. C also retains investigation fields such as `CAUSE_OF_FIRE`, `SOURCE_OF_IGNITION` and `ITEM_IGNITED`, whose recorded values may be revised after arrival. This comparison adds arrival-state information to retrospective incident information; it does not demonstrate prediction from information actually available to crews at first arrival. A feature set restricted to information available at arrival and an arrival-state-only baseline have not been evaluated.

Both B and C retain `OCCUPIED_TIME`. The [official field guidance](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance#variable-by-variable---situation) counts people in buildings to which a fire has spread, so this field may encode spread that has already occurred. No removal ablation was run, and neither its performance impact nor the size of any outcome-proxy bias is known. Permutation importance cannot quantify that bias.

## Temporal stability and sensitivity

| test_year | positive_prevalence | average_precision | ap_absolute_lift | normalized_ap | roc_auc |
|---|---|---|---|---|---|
| 2020/21 | 0.325 | 0.687 | 0.362 | 0.537 | 0.840 |
| 2021/22 | 0.272 | 0.643 | 0.371 | 0.510 | 0.840 |
| 2022/23 | 0.286 | 0.655 | 0.369 | 0.517 | 0.838 |
| 2023/24 | 0.244 | 0.628 | 0.384 | 0.508 | 0.842 |

Across these 4 annual rows, AP ranged from 0.628 to 0.687 and had the same rank ordering as prevalence (Spearman 1.000). ROC-AUC varied by only 0.004, AP absolute lift by 0.022, and normalized AP by 0.029. These are descriptive summaries across two development-period rows (2020/21–2021/22) and two later-year evaluations (2022/23–2023/24), not evidence from four independent test folds. They show how the recorded metrics accompany prevalence changes; they neither establish four-fold independent temporal stability nor identify why prevalence changed.

Expanding-window F1, precision, recall and balanced accuracy use a fixed descriptive threshold of 0.5 and are not directly comparable with the main table's validation-F1 operating point. The 2020/21–2021/22 validation window overlaps the COVID-disrupted period, and 2020/21 has the highest expanding-window prevalence (0.325); this may affect selected settings and thresholds. No policy or COVID attribution is made.

| analysis | test_period | n | positive_prevalence | average_precision | ap_absolute_lift | normalized_ap | roc_auc | f1 |
|---|---|---|---|---|---|---|---|---|
| main_temporal_definition | 2022/23-2023/24 | 25814 | 0.265 | 0.640 | 0.375 | 0.510 | 0.840 | 0.638 |
| roofs_roof_spaces_positive | 2022/23-2023/24 | 26577 | 0.286 | 0.658 | 0.371 | 0.520 | 0.840 | 0.653 |
| include_late_calls | 2022/23-2023/24 | 26108 | 0.263 | 0.639 | 0.375 | 0.510 | 0.840 | 0.638 |
| include_2024_25_exclude_suffolk | 2024/25 | 12553 | 0.241 | 0.630 | 0.390 | 0.513 | 0.846 | 0.623 |

Across target, cohort and new-year checks, normalized AP ranged only from 0.510 to 0.520. The roof-positive definition had higher raw AP but slightly lower absolute lift (0.371) than the main definition (0.375); it should not be read as unambiguously better performance. Across 20 split assignments with a fixed estimator seed, random-holdout AP had median 0.654 (IQR 0.649–0.657) and ranged from 0.639 to 0.668.

Building-type subgroup AP ranged from 0.051 for Prison (prevalence 0.034) to 0.731 for Shed / Garage / Greenhouse / Summer house (0.592). At the single global validation-F1 threshold, the largest retained subgroup with zero recall was Prison, with 96 positives among 2,852 incidents. This is evidence that the global analytical threshold does not transfer uniformly across prevalence-defined subgroups; it is not evidence that building type causes fire spread or that the remaining fields lack within-group signal.

## Limitations

- The public file has no incident identifier or exact date/month, limiting dependence checks and finer temporal validation.
- Incident fields may reflect officer judgement; cause/ignition fields may be revised after investigation, and delay fields may be estimated.
- Block B is retrospective and not strictly dispatch-time information.
- Block C combines retrospective incident information with arrival-state information. Its proximity to the final outcome and inherited investigation fields prevent interpreting the reported performance as an arrival-time deployment result.
- `OCCUPIED_TIME` may contain already-observed spread information; it remains in the saved B/C models, and its removal has not been evaluated.
- Average precision is prevalence-sensitive; cross-split and subgroup comparisons require their respective positive prevalences.
- Hyperparameters and analytical thresholds were selected using 2020/21–2021/22, a validation window that overlaps the COVID-disrupted period and includes an unusually high-prevalence first year.
- Expanding-window results for those same selection years are descriptive development-period results. The four annual rows must not be presented as four independent temporal test folds.
- Bootstrap intervals condition on the fixed splits, fitted models and selected settings; they do not represent repeated end-to-end model-selection uncertainty.
- `FRS_TERRITORY` is an available pre-incident geographic field excluded by scope rather than outcome leakage; no territory-inclusive sensitivity was run, so its incremental predictive value is unknown.
- Subgroup and permutation results are descriptive model diagnostics, not evidence of differential or variable-level causal effects.
- Temporal performance differences do not by themselves identify why distributions changed.
- Future changes prompted by this review and evaluated on the already-inspected 2022/23–2023/24 holdout must be reported as post-review exploratory analyses. The 2024/25 sensitivity results have also already been inspected; that period is not a new untouched test set.
- The publisher URL can be replaced in future. Checksums verify retained files but cannot recover them; the ODS and reproducibility Parquet files require a separate durable institutional deposit.

## Reproducibility

Saved tables, figures, selected pipelines, split assignments, model-selection settings and historical environment records are under `outputs/` and `reports/`. They were not regenerated for this correction. The unchanged figure captions and `outputs/tables/feature_policy.csv` may retain earlier wording; read them with the corrections above and the [revision note](no_rerun_revision.md).

`python scripts/06_build_report.py` is a full research rerun that needs separately supplied historical source data and the recorded CUDA environment. `python scripts/07_render_report.py` instead reads existing saved outputs and recalculates reporting summaries and figures without model fitting or data acquisition. Neither command was run for this revision. The saved snapshot and software receipt do not establish an immutable link to the original training commit; missing provenance is not reconstructed from the current checkout.
