# Revision without rerunning the study

## Status and artifact boundary

This revision corrects code reliability and research interpretation without re-importing source data, reconstructing the cohort, fitting models, recalculating research metrics or regenerating figures. It does not constitute an independent replication or show that corrected code reproduces historical performance.

The reviewed repository snapshot is [`c67f6fcb52834abe2695bfc05ba5bcf6a9e40498`](https://github.com/jiabin2001/england-non-domestic-fire-ml/tree/c67f6fcb52834abe2695bfc05ba5bcf6a9e40498). It contains the historical artifacts under review; it is **not established as the commit that trained the models**. No training-commit attribution, historical environment or data receipt has been reconstructed from the current checkout.

The saved numerical tables, selected-model files, per-record predictions, figures and historical JSON receipts under `outputs/` remain unchanged. Markdown reports have been edited directly to clarify interpretation while retaining their historical numerical values. Tests of synthetic fixtures or saved-artifact consistency are software checks, not a rerun of the research experiment.

## Verification of this revision

- Local default test suite: `python -m pytest -q` returned **54 passed, 14 deselected** under Python 3.12.14 with the project's pinned dependencies. The deselected checks require saved study artifacts or external research data; they were not run for this revision.
- Synthetic and mocked tests cover cache-only loading, required-field failures, configurable year windows, report interpretation and preservation of historical receipts. The final-report template is exercised only with synthetic temporary inputs. No research model was fitted or loaded for these tests.
- Before/after SHA-256 comparisons matched for all **74** protected files: every pre-existing file under `outputs/`, `data/raw/source_metadata.json`, and `config/analysis.yaml`. No files were added under `outputs/`.
- All original numeric table rows in the Markdown reports were retained. Python syntax checks and `git diff --check` passed.
- The added GitHub Actions workflow runs the data-independent unit suite on a CPU runner. This local validation does not assert that a remote Actions run has completed.

## Corrected interpretation

### Block C and information timing

Block C is **retrospective incident information plus arrival-state information**. It extends B and therefore includes cause, ignition and other investigation information that may be established or revised after crews arrive. The saved C result cannot establish prediction using only information available at first arrival. Its reported performance and feature list are retained; no arrival-time-only model has been fitted.

`FIRE_SIZE_ON_ARRIVAL` is also a state measurement close to the final-spread outcome. The high C score does not demonstrate incremental value over an arrival-state-only baseline, which has not been evaluated.

### Occupancy as a potential outcome proxy

The [official guidance for `OCCUPIED_TIME`](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance#variable-by-variable---situation) includes people in buildings to which a fire has spread. The field may therefore contain already-observed spread information rather than only the initial occupancy of the building of origin.

The field remains in the saved B/C models and their configuration. Flagging this risk does not remove it from those models. No field-removal ablation has been performed, and no estimate of its effect on AP or of leakage-related bias is available. Permutation importance measures dependence of the fitted model; it is not a substitute for refitting without the field.

### Annual evaluation and model selection

Temporal family and hyperparameter selection used 2020/21–2021/22. The expanding-window analysis reused those settings for annual evaluations starting in 2020/21, so the first two annual rows already contributed labels to model selection even though each annual model was fitted on earlier records.

| Annual rows | Correct interpretation |
|---|---|
| 2020/21–2021/22 | Descriptive development-period analyses; not independent test evaluations |
| 2022/23–2023/24 | Later-year evaluations after the original selection period |

The four historical rows and their numerical summaries are retained, but they must not be presented as four independent temporal test folds. This correction does not change the primary 2010/11–2019/20 training, 2020/21–2021/22 validation and 2022/23–2023/24 test split. It also does not imply that annual folds with overlapping training histories are statistically independent replicates.

### Historical figures and exported policy table

Existing `outputs/figures/*.png`, `outputs/figures/*.pdf` and `outputs/tables/feature_policy.csv` are preserved historical artifacts. Their captions, Block C labels or field-risk descriptions can retain superseded wording. Read them with this correction and the updated Markdown reports. Their preservation does not certify the earlier interpretation, and code edits do not retroactively change how those artifacts were produced.

## Reliability changes and their scope

The code changes address cached-data loading without an unnecessary raw-file access, strict required-feature checks, configurable temporal window lengths, and separation of rendering-environment records from the saved historical runtime receipt. Reporting templates use result-derived labels and qualified interpretation. Data-independent tests and opt-in artifact/integration checks clarify which verification requires externally supplied data.

These changes improve future runs and reporting. They do not supply the missing historical data archive, reconstruct original training provenance or prove numerical equivalence between the revised code and the saved study. In particular, the preserved runtime receipt is not retrospectively repaired from current package versions.

`python scripts/06_build_report.py` remains the full analysis entry point: it imports or reads source data and reruns modelling, uncertainty, interpretation and robustness stages. Acquisition does not download the ODS automatically. A fresh checkout includes `data/raw/source_metadata.json` but lacks the ODS and raw/cohort Parquet files; checksums and historical `exists` fields in the archive manifest do not make those data files available.

`python scripts/07_render_report.py` reads the committed source metadata and existing saved artifacts and recalculates reporting summaries, tables and figures. It does not require the external ODS or raw/cohort Parquet files, acquire data, fit models or repeat bootstrap/permutation analyses. It preserves the historical runtime and archive manifest, recording the rendering environment separately. Running it is a reporting calculation, distinct from both a full model rerun and a documentation-only change. It was not run for this revision.

## Experiments still needed

The smallest useful follow-up is a fixed-configuration **Block B without `OCCUPIED_TIME`** comparison on the original random and temporal partitions. It should use the same data snapshot, keep the original results separately, save new per-record predictions and compute new metrics and intervals for the revised models. Old confidence intervals cannot be transferred to a newly fitted model.

Additional useful experiments are a building-type-only baseline, an arrival-state-only baseline, and an arrival-time feature whitelist if an operational arrival-time claim is retained. Four independently selected rolling evaluations would require model selection using only data available before each evaluation year. None of these experiments is part of this revision.

The repository saves per-record predictions for the selected XGBoost models but only aggregate performance for Logistic Regression and Random Forest. A paired XGBoost–Logistic Regression AP-difference interval cannot be reconstructed from the aggregate AP values; it requires corresponding per-record predictions, obtained from a retained model or a new fit.

The existing 2022/23–2023/24 holdout and the 2024/25 sensitivity period have already been inspected. Review-driven experiments using them must be reported as post-review exploratory analyses. Reusing the dates does not restore an untouched test set. A later confirmatory evaluation would require an uninspected, suitable period or independent dataset and a protocol fixed before its results are examined.
