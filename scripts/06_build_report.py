"""Rebuild the complete analysis from the cached ODS or Parquet data."""

from fireml.acquire import acquire_and_cache
from fireml.audit import run_audit
from fireml.cohort import construct_cohort
from fireml.interpretability import run_grouped_permutation_analysis
from fireml.modelling import run_core_models
from fireml.reporting import build_report
from fireml.robustness import run_temporal_robustness
from fireml.uncertainty import run_uncertainty_analysis


if __name__ == "__main__":
    acquire_and_cache()
    run_audit()
    construct_cohort()
    run_core_models()
    run_uncertainty_analysis()
    run_grouped_permutation_analysis()
    run_temporal_robustness()
    build_report()
    print("Complete analysis rebuilt. See reports/final_analysis_report.md.")
