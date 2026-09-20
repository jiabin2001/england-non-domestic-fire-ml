"""Render reports and figures from saved analysis artifacts without fitting models.

Recomputes confusion/calibration and table summaries from saved predictions/results.
Does not acquire data, reconstruct a cohort, run bootstrap/permutation analysis, or
rewrite historical training/source provenance. Missing inputs must be restored.
"""

import argparse

from fireml.reporting import build_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        build_report()
    except (FileNotFoundError, ValueError) as error:
        parser.exit(status=1, message=f"{error}\n")
    print(
        "Reports, figures and descriptive summaries rendered from saved artifacts. "
        "No data acquisition or model training was run. "
        "See reports/final_analysis_report.md and outputs/metrics/report_environment.json."
    )
