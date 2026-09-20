"""Run four fixed exploratory diagnostic fits and/or saved-prediction intervals."""

import argparse

from fireml.review_experiments import run_review_fits, run_review_uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--fit-only", action="store_true", help="Run exactly four fits, no bootstrap.")
    phase.add_argument("--uncertainty-only", action="store_true", help="Read completed predictions; never refit.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted fit run after verifying inputs and completed artifacts.")
    parser.add_argument("--overwrite-uncertainty", action="store_true", help="Explicitly replace only the new review interval outputs.")
    args = parser.parse_args()
    if args.resume and args.uncertainty_only:
        parser.error("--resume applies only to the fit phase.")
    if args.overwrite_uncertainty and args.fit_only:
        parser.error("--overwrite-uncertainty requires the uncertainty phase.")
    if not args.uncertainty_only:
        output = run_review_fits(resume=args.resume)
    if not args.fit_only:
        output = run_review_uncertainty(overwrite=args.overwrite_uncertainty)
    print(f"Current diagnostic experiments saved to {output}.")
