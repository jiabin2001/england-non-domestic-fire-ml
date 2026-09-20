from fireml.cohort import construct_cohort


if __name__ == "__main__":
    cohort, flow = construct_cohort()
    print(flow.to_string(index=False))
    print(f"Final cohort: {len(cohort):,}; larger fires: {cohort['LARGER_FIRE'].sum():,} ({cohort['LARGER_FIRE'].mean():.1%})")

