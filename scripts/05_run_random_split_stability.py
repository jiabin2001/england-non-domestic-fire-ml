from fireml.robustness import run_random_split_stability


if __name__ == "__main__":
    stability = run_random_split_stability()
    print(f"Saved {len(stability)} configured split-seed results.")
