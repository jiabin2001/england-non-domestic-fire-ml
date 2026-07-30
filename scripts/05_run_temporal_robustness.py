from fireml.robustness import run_temporal_robustness


if __name__ == "__main__":
    results = run_temporal_robustness()
    for name, frame in results.items():
        print(f"\n{name}\n{frame.to_string(index=False)}")

