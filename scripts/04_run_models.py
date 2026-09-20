from fireml.modelling import run_core_models


if __name__ == "__main__":
    result = run_core_models()
    print(result["performance"][["design", "block", "model", "pr_auc", "roc_auc", "f1"]].to_string(index=False))

