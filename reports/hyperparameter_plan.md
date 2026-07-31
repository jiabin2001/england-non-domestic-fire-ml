# Hyperparameter plan

## Status and selection boundary

Candidate configurations are deliberately compact and are selected only by validation-set average precision, calculated with scikit-learn's `average_precision_score`. Holdout performance does not alter the candidate set. The selected configuration is a practical comparison setting, not a claim of theoretical optimality.

XGBoost runtime device: `cuda` (`hist` tree method). CUDA is used only when `nvidia-smi` confirms an available GPU; otherwise execution falls back to CPU.

## Baseline observations

| design | model | validation average precision | fit seconds |
|---|---|---:|---:|
| temporal | dummy | 0.2974 | 0.96 |
| temporal | logistic_regression | 0.6467 | 1.38 |
| temporal | random_forest | 0.6267 | 70.70 |
| temporal | xgboost | 0.6637 | 1.27 |
| random | dummy | 0.2572 | 0.91 |
| random | logistic_regression | 0.6312 | 1.25 |
| random | random_forest | 0.6117 | 73.73 |
| random | xgboost | 0.6473 | 1.22 |

## Logistic Regression

- `C` controls inverse L2 regularisation strength. scikit-learn's default is 1.0.
- Candidates: 0.1, 1.0 and 10.0. They cover stronger, default and weaker regularisation on a log scale without changing the algorithm or feature information.
- `penalty='l2'`, `solver='lbfgs'` and `max_iter=500` remain fixed. The larger iteration limit is convergence control, not model-complexity tuning.
- This compact range is fair because it gives the linear model meaningful regularisation flexibility while avoiding a much larger search budget than the tree models.

## Random Forest

- `n_estimators` controls Monte Carlo stability; software default is 100. Candidates use 100–300 trees.
- `max_depth` controls tree depth; default is unlimited (`None`). One capped value (20) tests a materially simpler forest.
- `min_samples_leaf` controls terminal-node regularisation; default is 1. Values 1 and 5 compare default with smoother leaves.
- `max_features` remains at the software classification default, `sqrt`, so feature subsampling is not separately optimised.
- Four joint configurations test default, more trees, larger leaves and capped depth. More configurations could favour this family merely by search volume, so the budget is intentionally limited.

## XGBoost

- The scikit-learn wrapper exposes `None` for many constructor defaults; the default-equivalent explicit configuration used here is 100 trees, learning rate 0.3, depth 6, minimum child weight 1, and full row/column sampling.
- `n_estimators` and `learning_rate` control boosting length and shrinkage. Configurations compare 100×0.3 with 200–300×0.1.
- `max_depth` controls interaction complexity; depth 3 is compared with the default-equivalent depth 6.
- `min_child_weight` regularises small child nodes; 1 and 5 are compared.
- `subsample` and `colsample_bytree` are 1.0 by default; a single 0.8/0.8 configuration checks moderate stochastic regularisation.
- Four joint configurations keep XGBoost's search budget equal to Random Forest's and avoid a large optimisation exercise.

## Candidate configurations

```json
{
  "logistic_regression": [
    {
      "C": 0.1
    },
    {
      "C": 1.0
    },
    {
      "C": 10.0
    }
  ],
  "random_forest": [
    {
      "n_estimators": 100,
      "max_depth": null,
      "min_samples_leaf": 1,
      "max_features": "sqrt"
    },
    {
      "n_estimators": 300,
      "max_depth": null,
      "min_samples_leaf": 1,
      "max_features": "sqrt"
    },
    {
      "n_estimators": 200,
      "max_depth": null,
      "min_samples_leaf": 5,
      "max_features": "sqrt"
    },
    {
      "n_estimators": 200,
      "max_depth": 20,
      "min_samples_leaf": 1,
      "max_features": "sqrt"
    }
  ],
  "xgboost": [
    {
      "n_estimators": 100,
      "learning_rate": 0.3,
      "max_depth": 6,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0
    },
    {
      "n_estimators": 300,
      "learning_rate": 0.1,
      "max_depth": 6,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0
    },
    {
      "n_estimators": 200,
      "learning_rate": 0.1,
      "max_depth": 3,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0
    },
    {
      "n_estimators": 200,
      "learning_rate": 0.1,
      "max_depth": 6,
      "min_child_weight": 5,
      "subsample": 0.8,
      "colsample_bytree": 0.8
    }
  ]
}
```

## Stability rule

All candidate validation AP values are retained. The selected configuration's margin over adjacent candidates is reported in `hyperparameter_stability_summary.csv`. Expanding-window and sensitivity analyses reuse the selected configuration.
