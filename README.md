# Interpretable Heart Disease Risk Scores with FasterRisk

A short exploration applying FasterRisk (Liu et al., NeurIPS 2022) to the UCI
Heart Disease (Cleveland) dataset and comparing it with logistic regression baselines.

## Data
297 patients (6 with missing values removed); 137 with heart disease (num > 0).
The 13 original features were converted into 24 binary features using clinical
thresholds (e.g., resting BP >= 130 / >= 140, cholesterol >= 200 / >= 240) and
one-hot encoding for categorical variables.

## Methods
5-fold stratified cross-validation (random_state=0), comparing:
- Full logistic regression on all 24 features
- Sparse LR: keep the top-k features by |coefficient|, then refit
- Rounded LR: sparse LR with weights scaled so the largest is 5, then rounded
- FasterRisk with k = 3, 5, 7 (parent_size=10)

## Results (mean test AUC ± std over 5 folds)
| Method | k=3 | k=5 | k=7 |
|---|---|---|---|
| Sparse LR | 0.839 ± 0.034 | 0.902 ± 0.023 | 0.906 ± 0.026 |
| Rounded LR | 0.841 ± 0.031 | 0.903 ± 0.020 | 0.903 ± 0.029 |
| FasterRisk | 0.884 ± 0.048 | 0.904 ± 0.032 | 0.910 ± 0.030 |

Full LR (24 features): 0.902 ± 0.034

## Observations
1. A 5-feature integer score matched full logistic regression (0.904 vs 0.902).
2. At k=3, FasterRisk won in all 5 folds. The gap came mainly from folds 1 and 2,
   where the baseline swapped in "Male"; FasterRisk selected the same three
   features in every fold. Naive rounding itself cost almost nothing here,
   likely because all features are binary.
3. The k=3 score assigns 3 points to each feature, so it reduces to counting
   red flags: 0, 1, 2, or 3 flags give 5.5%, 33%, 80.5%, and 97.2% risk.
4. All 19 near-optimal 5-feature models share the same core three features;
   the remaining slots rotate among exercise-test measurements (ST depression,
   ST slope, exercise angina, max heart rate), which appear interchangeable.

## Limitations
Small single dataset; simple baselines (a stronger feature-selection method,
such as an L1 regularization path, might narrow the k=3 gap); the model-pool
analysis uses training AUC. Run on Colab with fasterrisk installed via
`pip install fasterrisk --no-deps` and newer package versions.
