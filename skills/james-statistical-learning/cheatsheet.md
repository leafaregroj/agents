# Cheatsheet — ISLP Decision Rules

## Choosing a method
| Situation | Reach for | Why |
|---|---|---|
| Inference / interpretability matters | Linear or logistic regression, lasso, GAM, single pruned tree | Restrictive models are explainable |
| Prediction only, tabular, moderate n | Boosting, random forest, BART | Strong accuracy, interactions handled |
| p large relative to n | Lasso, ridge, PCR; naive Bayes for classification | Variance control |
| Truth sparse vs. dense | Lasso vs. ridge | Matches the penalty to the signal |
| Linear boundary | Logistic regression or LDA (LDA if roughly Gaussian or classes well separated) | Low variance |
| Moderately non-linear boundary | QDA, naive Bayes, polynomial logistic | Mid flexibility |
| Highly non-linear, low p | KNN (K by CV), radial SVM | Flexible |
| Images, text, sequences, huge n | CNN, RNN/LSTM, transfer learning | Structured inputs |
| Time-to-event with censoring | Kaplan–Meier, log-rank, Cox PH | Never drop censored rows |
| No response | PCA, K-means, hierarchical clustering | Exploration |
| Many simultaneous tests | Holm (FWER) or BH (FDR) | Avoid false discoveries |

## Defaults & thresholds
- CV: k = 5 or 10, the same folds for all models, and the one-SE rule for simplicity. Bootstrap: B ≈ 1,000. Permutation: B ≈ 10,000.
- Random forest: m ≈ √p for classification, p/3 for regression. Bagging and random forests don't overfit as B grows.
- Boosting: λ 0.01–0.001, d = 1 (stumps) to 4, B by CV (it *can* overfit).
- BART: K = 200, B = 1,000, burn-in L = 100.
- Polynomial degree ≤ 4. Spline knots at quantiles. Prefer natural splines.
- K-means: n_init 20–50, fixed random_state.
- Linear regression diagnostics:
  - Outlier: |studentized residual| > 3.
  - High leverage: hᵢ ≫ (p+1)/n.
  - Collinearity: VIF > 5–10.
  - 95% CI ≈ β̂ ± 2·SE.
- Uncorrected FWER = 1 − (1−α)^m (0.994 at m = 100, α = 0.05).

## Decision rules
- **Never select or assess on training error.** Test MSE is U-shaped in flexibility; use CV.
- **Standardize** before ridge, lasso, KNN, PCA, K-means or SVM, unless the variables share units.
- **Tune → refit**: choose λ, M, K, C or α by CV, then refit on all the data.
- **CV curve**: trust where its minimum is, not the absolute error level.
- **Many predictors**: check the overall F-test before individual p-values.
- **Interaction in the model**: keep the main effects (hierarchical principle).
- **Heteroscedastic residuals (funnel)**: transform Y (log, √) or use weighted least squares.
- **Imbalanced classes**:
  - Compare against the null classifier.
  - Move the threshold according to error costs (Default: 0.5 → 0.2 cut missed defaulters from 75.7% to 41.4%).
  - Report sensitivity and specificity, not just the error rate.
- **Counts**: use Poisson regression, not least squares.
- **Survival**:
  - Covariates matter → use Cox, not log-rank.
  - Check proportional hazards.
  - Evaluate with the C-index.
- **Choosing PCA M**: use the scree-plot elbow, or CV only when the PCs feed a supervised model.
- **Clustering**: try several linkages, K values and subsets, and report only stable clusters.
- **Testing**: always prefer Holm to Bonferroni. Use FDR (BH) when m is large and the work exploratory.
- **SVM in sklearn**: `C` is a cost. A larger sklearn C means a *narrower* margin, the opposite of the book's budget C.
- **Small data**: try simple models before a neural network; they often tie.

## Tells & smells
- Training error ≪ CV error → overfitting; reduce flexibility.
- A coefficient flips sign when a variable is added → confounding or collinearity.
- Residuals curved vs fitted → missing non-linearity; add polynomial or spline terms.
- Residuals correlated in time → SEs are too small.
- One tree changes drastically on resampling → high variance; switch to an ensemble.
- p > n and training R² ≈ 1 → meaningless; Cp/BIC/R² are invalid there.
- Choosing a tuning parameter from training ROC → it will look best and test worst.
- A post hoc best-vs-worst comparison → data snooping; use Tukey or a correction.
