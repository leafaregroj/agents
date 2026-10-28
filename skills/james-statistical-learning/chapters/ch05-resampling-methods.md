# Chapter 5: Resampling Methods

## Core Idea
Refit a model repeatedly on resampled versions of the training data to learn what a single fit cannot tell you: **cross-validation** estimates test error (for *model assessment* and *model selection*), and **the bootstrap** estimates the sampling variability (standard error) of almost any estimator.

## Frameworks Introduced
- **Validation set approach**: randomly split the n observations into a training set and a validation (hold-out) set; fit on the first, compute MSE (or error rate) on the second.
  - When to use: very large n, or a quick sanity check; expensive models where refitting k times is unaffordable.
  - How: split (e.g. 50/50); fit each candidate; pick the one with lowest validation error; refit on all data.
  - Failure mode: (1) highly variable estimate, since it depends on which observations land in each half; (2) fits on only part of the data, so it **overestimates** test error.
- **Leave-one-out cross-validation (LOOCV)**: for i = 1..n, fit on the n − 1 points excluding i, compute MSEᵢ = (yᵢ − ŷᵢ)²; average.
  - When to use: small n; least squares / polynomial regression (and smoothing splines), where the shortcut formula makes it as cheap as one fit.
  - How: CV₍ₙ₎ = (1/n) Σ MSEᵢ. No randomness in the splits.
  - Why / failure mode: nearly unbiased (each fit uses n − 1 points), but **high variance** because the n fitted models are almost identical, so their errors are highly correlated.
- **k-fold CV**: randomly partition into k folds of ~equal size; each fold in turn is the validation set, fit on the other k − 1 folds; average the k MSEs.
  - When to use: the default for estimating test error and choosing a tuning parameter (degree, λ, K in KNN, tree size…).
  - How: **k = 5 or k = 10**; plot CV error vs flexibility, choose the minimum (or apply the one-standard-error rule, Ch 6). LOOCV is the special case k = n.
  - Why it works: intermediate bias (trains on (k−1)n/k points) with lower variance than LOOCV — the bias-variance trade-off of the CV estimate itself. Even when the CV error *level* is off, the *location of the minimum* is usually close to the true test-error minimum, which is what model selection needs.
- **CV for classification**: identical, replacing MSE with the misclassification rate Err_i = I(yᵢ ≠ ŷᵢ).
- **The bootstrap**: create B data sets by sampling n observations **with replacement** from the original data; recompute the statistic α̂ on each; the standard deviation of the B estimates is the bootstrap SE.
  - When to use: any statistic whose SE has no convenient formula (ratios, portfolio weights, medians, coefficients when the model is misspecified).
  - How: B ≈ 1,000; SE_B(α̂) = sqrt( (1/(B−1)) Σ_r (α̂*ʳ − ᾱ*)² ).
  - Why it works: the empirical distribution of the sample stands in for the unknown population, so resampling it mimics drawing new data sets.

## Key Formulas
- LOOCV estimate: `CV₍ₙ₎ = (1/n) Σᵢ MSEᵢ`.
- LOOCV shortcut for least squares / polynomial regression: `CV₍ₙ₎ = (1/n) Σᵢ ( (yᵢ − ŷᵢ) / (1 − hᵢ) )²`, where ŷᵢ is the full-data fit and hᵢ the leverage (Ch 3); high-leverage residuals are inflated. Requires only one fit.
- k-fold CV: `CV₍ₖ₎ = (1/k) Σⱼ MSEⱼ`.
- Classification LOOCV: `CV₍ₙ₎ = (1/n) Σᵢ I(yᵢ ≠ ŷᵢ)`.
- Minimum-variance portfolio weight (bootstrap example): invest α in X and 1 − α in Y; `α = (σ²_Y − σ_XY) / (σ²_X + σ²_Y − 2σ_XY)`, estimated by plugging in sample variances/covariance.
- Bootstrap SE: `SE_B(α̂) = sqrt( (1/(B−1)) Σ_{r=1..B} (α̂*ʳ − (1/B) Σ_{r'} α̂*ʳ')² )`.

## Key Concepts
- **Model assessment**: estimating a chosen model's test error.
- **Model selection**: choosing the appropriate level of flexibility.
- **Validation set / hold-out set**: data held out from fitting and used only to estimate error.
- **Fold**: one of the k disjoint groups in k-fold CV.
- **Leverage shortcut**: the (1 − hᵢ) correction that turns one least-squares fit into an exact LOOCV.
- **Bootstrap data set**: a size-n sample drawn with replacement; some observations appear multiple times, ~1/3 are absent.
- **Bootstrap replicate**: the statistic recomputed on one bootstrap data set (α̂*ʳ).
- **Sampling with replacement**: the operation that lets the observed data emulate the population.

## Mental Models
- **Think of k as a bias-variance dial for the error estimate**: small k (2, validation set) → pessimistic bias, high variance; k = n → low bias, high variance (correlated fits); k = 5–10 is the empirically good middle.
- **Use CV for the *argmin*, not the absolute number**: CV curves can misstate the error level but usually locate the best flexibility correctly.
- **Use the bootstrap whenever a formula's assumptions are doubtful**: textbook OLS SEs assume a correct model, fixed xᵢ, and σ² estimated from residuals; the bootstrap assumes none of these.
- **Think of the bootstrap as "the sample is the population"**: resample the data, not the population.

## Anti-patterns
- **Trusting a single validation split**: different random splits give visibly different error curves (10 splits of Auto disagree on the level, though they agree quadratic beats linear).
- **Reporting the validation-set error as the final test error**: it is biased upward because only half the data was used to fit.
- **Choosing LOOCV by default for expensive models**: n refits with no shortcut and higher variance than 10-fold.
- **Using the bootstrap naively for prediction error**: the bootstrap samples overlap ~2/3 with the original data, so evaluating on the original set underestimates test error; use CV instead.
- **Selecting predictors on all data and then cross-validating only the final fit**: any step that uses the response must be inside the CV loop (see Ch 6 and exercises).

## Code Examples
```python
from sklearn.model_selection import cross_validate, KFold, ShuffleSplit
from ISLP.models import sklearn_sm, ModelSpec as MS
import statsmodels.api as sm

hp_model = sklearn_sm(sm.OLS, MS(['horsepower']))
X, Y = Auto.drop(columns=['mpg']), Auto['mpg']
cv_results = cross_validate(hp_model, X, Y, cv=Auto.shape[0])  # cv=n -> LOOCV
np.mean(cv_results['test_score'])                               # 24.23

cv = KFold(n_splits=10, shuffle=True, random_state=0)   # same splits for every degree
for i, d in enumerate(range(1, 6)):
    X = np.power.outer(H, np.arange(d+1))
    cv_error[i] = np.mean(cross_validate(M, X, Y, cv=cv)['test_score'])
```
- **What it demonstrates**: `sklearn_sm` wraps a statsmodels model for sklearn's `cross_validate`; integer `cv` gives k-fold (n gives LOOCV), and a fixed `KFold` object makes comparisons across models fair. `ShuffleSplit(n_splits=1, test_size=196)` reproduces the validation set approach.

```python
def boot_SE(func, D, n=None, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index, n, replace=True)
        value = func(D, idx)
        first_ += value; second_ += value**2
    return np.sqrt(second_ / B - (first_ / B)**2)

boot_SE(alpha_func, Portfolio, B=1000, seed=0)   # 0.0912
```
- **What it demonstrates**: a generic bootstrap — any function `func(D, idx)` returning a statistic (α̂, or OLS coefficients via `partial(boot_OLS, model_matrix, 'mpg')`) gets an SE.

## Reference Tables
| Method | Bias of test-error estimate | Variance | Cost |
|---|---|---|---|
| Validation set | High (overestimates) | High (split-dependent) | 1 fit |
| LOOCV | ~Unbiased | High (correlated fits) | n fits (1 for OLS via leverage) |
| k-fold (k = 5/10) | Moderate | Lower | k fits |

| Auto `mpg ~ horsepower` SE | Bootstrap | Formula (`sm.OLS`) |
|---|---|---|
| Linear intercept / slope | 0.85 / 0.0074 | 0.717 / 0.006 |
| Quadratic fit (intercept) | 2.07 | 1.80 |
Linear fit is misspecified, so formula SEs are less trustworthy; with the better quadratic fit the two agree closely.

## Worked Example
**Auto data, mpg vs horsepower (392 obs).** Validation set (196/196): test MSE for linear / quadratic / cubic = 23.62 / 18.76 / 18.80; a different seed gives 20.76 / 16.95 / 16.97. The *level* moves with the split, but the conclusion is stable: quadratic is much better than linear, cubic adds nothing. LOOCV for the linear fit = 24.23; 10-fold CV over degrees 1–5 shows a sharp drop at degree 2 and no clear gain beyond.

**Portfolio bootstrap.** Simulated: 1,000 data sets from the true population (true α = 0.6) give mean α̂ = 0.5996 and SE ≈ 0.083. From one data set, 1,000 bootstrap samples give SE ≈ 0.087, almost the same. In the lab with the Portfolio data (100 obs): α̂ = 0.576, bootstrap SE = 0.091.

## Key Takeaways
1. Training error is the wrong yardstick; use held-out error, preferably k-fold CV with k = 5 or 10.
2. The validation set approach is simple but noisy and pessimistic; LOOCV removes randomness and bias but has high variance and cost.
3. For least squares, LOOCV is free via the leverage formula.
4. CV is mainly for finding the minimum of the error curve; pair it with the one-standard-error rule for parsimony.
5. The bootstrap gives SEs (and CIs) for nearly any statistic by resampling with replacement, B ≈ 1,000.
6. When bootstrap and formula SEs disagree, suspect the formula's model assumptions first.

## Connects To
- **Ch 2**: bias-variance trade-off; training vs test MSE that CV estimates.
- **Ch 3**: leverage hᵢ, formula SEs of β̂ that the bootstrap checks.
- **Ch 6**: CV selects λ (ridge/lasso), subset size, M in PCR/PLS; one-standard-error rule.
- **Ch 7–8**: CV chooses degrees of freedom, knots, tree size; bagging and random forests reuse the bootstrap (out-of-bag error).
- **Ch 13**: resampling/permutation approaches to p-values.
- **External**: nested CV for honest assessment after tuning; bootstrap confidence intervals (percentile, BCa).
