# Chapter 6: Linear Model Selection and Regularization

## Core Idea
Least squares on all p predictors has high variance when p is large relative to n (and is undefined when p > n) and is hard to interpret. Three families improve on it for prediction and interpretability: **subset selection** (fit on a chosen subset), **shrinkage / regularization** (ridge, lasso: shrink coefficients toward zero), and **dimension reduction** (PCR, PLS: regress on M < p linear combinations).

## Frameworks Introduced
- **Best subset selection**: for each k = 0..p, fit all C(p,k) models and keep the lowest-RSS one (M_k); then choose among M₀..M_p with CV, Cp, BIC or adjusted R².
  - When to use: p ≲ 30–40 (2ᵖ models; p = 20 → >1M fits).
  - Failure mode: computationally infeasible for large p; the huge search space can overfit and give high-variance estimates.
- **Forward stepwise selection**: start from the null model, add the one predictor that most reduces RSS at each step; choose among M₀..M_p as above. Fits 1 + p(p+1)/2 models; usable even when p > n (stop at M_{n−1}).
- **Backward stepwise selection**: start from the full model, remove the least useful predictor each step. Requires n > p. Same 1 + p(p+1)/2 count.
  - Failure mode (both stepwise): not guaranteed to find the best model (Credit: best 4-var model swaps out `rating`, forward stepwise cannot). **Hybrid** approaches add and drop.
- **Choosing the final model**: never by RSS/R² (always favor the largest model). Either *indirectly* adjust training error (Cp, AIC, BIC, adjusted R²) or *directly* estimate test error with validation/CV (preferred: fewer assumptions, works when d or σ² is hard to define).
  - **One-standard-error rule**: compute the CV-error SE at each model size; pick the *simplest* model whose estimated test error is within one SE of the minimum.
- **Ridge regression**: minimize `RSS + λ Σⱼ βⱼ²` (intercept not penalized).
  - When to use: many predictors, each with a modest effect; multicollinearity; p close to n.
  - How: **standardize predictors first** (ridge is not scale equivariant); fit over a grid of λ; choose λ by CV; refit on all data with that λ.
  - Why it works: as λ increases, variance falls sharply at a small cost in bias; works even when p > n. Failure mode: keeps all p predictors (no zeros) — poor interpretability.
- **The lasso**: minimize `RSS + λ Σⱼ |βⱼ|`.
  - When to use: you expect only a subset of predictors matters (sparse truth) and want an interpretable model.
  - How: standardize, CV over λ, refit. Yields **sparse models** — some coefficients exactly zero (variable selection).
  - Why it works: the ℓ₁ constraint region (diamond) has corners on the axes, so the RSS ellipse often first touches it at a corner; the ℓ₂ circle has none. Failure mode: when all predictors carry signal, ridge predicts better.
- **Selecting λ**: build a grid, compute CV error per λ, pick the minimizer (or 1-SE), refit on all data.
- **Principal components regression (PCR)**: compute the first M principal components Z₁..Z_M of standardized X (unsupervised), regress y on them by least squares.
  - When to use: the directions of greatest X-variance are also the ones associated with y.
  - How: choose M by CV; standardize first. Not feature selection — each Z is a combination of all p predictors (closely related to ridge, a "continuous" PCR).
- **Partial least squares (PLS)**: supervised dimension reduction; Z₁ = Σ φⱼ₁Xⱼ with φⱼ₁ = simple-regression coefficient of y on Xⱼ; subsequent directions from residuals after orthogonalizing. Choose M by CV. In practice often no better than ridge or PCR — reduces bias but may increase variance.

## Key Formulas
(d = number of predictors in the model, σ̂² = error-variance estimate from the full model, n = observations, TSS = Σ(yᵢ − ȳ)²)
- Mallows's `Cp = (1/n)(RSS + 2dσ̂²)` — penalty grows with d; unbiased for test MSE if σ̂² is unbiased. (Equivalent form: RSS/σ̂² + 2d − n.)
- `AIC = (1/n)(RSS + 2dσ̂²)` for Gaussian linear models (up to constants), so proportional to Cp.
- `BIC = (1/n)(RSS + log(n) d σ̂²)` — log n > 2 for n > 7, so BIC picks **smaller** models than Cp.
- `Adjusted R² = 1 − [RSS/(n − d − 1)] / [TSS/(n − 1)]` — maximize; heuristic, not theory-backed.
- Ridge: `β̂ᴿ_λ = argmin Σᵢ(yᵢ − β₀ − Σⱼ βⱼxᵢⱼ)² + λ Σⱼ βⱼ²`; ℓ₂ norm ‖β‖₂ = sqrt(Σβⱼ²); shrinkage measured by ‖β̂ᴿ_λ‖₂ / ‖β̂‖₂.
- Lasso: `β̂ᴸ_λ = argmin RSS + λ Σⱼ |βⱼ|`; equivalent constrained forms: minimize RSS s.t. Σ|βⱼ| ≤ s (lasso), Σβⱼ² ≤ s (ridge), Σ I(βⱼ ≠ 0) ≤ s (best subset).
- Standardization: `x̃ᵢⱼ = xᵢⱼ / sqrt((1/n) Σᵢ (xᵢⱼ − x̄ⱼ)²)`.
- Special case (n = p, X = I, no intercept): ridge `β̂ⱼ = yⱼ/(1 + λ)` (proportional shrinkage); lasso **soft-thresholding** `β̂ⱼ = yⱼ − λ/2 if yⱼ > λ/2; yⱼ + λ/2 if yⱼ < −λ/2; 0 if |yⱼ| ≤ λ/2`.
- Bayesian view: ridge = posterior mode (and mean) under a Gaussian prior on β; lasso = posterior mode under a double-exponential (Laplace) prior.
- Dimension reduction: `Z_m = Σⱼ φⱼₘ Xⱼ`, fit `yᵢ = θ₀ + Σₘ θₘ zᵢₘ + εᵢ`; implies `βⱼ = Σₘ θₘ φⱼₘ` (a constraint on β). First PC: loadings maximize Var(Z₁) subject to Σφⱼ₁² = 1.

## Key Concepts
- **Shrinkage penalty / tuning parameter λ**: λ = 0 gives least squares; λ → ∞ gives the null model.
- **Sparse model**: one involving only a subset of variables (lasso output).
- **Scale equivariance**: OLS has it (rescaling Xⱼ by c rescales β̂ⱼ by 1/c); ridge/lasso do not, hence standardize.
- **ℓ₁ / ℓ₂ norm**: Σ|βⱼ| / sqrt(Σβⱼ²).
- **Principal component loadings / scores**: the φ weights / the Z values for each observation.
- **High-dimensional setting**: p ≥ n (or p ≈ n); least squares interpolates (training R² = 1, zero residuals) and Cp/AIC/BIC/adj-R² break because σ̂² = 0.
- **Curse of dimensionality**: adding noise features raises test error even with regularization; more features help only if they are truly associated with y.
- **Multicollinearity in high dimensions**: any variable is a linear combination of others, so the selected set is one of many equally good ones — never claim it is *the* true set.

## Mental Models
- **Think of λ as a bias-variance knob**: increasing λ trades a little bias for a big variance cut until bias dominates; the U-shaped test MSE minimum is found by CV.
- **Use the lasso when you believe the truth is sparse, ridge when it is dense**: neither dominates; CV between them (or elastic net) when unsure.
- **Think of the lasso as soft-thresholding and ridge as proportional shrinkage**: ridge shrinks every coefficient by the same factor; the lasso subtracts a constant and zeroes small ones.
- **Think of PCR as an unsupervised bet**: it assumes high-variance directions of X predict y; PLS hedges by using y to pick directions.

## Anti-patterns
- **Choosing model size by training RSS or R²**: always picks the full model.
- **Fitting ridge/lasso on unstandardized predictors**: penalty depends on units; results change with scale.
- **Reporting training R²/MSE in p > n**: trivially perfect; use independent test or CV error.
- **Interpreting the lasso/stepwise selection as the unique true model**: under collinearity many subsets predict equally well.
- **Calling PCR feature selection**: every component uses all original predictors.
- **Using Cp/AIC/BIC with p > n**: σ̂² is unavailable/zero.
- **Doing feature screening on the whole data before CV**: leaks response information; tuning and selection must be inside each fold (nested CV for honest assessment).

## Code Examples
```python
import sklearn.linear_model as skl, sklearn.model_selection as skm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

K = 5
kfold = skm.KFold(K, random_state=0, shuffle=True)
scaler = StandardScaler(with_mean=True, with_std=True)
lambdas = 10**np.linspace(8, -2, 100) / Y.std()
ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold)   # l1_ratio=0 -> ridge
pipeCV = Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)])
pipeCV.fit(X, Y)
tuned_ridge = pipeCV.named_steps['ridge']
np.min(tuned_ridge.mse_path_.mean(1))        # 115526.71

lassoCV = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=kfold)    # l1_ratio=1 -> lasso
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)]).fit(X, Y)
pipeCV.named_steps['lasso'].alpha_            # 3.147 (sklearn "alpha" = book's λ)
```
- **What it demonstrates**: ridge and lasso both via `ElasticNet` (`l1_ratio` 0 vs 1), standardization inside a `Pipeline` so it is refit per fold, CV over λ with `ElasticNetCV`; `mse_path_` gives CV curves for the 1-SE rule.

```python
from ISLP.models import Stepwise, sklearn_selected
strategy = Stepwise.first_peak(design, direction='forward', max_terms=len(design.terms))
hitters_Cp = sklearn_selected(OLS, strategy, scoring=neg_Cp).fit(Hitters, Y)
pipe = Pipeline([('scaler', scaler), ('pca', PCA()), ('linreg', skl.LinearRegression())])
grid = skm.GridSearchCV(pipe, {'pca__n_components': range(1, 20)}, cv=kfold,
                        scoring='neg_mean_squared_error')
```
- **What it demonstrates**: forward stepwise scored by Cp (ISLP helpers), and PCR as a pipeline with M tuned by grid-search CV (`PLSRegression(n_components=...)` analogously).

## Reference Tables
| Method | Selects variables? | Tuning param | Works p > n? | Best when |
|---|---|---|---|---|
| Best subset | Yes | k | No (≤ n−1) | small p |
| Forward stepwise | Yes | k | Yes (to n−1) | large p, quick |
| Backward stepwise | Yes | k | No | n > p |
| Ridge | No (all shrunk) | λ | Yes | many small/dense effects |
| Lasso | Yes (exact zeros) | λ | Yes | sparse truth |
| PCR | No | M | Yes | top X-variance directions predict y |
| PLS | No | M | Yes | supervised directions; ~ridge/PCR in practice |

| Criterion | Penalty per predictor | Tends to pick |
|---|---|---|
| Cp / AIC | 2σ̂² | larger models |
| BIC | log(n)σ̂² | smaller models |
| Adjusted R² | via n − d − 1 | moderately large |
| CV + 1-SE rule | empirical | parsimonious |

## Worked Example
**Credit data (10 predictors).** Best subset and forward stepwise agree for 1–3 variables (`rating`; + `income`; + `student`), but best 4-var adds `limit` and swaps `rating` for `cards`. Cp picks 6 vars (income, limit, rating, cards, age, student); BIC picks 4 (income, limit, cards, student); adjusted R² picks 7. Validation and CV curves are flat from ~3 variables; the **one-standard-error rule picks the 3-variable model**. On the lasso path, `rating` enters first, then `student` and `limit` almost together, then `income`.

**Simulations.** n = 50, p = 45, all 45 predictors related to y: ridge slightly beats the lasso (minimum MSE). Only 2 of 45 related: the lasso clearly wins in bias, variance and MSE, and CV picks a λ that keeps only the two signal variables.

**Hitters lab (263 players, 19 predictors, Salary).** 5-fold CV MSE: ridge 115,527; lasso 114,691 with λ = 3.15 and **6 of 19 coefficients exactly zero** (13-variable model); null model ≈ 204,139. The lasso matches ridge in accuracy while being more interpretable.

## Key Takeaways
1. Pick model complexity by estimated *test* error (CV), not training fit; use the one-standard-error rule for parsimony.
2. Ridge and lasso trade small bias for large variance reduction; always standardize and tune λ by CV.
3. The lasso performs variable selection via ℓ₁ geometry/soft-thresholding; ridge never zeros coefficients.
4. Neither ridge nor lasso dominates — it depends on how sparse the true signal is.
5. PCR/PLS reduce p to M components; they don't select variables and PCR's success depends on an unsupervised assumption.
6. In high dimensions, training metrics and Cp/BIC are meaningless, added noise features hurt, and selected models are not unique.

## Connects To
- **Ch 3**: least squares, RSS, R², collinearity — the baseline being improved.
- **Ch 5**: CV and validation set used to select λ, k, M.
- **Ch 4 / 10**: same penalties apply to logistic regression and neural network weights (weight decay = ridge).
- **Ch 12**: principal components analysis in depth.
- **External**: elastic net (mixes ℓ₁ and ℓ₂; `l1_ratio` in sklearn), group lasso, relaxed lasso, glmnet.
