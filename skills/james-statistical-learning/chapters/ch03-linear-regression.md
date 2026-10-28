# Chapter 3: Linear Regression

## Core Idea
Linear regression fits `Y ≈ β₀ + Σβⱼ Xⱼ` by least squares. It is the interpretable, low-variance baseline for quantitative responses, with standard errors, t-tests and F-tests for inference. Interactions and polynomial terms extend it, and diagnostics expose where its assumptions fail.

## Frameworks Introduced
- **Simple linear regression / least squares**: `Y ≈ β₀ + β₁X`. Choose β̂₀, β̂₁ to minimize RSS.
  - When to use: one predictor and a roughly linear relationship. Also as a baseline.
  - How: closed-form estimates (below). Assess with SE, t-statistic, p-value, RSE and R².
- **Multiple linear regression**: `Y = β₀ + β₁X₁ + … + βₚXₚ + ε`. βⱼ is the average effect on Y of a one-unit increase in Xⱼ, *holding the other predictors fixed*.
  - How: (1) test H₀: β₁=…=βₚ=0 with the F-statistic; (2) if rejected, decide which variables matter (variable selection: **forward selection**, **backward selection**, **mixed selection**, or criteria such as Mallow's Cₚ, AIC, BIC, adjusted R²; see Ch 6); (3) assess fit with RSE and R², and plot the residuals; (4) predict with confidence and prediction intervals.
  - Why it works / failure mode: it is unbiased and low-variance when the model is approximately correct. It fails when the relationship is non-linear, when errors are correlated, or when collinearity inflates SEs.
- **Qualitative predictors (dummy variables)**: a factor with L levels gets L−1 dummies. The intercept is the baseline mean, and each dummy coefficient is a difference from the baseline. Coding (0/1 or ±1) changes interpretation but not fitted values.
- **Extensions that relax additivity and linearity**:
  - *Interaction term* `X₁X₂` (synergy): `Y = β₀ + β₁X₁ + β₂X₂ + β₃X₁X₂ + ε`. The slope on X₁ becomes `β₁ + β₃X₂`.
  - **Hierarchical principle**: if an interaction is included, also include the main effects, even when their p-values are not significant.
  - *Polynomial regression*: add X², X³, …. The model is still linear in the β's.
- **Six potential problems** (diagnose each):
  1. *Non-linearity*: residuals against fitted values show a pattern. Fix with transforms (log X, √X, X²).
  2. *Correlated error terms* (e.g. time series): SEs are underestimated and intervals too narrow. Check residuals against time for tracking.
  3. *Non-constant variance (heteroscedasticity)*: the residual plot has a funnel shape. Fix with log Y or √Y, or weighted least squares (weights ∝ 1/variance).
  4. *Outliers*: unusual yᵢ. Flag if |studentized residual| > 3. They may barely move the fit but inflate RSE and deflate R².
  5. *High-leverage points*: unusual xᵢ. They can strongly move the fit. Flag if hᵢ ≫ (p+1)/n.
  6. *Collinearity*: correlated predictors make SE(β̂ⱼ) large and t small, which reduces power. Detect with VIF > 5–10. Fix by dropping one predictor or combining them.
- **KNN regression (non-parametric comparison)**: `f̂(x₀) = (1/K) Σ_{xᵢ∈𝒩₀} yᵢ`.
  - When to use: the truth is clearly non-linear and p is small.
  - Failure mode: the **curse of dimensionality**. With many predictors, the nearest neighbors are far away, and KNN degrades quickly when noise predictors are added.

## Key Formulas
- `RSS = Σᵢ (yᵢ − ŷᵢ)²`.
- Least squares (simple): `β̂₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²` and `β̂₀ = ȳ − β̂₁x̄`.
- Standard errors: `SE(β̂₁)² = σ² / Σ(xᵢ−x̄)²` and `SE(β̂₀)² = σ²[1/n + x̄²/Σ(xᵢ−x̄)²]`, where σ² = Var(ε) is estimated by RSE².
- Approximate 95% CI: `β̂₁ ± 2·SE(β̂₁)` (exactly, the 97.5% quantile of a t-distribution with n−2 df).
- t-statistic for H₀: β₁ = 0: `t = (β̂₁ − 0) / SE(β̂₁)`, which follows t with n−2 df under H₀.
- `RSE = √(RSS/(n−p−1))`. In simple regression this is `√(RSS/(n−2))`. It is the average deviation from the true regression line, in units of Y.
- `R² = (TSS − RSS)/TSS = 1 − RSS/TSS`, with `TSS = Σ(yᵢ−ȳ)²`. In simple regression R² = Cor(X,Y)². In multiple regression R² = Cor(Y,Ŷ)².
- Overall F-test: `F = [(TSS−RSS)/p] / [RSS/(n−p−1)]`. F ≈ 1 under H₀ and F > 1 under Hₐ.
- Partial F-test that q specific coefficients are zero: `F = [(RSS₀−RSS)/q] / [RSS/(n−p−1)]`, where RSS₀ is from the model without those q variables. With q = 1, F equals t².
- Leverage (simple): `hᵢ = 1/n + (xᵢ−x̄)² / Σ(xᵢ'−x̄)²`, with 1/n ≤ hᵢ ≤ 1. The average over all observations is (p+1)/n.
- `VIF(β̂ⱼ) = 1 / (1 − R²_{Xⱼ|X₋ⱼ})`, where R²_{Xⱼ|X₋ⱼ} is from regressing Xⱼ on all other predictors. The minimum is 1 (no collinearity).

## Key Concepts
- **Population regression line vs. least squares line**: the true `f` (usually unknown) vs. its estimate from one sample. β̂ is unbiased across samples.
- **Standard error**: the average amount an estimate differs from the true value across repeated samples.
- **p-value**: probability of a |t| at least this large if H₀ is true. A small p-value means an association is likely.
- **Confidence interval vs. prediction interval**: a CI covers the *average* response f(x). A PI covers an *individual* Y = f(x) + ε, so it is always wider.
- **Residual plot**: residuals eᵢ against fitted ŷᵢ. An ideal plot shows no pattern.
- **Studentized residual**: eᵢ divided by its estimated standard error.
- **Leverage**: how unusual xᵢ is, and so how much it can pull the fit.
- **Collinearity / multicollinearity**: two or more predictors closely related, possibly through a combination of several variables (so the pairwise correlation matrix can miss it).
- **Synergy (interaction) effect**: one predictor's effect depends on another predictor's level.
- **Baseline level**: the factor level without a dummy. It is absorbed into the intercept.

## Mental Models
- **Use linear regression when the parametric form is close to the truth.** It beats KNN even with mild non-linearity. As non-linearity grows, KNN catches up only for small p.
- **Think of a multiple-regression coefficient as a partial effect.** A simple-regression slope can reflect a confounder (newspaper "matters" alone only because it correlates with radio, r = 0.35).
- **Use the F-statistic before individual p-values when p is large.** With p = 100 and H₀ true, about 5 individual p-values will fall below 0.05 by chance. F corrects for p.
- **Treat a pattern in the residual plot as a model-specification signal**: curvature means add a non-linear term. A funnel means transform Y.

## Anti-patterns
- **Reading individual t-tests alone when p is large**: you get false discoveries. Use the overall F-test, or multiple-testing control (Ch 13).
- **Dropping main effects but keeping their interaction**: this violates the hierarchical principle and changes the meaning of the interaction.
- **Trusting R² increases when adding variables**: R² never decreases, even for noise variables (newspaper adds 0.8972 → 0.89719 for R², essentially nothing).
- **Interpreting collinear coefficients**: SEs explode and signs and significance are unstable. Check VIF.
- **Ignoring correlated errors in time series**: the CIs are falsely narrow.
- **Using KNN when p is large**: the curse of dimensionality wipes out its non-parametric advantage.
- **Using p ≥ n least squares**: the fit is not unique (a Ch 6 problem).

## Code Examples
```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize, poly

Boston = load_data('Boston'); y = Boston['medv']
design = MS(['lstat'])
X = design.fit_transform(Boston)          # adds intercept column
results = sm.OLS(y, X).fit()
summarize(results)
newX = design.transform(pd.DataFrame({'lstat': [5, 10, 15]}))
pred = results.get_prediction(newX)
pred.conf_int(alpha=0.05)                 # confidence interval
pred.conf_int(obs=True, alpha=0.05)       # prediction interval (wider)
```
- **What it demonstrates**: the ISLP `ModelSpec` transform (`fit`/`transform`), `sm.OLS` (which specifies the model; `.fit()` fits it), and confidence vs. prediction intervals.

```python
X = MS(['lstat', 'age', ('lstat', 'age')]).fit_transform(Boston)   # interaction
X3 = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
results3 = sm.OLS(y, X3).fit()
anova_lm(results1, results3)               # nested-model F-test: F≈177.3, p≈7e-35
vals = [VIF(X, i) for i in range(1, X.shape[1])]
infl = results.get_influence(); infl.hat_matrix_diag   # leverage
```
- **What it demonstrates**: interactions as tuples, `poly()` for polynomial terms (orthogonal by default), `anova_lm` for comparing nested models, VIF, and leverage.

## Reference Tables
**Advertising (n = 200), sales in thousands of units, budgets in $1000s**
| Model | Term | Coef | SE | t | p |
|---|---|---|---|---|---|
| sales ~ TV | Intercept | 7.0325 | 0.4578 | 15.36 | <0.0001 |
| | TV | 0.0475 | 0.0027 | 17.67 | <0.0001 |
| sales ~ TV+radio+newspaper | Intercept | 2.939 | 0.3119 | 9.42 | <0.0001 |
| | TV | 0.046 | 0.0014 | 32.81 | <0.0001 |
| | radio | 0.189 | 0.0086 | 21.89 | <0.0001 |
| | newspaper | −0.001 | 0.0059 | −0.18 | 0.8599 |
| + TV×radio | Intercept / TV / radio / TV×radio | 6.7502 / 0.0191 / 0.0289 / 0.0011 | | | all <0.01 |

| Model | RSE | R² | F |
|---|---|---|---|
| sales ~ TV | 3.26 | 0.612 | 312.1 |
| sales ~ TV+radio+newspaper | 1.69 | 0.897 | 570 |
| sales ~ TV+radio+TV×radio | — | 0.968 | — |

**Linear regression vs. KNN regression**
| Situation | Winner |
|---|---|
| True f linear | Linear regression (KNN's extra variance is pure cost) |
| Strong non-linearity, p = 1–2 | KNN |
| Non-linear, p ≥ 4 with noise predictors | Linear regression (curse of dimensionality) |
| Interpretability needed | Linear regression |

## Worked Example
**Advertising marketing plan.** (1) *Is there a relationship?* The overall F = 570 with tiny p, so yes. (2) *Which media contribute?* TV and radio do (p < 0.0001). Newspaper does not (p = 0.86), even though it is significant alone (simple-regression slope 0.055, p = 0.00115). That simple-regression effect is a surrogate effect via radio (r = 0.35). (3) *How large?* 95% CI for TV in simple regression: [0.042, 0.053] per $1000, about 42–53 units. (4) *Prediction:* for $100k TV and $20k radio, the CI for mean sales is [10,985, 11,528] units, and the PI for a single city is [7,930, 14,580]. (5) *Linear?* No. Residuals show synergy: the plane underpredicts balanced TV/radio splits. Adding TV×radio raises R² from 89.7% to 96.8%, so (96.8−89.7)/(100−89.7) = 69% of the remaining variability is explained. Interpretation: +$1000 TV gives (0.0191 + 0.0011·radio)·1000 extra units.

**Collinearity (Credit)**: regressing balance on age + limit gives limit SE = 0.005. Regressing on rating + limit inflates limit's SE about 12-fold (0.064, p = 0.70). VIFs for age, rating, limit are 1.01, 160.67, 160.59.

**Qualitative (Credit)**: balance ~ own gives intercept 509.80 and own[Yes] 19.73 (p = 0.669). balance ~ region gives East baseline 531.00, and South and West differences under $20 (p ≈ 0.8). The overall F-test p = 0.96, so region has no effect.

**Polynomial (Auto)**: `mpg = 56.9 − 0.466·hp + 0.0012·hp²`. The quadratic term is highly significant (t ≈ 10). R² rises from 0.606 (linear) to 0.688 (quadratic).

## Key Takeaways
1. Least squares gives unbiased β̂. Use SE, t, p and CIs for inference, and RSE and R² for fit.
2. In multiple regression, test the overall F first. Coefficients are partial effects, and simple-regression slopes can mislead.
3. Prediction intervals are always wider than confidence intervals, because they add Var(ε).
4. Interactions and polynomials keep the model linear in β but capture synergy and curvature. Respect the hierarchical principle.
5. Run diagnostics: residual plots (non-linearity, heteroscedasticity), studentized residuals > 3 (outliers), hᵢ ≫ (p+1)/n (leverage), VIF > 5–10 (collinearity).
6. Parametric beats non-parametric when the form is close to right or p is large relative to n.

## Connects To
- **Ch 2**: linear regression is the low-flexibility, low-variance end of the bias-variance trade-off.
- **Ch 4**: logistic regression and GLMs (Poisson) extend the linear model to qualitative and count responses.
- **Ch 5**: the bootstrap and CV give SEs and test error without the model assumptions.
- **Ch 6**: subset selection, ridge and lasso address variable selection, collinearity and p > n.
- **Ch 7**: polynomials, splines and GAMs relax linearity.
- **Ch 13**: multiple testing generalizes the "many p-values" warning.
