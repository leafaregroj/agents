# Chapter 7: Moving Beyond Linearity

## Core Idea
Relax linearity while keeping as much interpretability and least-squares machinery as possible: replace X with a set of **basis functions** (polynomials, step functions, splines) fit by ordinary least squares, use penalized/local fits (smoothing splines, local regression) for a single predictor, and combine per-predictor non-linear functions additively in **generalized additive models (GAMs)**.

## Frameworks Introduced
- **Polynomial regression**: yᵢ = β₀ + β₁xᵢ + β₂xᵢ² + … + β_d xᵢᵈ + εᵢ, fit by least squares.
  - When to use: smooth, global curvature in one predictor.
  - How: degree d rarely > 3 or 4; choose d by nested F-tests (ANOVA) or CV. Pointwise SE of f̂(x₀) from the coefficient covariance: Var f̂(x₀) = ℓ₀ᵀ Ĉ ℓ₀, ℓ₀ = (1, x₀, …, x₀ᵈ); bands = f̂ ± 2·SE.
  - Failure mode: high degree is overly flexible and wild at the boundaries; it is a *global* fit (changing data in one region changes the fit everywhere).
- **Step functions (piecewise constant)**: cut X at c₁ < … < c_K into K + 1 bins via indicator variables C_k(X) = I(c_k ≤ X < c_{k+1}); regress on C₁..C_K (C₀ absorbed in intercept).
  - When to use: natural breakpoints exist (e.g. 5-year age bands).
  - Failure mode: misses trends within bins unless breakpoints are well-placed.
- **Basis functions (general framework)**: y = β₀ + Σ βⱼ bⱼ(x) + ε with fixed, known bⱼ — polynomials, steps, splines, wavelets, Fourier series are all just a design matrix, so all least-squares inference (SEs, F-tests) applies.
- **Regression splines**: piecewise degree-d polynomials joined at **knots**, constrained to be continuous with continuous derivatives up to order d − 1 (**cubic spline**: continuous through the 2nd derivative).
  - How: cubic spline with K knots uses K + 4 coefficients (basis: x, x², x³, plus one **truncated power basis** function h(x, ξ) = (x − ξ)₊³ per knot). Specify **df** and let software place knots at uniform quantiles; choose df / K by CV.
  - **Natural spline**: extra constraint of linearity beyond the boundary knots → much smaller boundary variance; frees 4 df for the same knots.
  - Why it works: flexibility is local and controlled by number of knots, not by a high degree; more stable than polynomials, especially near the edges.
- **Smoothing splines**: minimize `Σ(yᵢ − g(xᵢ))² + λ ∫ g''(t)² dt` over all functions g.
  - When to use: one predictor, want automatic knot placement; λ the only tuning knob.
  - How: solution is a *shrunken* natural cubic spline with knots at every unique xᵢ; λ chosen by LOOCV (cheap — closed form) or by specifying effective df.
  - Why: the penalty on total curvature trades roughness for fit; λ = 0 interpolates, λ → ∞ gives the linear least-squares line.
- **Local regression** (Algorithm 7.1): at target x₀ (1) take the fraction s = k/n of training points nearest x₀; (2) weight them with Kᵢ₀ (zero outside, highest at nearest); (3) fit weighted least squares minimizing Σ Kᵢ₀(yᵢ − β₀ − β₁xᵢ)²; (4) f̂(x₀) = β̂₀ + β̂₁x₀. Repeat for every x₀ (memory-based, like KNN).
  - How: the **span s** is the key tuning parameter (smaller = wigglier); choose by CV. Can make it varying-coefficient (local in some variables, global in others).
  - Failure mode: degrades with more than 3–4 local dimensions (curse of dimensionality; few nearby points).
- **Generalized additive models (GAMs)**: `yᵢ = β₀ + Σⱼ fⱼ(xᵢⱼ) + εᵢ`, any fⱼ from the above (natural splines → one big least-squares fit; smoothing splines → **backfitting**: update each fⱼ against the partial residual, cycle until convergence).
  - When to use: several predictors, want non-linearity per variable while retaining interpretable partial-effect plots.
  - How: pick each term's df (e.g. 4–5 for continuous predictors, step/indicator for qualitative); compare nested GAMs (no term vs linear vs smooth) with ANOVA F-tests.
  - Failure mode: additivity misses interactions unless added explicitly (Xⱼ×X_k or bivariate fⱼₖ(Xⱼ,X_k) via local regression / 2-D splines).
- **Logistic GAM**: `log(p(X)/(1 − p(X))) = β₀ + f₁(X₁) + … + f_p(X_p)`.

## Key Formulas
- Polynomial logistic: `Pr(yᵢ > 250 | xᵢ) = exp(β₀ + β₁xᵢ + … + β_d xᵢᵈ) / (1 + exp(β₀ + … + β_d xᵢᵈ))`.
- Step-function basis: `C₀(X) = I(X < c₁)`, `C_k(X) = I(c_k ≤ X < c_{k+1})`, `C_K(X) = I(c_K ≤ X)`; Σ C_k = 1.
- Truncated power basis: `h(x, ξ) = (x − ξ)³ if x > ξ, else 0` — adding it for each knot ξ preserves continuity of f, f', f'' and only breaks f'''.
- Parameter counts: piecewise cubic with K knots, unconstrained = 4(K + 1); cubic spline = **K + 4**; natural cubic spline ≈ K (4 fewer than cubic spline with same knots, counting intercept).
- Smoothing spline: `minimize_g Σᵢ(yᵢ − g(xᵢ))² + λ ∫ g''(t)² dt`.
- Effective degrees of freedom: `ĝ_λ = S_λ y`, `df_λ = Σᵢ {S_λ}ᵢᵢ` (trace of the smoother matrix); decreases from n to 2 as λ goes 0 → ∞.
- LOOCV for smoothing splines (one fit): `RSS_cv(λ) = Σᵢ ( (yᵢ − ĝ_λ(xᵢ)) / (1 − {S_λ}ᵢᵢ) )²`.
- Local linear fit at x₀: `minimize Σᵢ Kᵢ₀ (yᵢ − β₀ − β₁xᵢ)²`.

## Key Concepts
- **Knot**: a point where the piecewise polynomial's coefficients change.
- **Boundary knots / natural constraint**: linear extrapolation beyond the range of the data to tame edge variance.
- **Degrees of freedom (df)**: number of free parameters; for smoothers, the *effective* df (trace of S_λ) measures flexibility.
- **Smoother matrix S_λ**: linear map from y to fitted values for a smoothing spline.
- **Roughness penalty**: λ∫g''(t)²dt, total curvature of g.
- **Span s**: fraction of points used in each local regression; plays the role λ does for splines.
- **Backfitting**: iterative fitting of one GAM term at a time to partial residuals.
- **Partial dependence plot**: fⱼ(xⱼ) with pointwise SE bands, holding other terms fixed.

## Mental Models
- **Think of every method here as "linear regression on a cleverly built design matrix"** (except smoothing splines/local regression, which are linear smoothers): all of Ch 3's inference carries over.
- **Use regression/natural splines instead of high-degree polynomials**: add knots for local flexibility instead of raising the degree; put more knots where f changes fast (in practice, uniform quantiles + CV on df).
- **Think of λ (smoothing spline), df (spline), span s (loess), d (polynomial) as one bias-variance dial**, chosen by CV.
- **Use GAMs as the interpretable middle ground** between linear models and fully non-parametric black boxes (trees/boosting, Ch 8): flexible per-variable, but additive.

## Anti-patterns
- **High-degree polynomials (d > 4)**: wild oscillations, especially near boundaries where SE bands explode.
- **Too many / badly placed knots without CV**: overfitting; too few in regions of rapid change underfits.
- **Interpreting the pointwise SE band as a simultaneous band**: it covers each x₀ separately.
- **Using unrestricted cubic splines when extrapolation near edges matters**: use natural splines.
- **Local regression in many dimensions**: too few neighbors (curse of dimensionality).
- **Assuming a GAM captures interactions**: it does not unless you add them.
- **Applying smoothing to a qualitative predictor in pygam**: use `f_gam(..., lam=0)` (factor term, unpenalized).

## Code Examples
```python
from ISLP.models import ModelSpec as MS, poly, bs, ns
from statsmodels.stats.anova import anova_lm

models = [MS([poly('age', degree=d)]) for d in range(1, 6)]
Xs = [model.fit_transform(Wage) for model in models]
anova_lm(*[sm.OLS(y, X_).fit() for X_ in Xs])        # nested F-tests, degrees 1..5

bs_age = MS([bs('age', internal_knots=[25, 40, 60])])  # cubic spline, 3 knots -> 6 basis cols (+intercept)
ns_age = MS([ns('age', df=5)]).fit(Wage)               # natural spline, knots at quantiles
```
- **What it demonstrates**: polynomials, B-splines and natural splines are just `ModelSpec` transforms fed to `sm.OLS`; `anova_lm` compares nested models.

```python
from pygam import s as s_gam, l as l_gam, f as f_gam, LinearGAM, LogisticGAM
from ISLP.pygam import approx_lam, degrees_of_freedom, anova as anova_gam

gam_full = LinearGAM(s_gam(0) + s_gam(1, n_splines=7) + f_gam(2, lam=0))
gam_full.fit(Xgam, y)                                  # columns: age, year, education
age_term = gam_full.terms[0]
age_term.lam = approx_lam(Xgam, age_term, df=4+1)      # set smoothness via df (+1 for intercept)
gam_0      = LinearGAM(age_term + f_gam(2, lam=0))                    # no year
gam_linear = LinearGAM(age_term + l_gam(1, lam=0) + f_gam(2, lam=0))  # linear in year
anova_gam(gam_0, gam_linear, gam_full)
```
- **What it demonstrates**: smoothing-spline GAMs with `s_gam`/`l_gam`/`f_gam` terms, specifying smoothness by df rather than λ, and ANOVA on nested GAMs.

## Reference Tables
| Method | Local/global | Tuning | Linear in params (OLS)? | Notes |
|---|---|---|---|---|
| Polynomial | Global | degree d (≤ 4) | Yes | boundary instability |
| Step functions | Local (bins) | cutpoints | Yes | discontinuous |
| Regression (cubic) spline | Local | knots / df (K + 4) | Yes | smooth to 2nd deriv. |
| Natural spline | Local | knots / df | Yes | linear at edges, lower boundary variance |
| Smoothing spline | Local | λ or effective df | Linear smoother | knots at all xᵢ; LOOCV in closed form |
| Local regression | Local | span s | Linear smoother | memory-based; ≤ 3–4 dims |
| GAM | Additive | per-term df/λ | Yes (ns) / backfitting (ss) | no interactions by default |

## Worked Example
**Wage data (3,000 men), wage vs age.** Degree-4 polynomial fits the rise-then-fall; a logistic degree-4 polynomial for Pr(wage > 250 | age) shows wide SE bands where high earners are sparse. ANOVA on degrees 1–5: linear→quadratic p ≈ 0, quadratic→cubic p = 0.0017, cubic→quartic p ≈ 0.05, quartic→quintic p = 0.37 → choose cubic or quartic (the squared t-statistics from orthogonal `poly` equal the F-statistics). Step function with quartile cuts (≤33.75, 33.75–42, 42–51, >51) gives means ≈ 94, 117, 119, 117 (thousand $). A natural cubic spline with 4 df puts its 3 knots at the age quartiles (25th/50th/75th percentiles); a natural spline with 15 df is far stabler at the edges than a degree-15 polynomial, which flails at the boundaries. Smoothing spline with 16 df vs LOOCV-chosen λ (**6.8 effective df**) look similar; prefer the simpler. Local linear regression: span 0.2 (≈16.4 df) vs 0.7 (≈5.3 df).

**GAM**: wage = β₀ + f₁(year) + f₂(age) + f₃(education), natural splines with 4 and 5 df plus education steps. Wage rises slightly with year, peaks at middle age, increases with education. ANOVA (lab, pygam): adding linear year to a no-year GAM, p = 0.002; smooth vs linear year, p ≈ 0.44 → a linear year term suffices, while age needs a non-linear term (p ≈ 0).

## Key Takeaways
1. Most non-linear extensions are basis expansions fit by least squares, so SEs and F-tests still work.
2. Splines beat high-degree polynomials: local flexibility via knots, stable boundaries with natural splines.
3. Smoothing splines put knots everywhere and control smoothness with λ; summarize flexibility as effective df; LOOCV is a single-fit computation.
4. Local regression fits weighted lines in a moving window; span s is the bias-variance knob.
5. GAMs extend this to many predictors additively, keep interpretability via per-variable plots, and work for logistic models too; add interactions explicitly.
6. Choose degree/knots/df/λ/span by CV or nested F-tests.

## Connects To
- **Ch 3**: least squares, F-tests, interactions and polynomial terms; the machinery reused here.
- **Ch 4**: logistic regression, extended to polynomial/spline/GAM logistic models.
- **Ch 5**: CV and the leverage-style LOOCV shortcut for choosing df/λ.
- **Ch 6**: smoothing splines as ridge-like penalized regression; effective df.
- **Ch 8**: trees/boosting as non-additive, interaction-capturing alternatives; BART and boosting with stumps yield additive models.
- **External**: B-spline bases, thin-plate/tensor-product splines (mgcv), LOESS/LOWESS, MARS.
