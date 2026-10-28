# Chapter 23: Appendix A (Standard Notation) + Appendix C (Return and Statistics Basics)

## Core Idea
A single notation runs through the whole book — bold lowercase for vectors, bold uppercase for matrices, plain subscripted letters for elements (r_n is the n-th element of **r**). Excess returns, sample statistics with T − 1, standard errors that shrink like 1/√T, and OLS beta = Cov/Var are the statistical toolkit everything else is built on.

## Frameworks Introduced
- **Notation convention**: vectors lowercase bold (**h**, **r**, **α**), matrices uppercase bold (**V**, **F**, **X**, **Δ**); elements non-bold with subscripts. Portfolio subscripts: P (portfolio), B (benchmark), PA (active = P − B), PR (residual).
- **Return hierarchy** used throughout: total return → excess return (over risk-free) → split into benchmark (beta) part and **residual** part → residual into common-factor and **specific** parts. The book works almost entirely with **excess returns**; occasionally total returns; never explicitly total rates of return.
- **Standard-error thinking**: every estimate (mean, std dev, variance, alpha, IR) carries an error that shrinks as 1/√T; means are by far the hardest to pin down.

## Key Formulas
**Returns (period t to t + Δt; price P, distributions d):**
- Total return: `R(t) = [P(t+Δt) + d(t)] / P(t)`
- Total rate of return: `i(t) = R(t) − 1`
- Risk-free total return / rate: `R_F = 1 + i_F`
- Excess return: `r(t) = R(t) − R_F(t)` (= i − i_F)
- Mid-period cash distributions: assume reinvestment at the risk-free rate or the asset's own return.

**Sample statistics (T observations of r_n(t)):**
- Mean: `r̄_n = (1/T)·Σ_t r_n(t)`
- Variance: `Var_n = (1/(T−1))·Σ_t [r_n(t) − r̄_n]²` — use T − 1 when the mean is also estimated (unbiased); use T if the mean is known independently.
- Std dev: `σ_n = √Var_n`
- Covariance: `Cov_nm = (1/(T−1))·Σ_t [r_n(t) − r̄_n]·[r_m(t) − r̄_m]`
- Correlation: `ρ_nm = Cov_nm / (σ_n·σ_m)`

**Standard errors (normal errors; approximations improve with large T):**
- Mean: `SE(r̄) = σ / √T`
- Std dev: `SE(σ̂) ≈ σ / √(2T)`
- Variance: `SE(σ̂²) ≈ σ²·√(2/T)`
- (Used elsewhere in the book) t statistic `t = estimate / SE`; `IR ≈ t_α / √(years)`; `SE(IR) ≈ 1/√(years)`.

**Simple linear regression (beta estimation):**
- Model: `r_n(t) = α_n + β_n·r_M(t) + ε_n(t)`
- OLS minimizes `Σ_t ε_n(t)²`
- `β̂_n = Cov(r_n, r_M) / Var(r_M)`; `α̂_n = r̄_n − β̂_n·r̄_M`
- `R² = 1 − Var(ε_n) / Var(r_n)` (= share of variance explained; for one regressor, R² = ρ²)
- OLS residuals are uncorrelated with the regressor: `Cov(ε_n, r_M) = 0`.
- In the book, betas are usually defined by the same Cov/Var formula but with **forecast** covariances from a risk model rather than sample estimates. Weighted / multivariate regressions follow the same principle (minimize weighted squared errors; R² definition unchanged).

## Key Concepts — Standard Notation (Appendix A)
Greek symbols reconstructed from the book's usage (they are garbled in the source text).

| Symbol | Meaning |
|---|---|
| **Realized returns** | |
| R | total return (P_new + dividend)/P_old |
| i_F | risk-free rate of return |
| R_F | risk-free total return |
| r | excess returns |
| θ | residual returns |
| b | factor returns |
| u | specific returns |
| **Expected returns** | |
| f | expected excess returns |
| μ | long-term expected excess returns |
| α | expected residual returns (alpha) |
| φ | expected exceptional returns |
| m | expected factor returns |
| **Risk** | |
| σ | total risk |
| ω | residual risk |
| ψ | active risk (tracking error) |
| β | asset betas |
| β_P | portfolio beta (exposure to benchmark risk) |
| β_PA | active portfolio beta |
| V | asset-by-asset covariance matrix |
| F | factor covariance matrix |
| Δ | specific variance matrix (diagonal) |
| **Portfolios and assets** | |
| h_P | portfolio holdings |
| h_PR | residual portfolio holdings |
| h_PA | active portfolio holdings (h_P − h_B) |
| X | matrix of all assets' factor exposures |
| x_P | portfolio P's factor exposures (Xᵀh_P) |
| **Performance and value added** | |
| λ_T | total risk aversion |
| λ_BT | benchmark timing risk aversion |
| λ_R | residual risk aversion |
| λ_A | active risk aversion |
| λ_S | short-term risk aversion |
| SR | Sharpe ratio |
| IR | information ratio |
| IC | information coefficient |
| BR | breadth |
| **Portfolio names** | |
| B | benchmark portfolio |
| M | market portfolio |
| Q | fully invested portfolio with maximum Sharpe ratio |
| C | fully invested portfolio with minimum risk |
| q | minimum-risk portfolio with expected return = 1 |
| A | minimum-risk portfolio with α = 1 (characteristic portfolio of alpha) |
| S | a further characteristic portfolio defined in the book's Ch 2 appendix (entry garbled in source) |
| **Other** | |
| e | vector of 1s (eᵀh = 1 ⇔ fully invested) |

## Mental Models
- Read any formula by its subscripts: `_P` portfolio, `_B` benchmark, `_PA` active, `_PR` residual; `_n` asset; λ with subscript = which risk is being penalized.
- Think of σ ⊇ ω ⊇ specific: total risk contains residual risk (after beta), which contains specific risk (after factors); ψ is risk relative to the benchmark (= ω when β_P = 1).
- Use SE(mean) = σ/√T to sanity-check any return claim: with 20% volatility, 5 years of data gives a mean SE of ~9%/yr — returns are nearly impossible to estimate from history; risk is easy.

## Anti-patterns
- **Mixing total, excess and rate-of-return definitions**: the book's formulas assume excess returns.
- **Using T vs. T − 1 carelessly in tiny samples**: they differ materially; avoid heavy statistics on very small samples.
- **Confusing sample beta with forecast beta**: the book's betas come from risk-model covariances, not raw OLS.
- **Ignoring corporate actions** (splits, stock dividends) in return calculations.

## Reference Tables
Accuracy vs. sample length (σ = 20%/yr, annual observations; illustrative from the formulas above):

| T (years) | SE(mean) | SE(σ̂) | SE(IR) |
|---|---|---|---|
| 1 | 20% | 14.1% | 1.00 |
| 5 | 8.9% | 6.3% | 0.45 |
| 10 | 6.3% | 4.5% | 0.32 |
| 25 | 4.0% | 2.8% | 0.20 |

(With monthly data SE(σ̂) falls much faster, since it depends on the number of observations, while SE(mean) depends only on the time span.)

## Key Takeaways
1. Default to excess returns; residual = excess − β·benchmark excess; active = portfolio − benchmark.
2. Estimate variance with T − 1; correlation = Cov/(σσ).
3. Means are noisy (SE = σ/√T), volatilities are precise (SE ≈ σ/√(2T)) — the root reason the book forecasts risk structurally and treats alpha forecasting with humility.
4. Beta = Cov(r_n, r_M)/Var(r_M); OLS residuals are orthogonal to the regressor; R² measures explained variance.
5. Keep the notation table at hand: λ subscripts, portfolio letters (B, M, Q, C, q, A) and Greek risk symbols (σ, ω, ψ) recur in every chapter.

## Connects To
- **Ch 2**: characteristic portfolios (C, Q, q, A) and CAPM beta.
- **Ch 3**: V = X F Xᵀ + Δ structural risk model; σ, ω, specific risk.
- **Ch 4–5**: active/residual decomposition; λ_R, λ_A, IR.
- **Ch 12 & 17**: t statistics, IR ≈ t/√T, SE(IR) ≈ 1/√T in performance and information analysis.
- **External**: OLS/GLS regression texts (Neter & Wasserman; Hogg & Craig); sampling distribution of variance (χ²).
