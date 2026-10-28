# Chapter 3: Risk

## Core Idea
Define risk as the **standard deviation of return**, because it is universal, symmetric, flexible, aggregates from assets to portfolios, and is the only risk measure that can be reliably forecast. Charge for risk in proportion to **variance**. Forecast the covariance matrix with a **structural multiple-factor risk model**, and use it to decompose active risk into its sources.

## Frameworks Introduced
- **Standard deviation as risk**: chosen over semivariance, target semivariance, shortfall probability, and VaR.
  - When to use: always, for building portfolios. Report risk in any transformation you like, but aggregate in σ.
  - Why it works: σ is stable and persistent. Alternative measures show **no persistence beyond what variance implies** (Kahn–Stefek 1996), they waste half the data, and they can't be aggregated across assets. Active returns are symmetric by construction.
  - Failure mode: options and dynamic strategies (portfolio insurance) have engineered asymmetries, so σ is imperfect there.
- **Structural multiple-factor risk model**: r_n = Σ_k X_{n,k}·b_k + u_n, so V = X·F·Xᵀ + Δ.
  - When to use: to forecast V for any universe. It replaces N(N+1)/2 noisy estimates with K factors plus N specific variances.
  - How: (1) choose **a priori** factors (exposures known at the start of the period): industries (0/1, or split for conglomerates) and risk indices built from descriptors and standardized to mean 0, SD 1; (2) run a monthly cross-sectional GLS regression of returns on exposures to get factor returns (Fama–MacBeth); (3) forecast F from the history of factor returns (weighting, Bayesian priors, ARCH/GARCH); (4) model specific risk separately.
  - Factor-choice criteria: **incisive** (separates returns), **intuitive** (a credible investment story), **interesting** (explains performance).
- **Three factor types**: external-response (macro) factors, which suffer from errors-in-variables, nonstationarity, and poor data and add nothing beyond cross-sectional factors; **cross-sectional attribute factors** (fundamental + market), which are powerful and preferred; and statistical factors, which are hard to interpret, prone to spurious fits, and can't capture time-varying exposures such as momentum. Avoid the last.
- **Bet classification**: **inherent** (benchmark risk you must bear), **intentional** (bets on your alphas; the highest-alpha stocks should have the highest marginal contribution to risk), **incidental** (unintended side effects, e.g. a yield screen creating industry bets, or risk piling up from stock-by-stock decisions).
- **Three uses of a risk model**: present (current risk analysis and decomposition), future (portfolio construction), past (performance attribution).

## Key Formulas
- Two-stock 50/50 portfolio: σ_P = ½·√(σ₁² + σ₂² + 2ρ₁₂σ₁σ₂) ≤ (σ₁ + σ₂)/2. Equality holds only if ρ = 1. **Risks don't add.**
- N uncorrelated stocks, equal weight: σ_P = σ/√N.
- Equal pairwise correlation ρ: σ_P = σ·√[(1 + ρ(N−1))/N], which tends to σ·√ρ as N → ∞.
- Time scaling (no autocorrelation): variance ∝ horizon. σ_annual = √12·σ_monthly. A 5% annual risk is 2.5% per quarter and 10% over 4 years.
- Active risk (tracking error): ψ_P = Std{r_P − r_B}.
- Residual risk: ω_P = √(σ_P² − β_P²·σ_B²).
- Cost of risk = λ·(risk)². For example, λ_A = 0.1 and ψ = 4% give 0.1·16 = 1.6% of expected return.
- Single-factor model: V_nm = β_n·β_m·σ_M² + δ_nm·ω_n². Constant-correlation model: V_nm = ρ·σ_n·σ_m.
- Historical covariance needs T > N (rank = min(N, T−1)). Otherwise some positions appear riskless.
- Exposure standardization: x = (x_raw − mean[x_raw]) / Std[x_raw].
- GLS factor returns: b = (XᵀΔ⁻¹X)⁻¹XᵀΔ⁻¹r. Each b_k is the return of a **factor portfolio** (unit exposure to k, zero to the other factors, minimum risk).
- Specific risk: E{u_n²} = S(t)·[1 + v_n(t)]. S is the average level (time-series model) and v_n is the cross-sectional deviation (factor model).
- Portfolio: x_P = Xᵀh_P; σ_P² = x_PᵀF x_P + h_PᵀΔh_P. Active: h_PA = h_P − h_B, ψ_P² = x_PAᵀF x_PA + h_PAᵀΔh_PA.
- Asset betas: β = V·h_B/σ_B² (factor part + specific part; the industry part dominates).
- Marginal contributions: MCTR = V·h_P/σ_P; MCRR = V·h_PR/ω_P (h_PR = h_P − β_P·h_B); MCAR = V·h_PA/ψ_P.
- MCAR = k₁·β + k₂·MCRR, with k₁ = β_PA·σ_B²/ψ_P and k₂ = ω_P/ψ_P (0 ≤ k₂ ≤ 1) [reconstructed].
- Factor marginal contribution (approx.): FMCAR ≈ F·x_PA/ψ_P. The specific-risk term is small except for thin industries (< ~10 members).
- **Risk attribution**: ψ_P = Σ_n h_PA(n)·MCAR(n). Each covariance pair is split half to each member. h_PA(n)·MCAR(n) is the *relative* marginal contribution (RMCAR).
- Correlation in one-factor model: ρ_nm = β_nβ_mσ_M²/(σ_nσ_m). With β = 1 and equal ω, ρ = σ_M²/(σ_M² + ω²), so **correlations rise when market volatility rises**.
- Woodbury inverse: V⁻¹ = Δ⁻¹ − Δ⁻¹X(XᵀΔ⁻¹X + F⁻¹)⁻¹XᵀΔ⁻¹. This needs only K×K inversions.

## Key Concepts
- **Volatility**: the standard deviation of return, quoted in percent per year.
- **Active risk / tracking error ψ**: the SD of portfolio minus benchmark return.
- **Residual risk ω**: the SD of return orthogonal to the benchmark.
- **Semivariance / downside risk**: variance from below-mean returns only. It is ≈ ½ variance for symmetric returns (Magellan: 21.6 vs 39.5, i.e. 55%).
- **Shortfall probability / VaR**: P(return < target), or the return at a target probability. These depend on tails and are poorly forecastable.
- **Descriptor**: one specific measurement within a risk index (e.g. option-implied volatility within Volatility).
- **Risk indices (US)**: volatility, momentum, size, liquidity, growth, value, earnings volatility, financial leverage (plus foreign-currency sensitivity).
- **Factor portfolio**: minimum-risk portfolio with unit exposure to one factor and zero to all others. It is long/short and not investible.
- **Specific return u_n**: return the factors can't explain. Specific returns are assumed uncorrelated across assets and with the factors.

## Mental Models
- Think of **active risk as active position × stock risk, not benchmark weight**. A 1% overweight in a small stock isn't necessarily riskier than a 1% overweight in a large one, so "±50% of benchmark weight" limits are misguided.
- Use **σ for building and any transformation for reporting**: under normality, VaR and shortfall are just functions of μ and σ.
- Think of a **risk model as the instrument panel**: it says not only how much risk you have but *why*, so you can keep risk only where you expect to outperform.
- Treat **industry factors as the model's intercept**. The market has unit total industry exposure, so industry factor returns absorb the market move and are the most volatile factors. Risk indices capture extra-market return.

## Anti-patterns
- **Historical covariance matrices**: they need T > N (40+ years monthly for the S&P 500), suffer from mergers, survivorship and selection bias, and give many bad estimates (5% of 125,250 entries is about 6,262 bad numbers).
- **Statistical (PCA) factors**: uninterpretable, spurious, and unable to model momentum or other changing exposures.
- **Macro response factors alone**: errors-in-variables from about 60-month regressions, nonstationary betas, and poor government data.
- **Downside-risk optimization**: no forecastable information beyond variance, half the data, and hard to aggregate.
- **Position limits proportional to cap weight**: they misallocate active risk.
- **Incremental stock-by-stock management**: it piles up incidental factor bets.
- **Thin industries (< ~10 names)**: large estimation error.
- **Ambiguous sector marginal contributions**: know which weighting is used. A reasonable default is the portfolio's own total industry weights within the sector.

## Reference Tables
Empirical distribution, BARRA HICAP (~1,200 largest US stocks), 60-month windows, averaged over 6/1980, 6/1990, 12/1998:

| Percentile | Total risk % | Residual risk % | Beta |
|---|---|---|---|
| 90 | 50.8 | 45.0 | 1.67 |
| 75 | 40.1 | 34.3 | 1.36 |
| 50 | 30.6 | 25.1 | 1.08 |
| 25 | 24.6 | 19.6 | 0.79 |
| 10 | 20.4 | 16.4 | 0.52 |

Typical ranges: total risk 25–40%, residual risk 20–35%, beta 0.80–1.35. Only the cap-weighted average beta must equal 1.

Model facts: monthly cross-sectional R² is 30–40% (about 1,000 assets, 50 factors), higher in big market moves. BARRA US (1998) had 65 factors, 52 of them industries. GLS weights ∝ 1/specific variance, or √cap as a proxy (specific variance falls by a factor of about 0.7 when size doubles).

Risk-forecast test (Kahn 1997, 29 funds, 1993/94): structural model gave 0 three-sigma and 1 two-sigma events (3%), with 76% of outcomes under 1σ. Historical 3-year SD gave 1 three-sigma (3%) and 5 two-sigma (17%), with 72% under 1σ. Neither was biased on average. Structural forecasts tracked realized risk much better.

## Worked Example
**MMI portfolio vs S&P 500, Dec 1992**: forecast σ_P = 20.5% and σ_B = 20.1%, tracking error ψ = 4.2% (so the annual return is within ±4.2% of the index about 2/3 of the time), β = 0.96 (a 100 bp S&P excess return implies 96 bp expected). Marginal contribution to active risk: American Express 0.006, so raising it from 2.28% to 3.28% adds about 0.6 bp of active risk. AT&T is negative (−0.009), the most diversifying holding. IBM is the largest (0.074), the most concentrating.

**Average correlation**: 20 MMI stocks have average σ = 27.8% and an equal-weighted portfolio σ of 20.4%. Solving 20.4 = 27.8·√[(1+19ρ)/20] gives ρ ≈ 0.52.

**Diversification (problem)**: with σ = 35% and ρ = 0.5, an equal-weighted portfolio has risk of about 27.1% at N = 5, 25.4% at N = 20, 24.9% at N = 100, and a floor of 24.7% (= 35·√0.5).

## Key Takeaways
1. Use standard deviation for risk and variance for the cost of risk. Alternative measures add nothing forecastable.
2. Risks don't add across assets or across time, but variances add across uncorrelated periods (√T scaling).
3. Institutional managers should focus on active and residual risk, which depend on active positions, not benchmark weights.
4. Use a structural factor model with a priori, cross-sectional, standardized factors (industries + risk indices) to forecast V.
5. Decompose risk (market/residual, factor/specific, by asset and factor marginal contribution) to separate intentional bets from incidental ones.
6. Check that your highest-alpha names carry the highest marginal contribution to risk. If not, the portfolio doesn't reflect your views.

## Connects To
- **Ch 2**: factor portfolios generalize characteristic portfolios. β comes from the risk model.
- **Ch 4–5**: active/residual risk and λ·variance feed value added and residual risk aversion. MCRR ∝ alpha at optimality.
- **Ch 7**: the same factor structure underlies the APT.
- **Ch 14**: the risk model in portfolio construction.
- **Ch 17**: risk-based performance attribution.
- **ARCH/GARCH (Engle 1982; Bollerslev et al. 1992)**: conditional variance forecasting for F.
