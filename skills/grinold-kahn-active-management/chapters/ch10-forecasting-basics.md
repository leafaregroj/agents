# Chapter 10: Forecasting Basics

## Core Idea
Active management is forecasting. The **basic forecasting formula** turns raw information into a *refined forecast* of exceptional return. For one signal it reduces to the forecasting rule of thumb: **alpha = volatility · IC · score**. That rule controls for expectations (through the score), skill (the IC) and volatility. Return forecasts barely change risk forecasts.

## Frameworks Introduced
- **Naive / raw / refined forecasts**:
  - *Naive* = consensus expected return, β·μ_B, which leads to holding the benchmark. Do not use historical averages: they have huge sample error and do not lead back to the benchmark.
  - *Raw* = the manager's information in its native units, such as an earnings estimate, a buy/sell list or a ranking.
  - *Refined* = the change in expected return caused by observing the raw forecast, i.e. the exceptional return (residual plus benchmark timing).
- **Basic forecasting formula**: E{r|g} = E{r} + Cov{r,g}·Var⁻¹{g}·(g − E{g}).
  - It is the best linear unbiased (minimum-MSE) estimate. Under joint normality it is also the conditional expectation and the MLE, and it attains the Cramér–Rao bound.
  - Its errors are uncorrelated with the raw forecasts. If they were correlated, information would be "left on the table".
- **Forecasting rule of thumb (one asset, one signal)**:
  - When to use: always, as the default refinement, especially in unstructured situations (tips, buy lists, fractiles, rankings).
  - How: (1) *score* the raw signal: z = (g − E{g})/Std{g}; (2) multiply by IC; (3) multiply by the volatility of the return being forecast.
  - Why it works: the score removes what was already expected (e.g. earnings that match expectations do not move the price). The IC = 0 case automatically gives zero alpha. Volatility supplies the units, so for the same score a volatile stock gets a bigger alpha (an Internet stock vs. a utility on the same buy list).
- **Combining multiple signals**: refined = volatility · Σ_j IC\*_j·z_j, where the adjusted ICs account for the correlation between signals.
  - Uncorrelated signals simply add.
  - Perfectly correlated signals mean the second adds nothing.
  - With three equal-IC signals where two are highly correlated with each other and uncorrelated with the third, the correlated pair is roughly halved, so the third counts as much as the pair together.
  - Alternatively, orthogonalize the scores first, after which combination is simple addition.
- **IC calibration guide** (use when there is no history): good = 0.05, great = 0.10, world-class = 0.15. **IC > 0.20 usually signals a faulty backtest or an imminent insider-trading investigation.**
- **Forecasts don't change risk**: the conditional variance depends only on skill (IC), *not on the forecast itself*, and at realistic ICs the reduction is negligible. Do not rewrite correlations because of a return view.
- **Discipline for advanced techniques**: master simple methods first. Test any sophisticated method (1) on random data, where it should find nothing, and (2) on simulated data with a known relationship, where it should find it. Use bootstrapping for significance.

## Key Formulas
- **Basic forecasting formula (10.1)**: E{r|g} = E{r} + Cov{r,g}·Var⁻¹{g}·(g − E{g}). Here r is the N-vector of excess returns and g the K-vector of raw forecasts.
- **Refined forecast (10.2)**: φ = E{r|g} − E{r}.
- **Naive forecast (10.3)**: E{r} = β·μ_B. In residual form (10.4–10.5), E{θ} = 0 and φ = Cov{θ,g}·Var⁻¹{g}·(g − E{g}).
- **Rule of thumb (10.9–10.11)**: φ = Cov{r,g}/Var{g}·(g − E{g}) = **ω · IC · z**, where z = (g − E{g})/Std{g} and IC = Corr{r,g}.
- **Regression version (10.12–10.16)**: r(t) = c0 + c1·g(t) + ε, with c1 = Cov{r,g}/Var{g} = IC·Std{r}/Std{g}. This gives the same ω·IC·z using sample estimates. If you have a strong prior mean m, replace c0 rather than trusting the noisy sample mean.
- **Two signals (10.20–10.22)**: φ = ω·(IC\*_g·z_g + IC\*_g′·z_g′), with
  - IC\*_g = (IC_g − ρ·IC_g′)/(1 − ρ²)
  - IC\*_g′ = (IC_g′ − ρ·IC_g)/(1 − ρ²)
  - where ρ = Corr{g, g′}.
- **Combined IC (10.23)**: IC²_comb = (IC_g² + IC_g′² − 2ρ·IC_g·IC_g′)/(1 − ρ²). With uncorrelated signals, IC²_comb = IC_g² + IC_g′².
- **K signals (10A.21–10A.23, matrix form)**: φ = ω·ICᵀ·ρ_g⁻¹·z, and IC²_comb = ICᵀ·ρ_g⁻¹·IC, where ρ_g = the signal correlation matrix and IC = the vector of individual ICs. The refined forecast is always a linear combination of scores.
- **Conditional variance**: Var{r|g} = Var{r} − Cov{r,g}·Var⁻¹{g}·Cov{g,r}. For one signal this gives **ω_POST = ω_PRIOR·√(1 − IC²)** (10.30).
- **Optimal holdings with uncorrelated residuals**: h_n = α_n/(2λ·ω_n²). With α = ω·IC·z, h_n·ω_n = IC·z_n/(2λ), so each asset has equal expected risk. A flat ±1% alpha would make the optimizer pile into the lowest-residual-risk buys.

## Key Concepts
- **Score (z-score)**: the raw forecast standardized to mean 0 and std 1. Over time for one stock, and across stocks at one time, it should have mean ≈ 0 and std ≈ 1.
- **Information coefficient (IC)**: the correlation between forecast and realized return, measuring skill.
- **Refined forecast / exceptional return**: φ = E{r|g} − E{r}.
- **Binary model**: return and forecast built from ±1 "unit bundles of uncertainty". Shared bundles are signal and unshared bundles are noise. Cov{r,g} = the number of shared bundles.
- **Forecast frequency vs. forecast horizon**: how often forecasts arrive vs. how long they keep predictive power. Treat the old forecast as a separate signal (Ch 13).
- **BLUE / m.v.u.e.**: the properties of the basic forecasting formula.
- **ARCH/GARCH**: volatility models with a long-run mean, mean reversion and shocks from recent returns. They are most useful for a few series (asset allocation, the market factor in a risk model), short horizons and option-like strategies.
- **Kalman filter**: a Bayesian prior-to-posterior updating scheme, of which the basic formula is a simple case.

## Mental Models
- "Alpha is volatility times IC times score." For a given signal, volatility and IC are constant, and only the score distinguishes one forecast from another.
- Think of a raw forecast as a mix of signal bundles and noise bundles. The IC is the share of signal scaled by the two volatilities, and you cannot unmix them; you can only weight the sum.
- When combining signals, pay for *new* information only. Correlation with signals you already have is redundancy.
- Neural nets work for nonlinear, stable, high signal-to-noise problems such as reverse-engineering bond ratings. Return forecasting is unstable and low signal-to-noise, and there is no definitive proof neural nets succeed there. Genetic algorithms suit one-off combinatorial searches, e.g. paring a portfolio to the best 50 names to track the S&P 500 (built in a weekend, about 48 CPU-hours, vs. about 6 person-months for a custom heuristic).

## Anti-patterns
- **Using historical average returns as the naive forecast**: they are noisy, unsuitable for new or changing stocks, and do not lead back to the benchmark.
- **Equal alphas for all buys (e.g. +1%/−1%)**: this ignores volatility, so the optimizer concentrates in low-risk names.
- **Changing correlations because of a return view**: e.g. replacing a 0.95 S&P 500/MIDCAP correlation with a negative one because you expect them to diverge. This confuses the conditional mean with the conditional covariance.
- **Letting sophisticated technique run you**: the insight gets buried and significance is lost. Test on random and simulated data.
- **Unchecked rankings**: first confirm whether rank 1 is the best or the worst.
- **Believing an IC > 0.20**.

## Reference Tables
**Table 10.2 – Converting a stock tip (residual vol 20%) to alpha = 20% × IC × score**

| Source IC | Very positive (z = 1) | Very, very positive (z = 2) |
|---|---|---|
| Great, 0.10 | 2.0% | 4.0% |
| Good, 0.05 | 1.0% | 2.0% |
| No information, 0.00 | 0.0% | 0.0% |

**Table 10.4 – Posterior volatility given IC (prior = 18%)**

| IC | 0.00 | 0.05 | 0.10 | 0.15 | 0.25 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|
| POST vol | 18.00 | 17.98 | 17.91 | 17.80 | 17.43 | 5.62 | 0.00 |

**Buy/sell list (Table 10.3)**: score ±1 and IC = 0.09, so α = ±0.09·ω. For example, IBM with ω = 30.32% gets ±2.73%, and 3M with ω = 13.41% gets ±1.21%.

## Worked Example
**Binary model, one asset and one signal**
- Return: r = 1.5 + Σ_{i=1..81} θ_i. Quarterly mean is 1.5% and vol is 9% (annual 6% / 18%).
- Forecast: g = 2 + (θ1+θ2+θ3) + Σ_{j=1..13} η_j. That is 3 signal bundles and 13 noise bundles, so Var{g} = 16 and Std{g} = 4.
- Cov{r,g} = 3, so IC = 3/(9·4) = **0.0833**.
- Refined forecast = (3/16)·(g − 2) = 9 × 0.0833 × z = **0.75·z**. It lies within ±0.75% two quarters in three and outside ±1.5% one quarter in twenty.

**Adding a second signal**
- g′ has 25 bundles, including 4 signal bundles (θ3–θ6), and shares θ3 and η10–η13 with g.
- So Cov{r,g′} = 4, giving IC_g′ = 4/(9·5) = 0.089. Cov{g,g′} = 5, giving ρ = 0.25.
- Alone, φ = 9 × 0.089 × z′ = 0.16·(g′ − E g′).
- Combined: IC\*_g = (0.0833 − 0.25 × 0.0889)/0.9375 = 0.065 and IC\*_g′ = 0.073. The raw-unit coefficients are c1 = 0.1467 and c2 = 0.1307.
- The combined IC is **0.1090**, compared with 0.0833 and 0.089 separately.

**Up/down market forecaster ("Charlie")**
- Monthly mean 0.50% and std 5.20%. Raw signals are +1 for up and −1 for down, with 2/3 of them up, so their mean is 1/3 and std 0.9428. The scores are therefore +0.707 for up and −1.414 for down.
- With IC = 0.075: up gives 0.50 + 5.20 × 0.075 × 0.707 = **0.78%**, and down gives 0.50 − 5.20 × 0.075 × 1.414 = **−0.05%**.
- The result is asymmetric because the market is expected to rise even with no information. Being "right two-thirds of the time" by always saying "up" is worthless.

## Key Takeaways
1. Default refinement: **α = ω · IC · z**. Score the signal, scale by skill, and give it units with volatility.
2. Zero skill gives zero alpha automatically, and higher-volatility assets get proportionally bigger alphas for the same score.
3. Combine signals with correlation-adjusted ICs. IC²_comb = (IC₁² + IC₂² − 2ρIC₁IC₂)/(1 − ρ²).
4. Without track records, use IC ≈ 0.05 (good), 0.10 (great) or 0.15 (world-class). Be suspicious above 0.20.
5. Keep risk forecasts unchanged when you have return views. The vol shrinkage is √(1 − IC²), under 1% relative at IC ≤ 0.15.
6. Treat stale (older) forecasts as additional signals.

## Connects To
- **Ch 6 (Fundamental Law)**: the IC is the skill term, and the binary model is reused here.
- **Ch 9**: growth-forecast shrinkage is an application of the basic formula.
- **Ch 11**: extends the rule to multiple assets, cross-sectional scores, factor forecasts and uncertain ICs.
- **Ch 12–13**: measuring the IC and IR, and the role of forecast horizon and old signals.
- **Ch 14**: alphas must be scaled correctly before optimization.
- **External**: BLUE / Gauss–Markov estimation; Bayesian updating; Kalman filters; Grinold (1994), "Alpha Is Volatility Times IC Times Score".
