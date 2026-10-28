# Chapter 11: Advanced Forecasting

## Core Idea
The basic forecasting formula carries over to many assets, but three practical refinements matter:
1. Rescale *cross-sectional* scores by volatility only when the signal's time-series volatility does **not** already scale with asset volatility.
2. A forecast for one factor implies nonzero forecasts for correlated factors. Never set them to zero.
3. Uncertainty about the IC means shrinking the alpha toward zero, using T·IC²/(1 + T·IC²).

## Frameworks Introduced
- **Time-series vs. cross-sectional scores**: the rule of thumb α_n = ω_n·IC·z_TS,n needs *time-series* scores (each stock's signal standardized over time). Managers usually have only *cross-sectional* scores (standardized across N stocks at one date).
  - **Case 1** (signal time-series std is the same for all stocks): z_TS = z_CS, so **α_n = ω_n·IC·z_CS,n**. Multiply by volatility.
  - **Case 2** (signal time-series std is proportional to the stock's volatility): **α_n = c_g·IC·z_CS,n**. Do *not* multiply by volatility. The signal already carries it. c_g is a per-signal constant.
  - How to decide: regress each stock's 60-month time-series signal std on its residual volatility across stocks. A strong fit means Case 2. A weak fit means Case 1.
  - Empirical verdict for BARRA signals on about 1,200 HICAP stocks (1994): DDM, estimate change, estimate revision, relative strength and residual reversal are all Case 2. Sector momentum (many stocks share the same score) is Case 1.
- **Testing alpha scaling via risk buckets**: with correct scaling and uncorrelated residuals, each asset's expected contribution to active residual variance is equal, ∝ IC²·E{z²}/(4λ²). Bucket assets by volatility (or any attribute), with enough names per bucket to control sampling error, and compare their risk contributions. Unequal contributions mean the volatility scaling is wrong, or that you implicitly assumed different ICs for different buckets.
- **Multiple forecasts per stock**: the Ch 10 single-asset combination formulas apply *asset by asset* only under restrictive assumptions:
  - each source has the same IC for all assets;
  - every source has the same cross-asset signal correlation matrix;
  - Corr{g_i,n, g_j,n} = ρ_ij, a constant.
  
  Otherwise you must impose other structure, e.g. a factor model. An empirical alternative is to build K factor portfolios, one per signal and each controlling for the other K − 1, then choose the weights on them that maximize IR.
- **Factor forecasts: "do not set other factor forecasts to zero"**. If you have a signal g₁ for factor 1, then every factor correlated with factor 1 inherits a forecast: E{b_j|g₁} = [Cov{b_j,b₁}/Var{b₁}]·E{b₁|g₁}.
  - Why it works: g₁ is informative about b₁, and b₁ is informative about every b_j that covaries with it.
  - Same logic for assets: a forecast for asset 1 alone implies forecasts for the other assets in proportion to their regression coefficient on asset 1. This links to Black–Litterman.
- **Bayesian shrinkage for uncertain ICs**: add the prior b = 0 as an extra weighted "observation" in the refining regression. This is equivalent to maximum likelihood combining the data likelihood with the prior. It shrinks the coefficient by about T·IC²/(1 + T·IC²).
  - With several signals, use each signal's *marginal* R². That is the total R² minus the R² with that coefficient set to zero, and the marginals sum to the total. This rewards parsimony: a new signal with little marginal power gets shrunk hard.

## Key Formulas
- **Multi-asset basic formula (11.1)**: E{r|g} = E{r} + Cov{r,g}·Var⁻¹{g}·(g − E{g}). r is an N-vector and g a K-vector, so K/N signals per asset.
- **One signal per asset (11.2)**: α_n = ω_n·IC·z_TS,n, assuming the same IC for all assets and that r_n is correlated with g_m only through g_n.
- **Scores**:
  - Time-series: z_TS,n = (g_n(t) − Mean_t{g_n})/Std_t{g_n}.
  - Cross-sectional: z_CS,n = (g_n − Mean_n{g})/Std_n{g}.
- **Case 1 (11.5–11.7)**: Std_TS{g_n} = c₁ for all n, so z_TS = z_CS and α_n = ω_n·IC·z_CS,n.
- **Case 2 (11.8–11.14, constant reconstructed)**: Std_TS{g_n} = c₂·ω_n, so α_n = ω_n·IC·g_n/(c₂·ω_n) = IC·g_n/c₂ = **c_g·IC·z_CS,n**.
  - c₂ is estimated cross-sectionally as Std{g_n/ω_n}.
  - c_g ≈ Std_CS{g}/c₂ is independent of n. It has units of return, like a typical residual volatility.
- **Pure cross-sectional version (11.16)**: α_n ≈ Std_CS{θ}·IC·z_CS,n. This is equivalent to Case 2 for a single date. It is unreliable alone because the industries that explain cross-sectional returns change every month (Internet stocks this month, health care next, banks after).
- **Factor-forecast spillover (11.22–11.25)**:
  - Assume g₁ = a·b₁ + noise·Z. Then Cov{b_j,g₁} = a·Cov{b_j,b₁}.
  - Therefore E{b_j|g₁} = (Cov{b_j,b₁}/Var{b₁})·E{b₁|g₁}, assuming E{b_j} = 0.
- **General orthogonalized form (11A.10–11A.12)**: write g − E{g} = A·y, with y standardized and uncorrelated (Var{y} = I). Then φ = Std{r}·Corr{r,y}·y. This is still "volatility × IC × score", with the IC now a matrix.
- **Refining regression (11.26–11.27)**: θ(t) = b·g(t) + ε(t). With zero means, b = Cov{θ,g}/Var{g} = IC·ω/Std{g}.
- **Shrinkage (11.31–11.32, reconstructed and verified against Table 11.4)**: b′ = b·T·R²/(1 + T·R²) ≈ **b·T·IC²/(1 + T·IC²)**, with prior b = 0. The exact form (11.31) involves the expected R² of the no-prior regression. Since R² ≈ IC² ≪ 1, the approximation is used.
- **Error decomposition (11A.24, reconstructed approximately)**: Var{Δb}/b² ≈ 1/(T·IC²) [from the IC] + O(1/T) [from ω and Std{g}]. With IC ≪ 1, uncertainty in the IC dominates, so shrinking b is effectively shrinking the IC.

## Key Concepts
- **Time-series score**: the standardization of one asset's signal over time. This is what the theory requires.
- **Cross-sectional score**: the standardization across assets at one date. This is what managers usually have.
- **c_g**: a signal-specific scaling constant converting IC·z_CS into return units in Case 2.
- **Alpha scaling test**: equal expected risk contribution per asset or bucket under correct scaling.
- **Factor-forecast spillover**: the implied forecasts for unforecast factors via their covariance with the forecast factor.
- **Shrinkage factor**: T·IC²/(1 + T·IC²). It is the Bayesian discount for estimation error in skill.
- **Marginal R²**: a signal's incremental explanatory power. It drives per-signal shrinkage in multi-signal models.
- **Industry-neutralized signal**: a signal defined relative to its industry. You cannot do this to sector momentum, because the signal would become zero.

## Mental Models
- Before multiplying cross-sectional scores by volatility, ask: "Does this signal's own volatility already scale with stock volatility?" For most stock-level signals (valuation, revisions, momentum, reversal) it does, so use IC × z_CS. For group-level signals (sector momentum) it does not, so use ω × IC × z_CS.
- Treat a factor forecast like a CAPM market forecast: a view on the market implies views on every stock through beta. A view on B/P likewise implies views on every factor correlated with B/P.
- IC estimates are noisy. Think of T·IC² as the "evidence count". You need T·IC² ≫ 1 before the data dominate the prior, so at IC = 0.05 you need hundreds of months.

## Anti-patterns
- **Multiplying cross-sectional scores by residual volatility by reflex**: in Case 2 it double-counts volatility and lowers IR, e.g. DDM IR 1.31 → 1.19 and residual reversal 2.51 → 2.18.
- **Setting forecasts to zero for factors you "have no view on"**: this throws away the information implied by factor covariance.
- **Estimating alphas from a single cross-section**: what you can forecast *consistently over time* requires both time-series and cross-sectional data.
- **Using backtested coefficients at face value**: even IC = 0.10 over 10 years keeps only about 55% of the estimated coefficient.
- **Adding marginal signals freely**: low marginal R² means heavy shrinkage. Parsimony pays.

## Reference Tables
**Table 11.1 – Signal time-series volatility vs. asset residual volatility (about 1,200 stocks, 60 months to Dec 1994)**

| Signal | R² | t(b) |
|---|---|---|
| DDM | 0.37 | 19.3 |
| Estimate change | 0.34 | 18.0 |
| Estimate revision | 0.31 | 17.0 |
| Relative strength | 0.72 | 54.3 |
| Residual reversal | 0.77 | 62.2 |
| Sector momentum | 0.01 | −3.8 |

**Table 11.2 – Backtest IR: cross-sectional score alone vs. volatility-scaled**

| Signal | IR with IC·z_CS | IR with ω·IC·z_CS |
|---|---|---|
| DDM | 1.31 | 1.19 |
| Estimate change | 1.92 | 1.87 |
| Estimate revision | 3.55 | 3.32 |
| Relative strength | 1.93 | 1.93 |
| Residual reversal | 2.51 | 2.18 |
| Sector momentum | 1.91 | **2.10** |

(Columns realigned from garbled text. They are consistent with the conclusion that only sector momentum benefits from volatility scaling. All signals except sector momentum were industry-neutralized.)

**Table 11.3 – B/P information spilled to other factors (BARRA US-E2, May 1990 – Apr 1995)**

| Strategy | IR |
|---|---|
| A: bet only on B/P | 3.26 |
| B: B/P information used for B/P plus other risk indices | 3.42 |
| C: B/P information used only on *other* factors | 1.57 |

Expect IR_B² ≈ IR_A² + IR_C². The text says this holds roughly.

**Table 11.4 – Shrinkage factor T·IC²/(1 + T·IC²)**

| Months T | IC = 0.00 | IC = 0.05 | IC = 0.10 |
|---|---|---|---|
| 36 | 0.00 | 0.08 | 0.26 |
| 60 | 0.00 | 0.13 | 0.38 |
| 90 | 0.00 | 0.18 | 0.47 |
| 120 | 0.00 | 0.23 | 0.55 |
| 240 | 0.00 | 0.38 | 0.71 |

## Worked Example
**Shrinking a backtested coefficient**
- A signal with IC = 0.05 over 60 months has T·IC² = 60 × 0.0025 = 0.15. The shrinkage is 0.15/1.15 = **0.13**, so you keep only 13% of the estimated coefficient.
- At IC = 0.10 over 240 months: 2.4/3.4 = **0.71**.
- *Problem 3*: with R² = 0.05 after 120 months, T·R² = 6, so b′ = (6/7)·b ≈ **0.86·b**, about a 14% shrink.

**Factor spillover**: suppose your signal implies E{b_B/P|g} = +1% and Cov{b_Size, b_B/P}/Var{b_B/P} = 0.3. Then you should forecast E{b_Size|g} = +0.3%, not 0. This is an illustrative application of Eq. 11.25. The chapter's evidence for the idea is Table 11.3.

## Key Takeaways
1. With cross-sectional scores, test whether signal volatility scales with stock volatility. If it does (most stock-level signals), α ∝ IC·z_CS. If it does not (group signals such as sector momentum), α = ω·IC·z_CS.
2. Validate alpha scaling by checking that volatility buckets contribute equal risk in the optimal portfolio.
3. A view on one factor or asset implies views on all correlated factors or assets, via regression coefficients on the forecast factor.
4. Shrink refined forecasts by T·IC²/(1 + T·IC²). Shrinkage is large even for good signals over long histories.
5. In multi-signal models, shrink each signal by its marginal R², which favors parsimonious signal sets.
6. Two equal-IC signals that both scale with volatility still get different weights in proportion to their c_g constants.

## Connects To
- **Ch 7 (APT)**: factor forecasts; m_k should incorporate the spillover from forecast factors.
- **Ch 10**: the basic formula, the rule of thumb and multi-signal combination extended here.
- **Ch 12 (Information Analysis)**: the backtest/IR methodology behind Tables 11.2–11.3.
- **Ch 14 (Portfolio Construction)**: alpha scaling and equal risk per asset.
- **External**: Black–Litterman (a view on a subset implies a full posterior); Connor (1997), "Sensible Return Forecasting"; Bayesian ridge-type shrinkage.
