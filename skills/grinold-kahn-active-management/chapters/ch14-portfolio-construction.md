# Chapter 14: Portfolio Construction

## Core Idea
Portfolio construction must maximize after-cost value added — `α_P − λ_A·ψ_P² − TC` — using inputs (alphas, covariances, costs, risk aversion) that are noisy and often biased. Most constraints and ad hoc rules are really indirect defenses against bad inputs; attack the cause directly with **alpha analysis** (scale, trim, neutralize) and good risk/cost estimates, then use a powerful optimizer (quadratic programming) with as few constraints as possible.

## Frameworks Introduced
- **Constraints ⇔ modified alphas**: any portfolio construction procedure (however constrained) that produces active holdings h_PA, active risk ψ_P and IR can be reproduced by an *unconstrained* mean/variance optimization with modified alphas `α′ = 2λ_A·V·h_PA` and active risk aversion `λ_A = IR/(2ψ_P)`.
  - When to use: to see what your constraints are really doing to your information — compare α′ to α.
  - Why it works: the modified alphas satisfy the unconstrained first-order conditions. In the MMI example constraints shrank alpha std from 2.00% to 0.57%, i.e., a large implicit cut in IC that should be explicit rather than "hidden under a rug of optimizer constraints".
- **Alpha analysis (preprocessing)** — perform neutralization first, then scaling and trimming as the final step:
  1. **Scale**: `Std(α) ≈ volatility · IC` (residual risk × IC). IC 0.05, residual risk 30% ⇒ scale 1.5%: ~⅔ of alphas within ±1.5%, ~5% beyond ±3%. Rescale if the input scale is wrong. Alternative: compute the IR implied by the alphas (IR₀) and rescale by IR/IR₀.
  2. **Trim**: examine alphas beyond **3 × scale**; zero those based on questionable data, pull genuine ones in to ±3 × scale. Extreme alternative: force alphas into a normal distribution (uses ranks only; recheck neutrality and scale afterwards).
  3. **Neutralize** (decide *a priori*, not by trying all variants): benchmark-neutral (α_B = 0 ⇒ no benchmark timing, optimal beta = 1), cash-neutral, industry-neutral (subtract cap-weighted industry average alpha), risk-factor-neutral (remove alphas on factors you can't forecast).
- **Transaction-cost no-trade band**: trade stock n only if its marginal contribution to value added falls outside `[−SC_n, PC_n]`.
  - Band width = PC + SC (e.g., 0.50% + 0.75% = 1.25%): after a purchase you sit at the top of the band; alpha must fall 1.25% before you sell.
  - Leland's dynamic result (60/40 stock/bond): optimal policy has a no-trade region; when outside, **trade back to the boundary, not the target** — cuts turnover and costs roughly in half with essentially no change in risk.
- **Four generic construction techniques** (screens, stratification, linear programming, quadratic programming) — judged by alpha, active risk, and transaction costs.
  - **Screens**: rank by alpha, take top N (e.g., 50), equal- or cap-weight. Rebalancing variant: of 200 followed stocks, top 40 = buy, next 60 = hold, bottom 100 = sell; tune 40/60/100 to regulate turnover. Simple, robust to wild alphas (ranks only); ignores alpha magnitudes, inherits biases (e.g., no utilities), fragmentary risk control — often riskier than managers think.
  - **Stratification** ("glorified screening"): screen within categories (e.g., 10 sectors × 3 sizes = 30 cells) and match benchmark weights per cell. Removes cross-category alpha biases; risk control only as good as the categories chosen.
  - **Linear programming** ("space-age stratification"): maximize alpha − costs, staying close to benchmark on many non-exclusive risk characteristics, with turnover and position limits. Hard to target a fixed number of names; don't control a dimension your alphas intend to bet on.
  - **Quadratic programming**: explicitly trades off alpha, risk, and costs; most consistent in tests — *if* covariance estimates are good.
- **Dispersion management** across separate accounts: control dispersion from differing betas/factor exposures; accept some asset-level dispersion because eliminating it costs transaction costs or return.

## Key Formulas
- **Objective**: `max  α_P − λ_A·ψ_P² − TC` (value added less transactions costs).
- **Modified alphas**: `α′ = 2·λ_A·V·h_PA`; risk aversion from intuition: `λ_A = IR / (2·ψ_P)`. IR 0.5, target 5% active risk ⇒ λ_A = 0.05 (check units: percent vs. decimal).
- **Implied IR of an alpha set**: `IR₀² = αᵀ·V⁻¹·α`; with λ_A = 0.5 and no constraints, optimal h = V⁻¹α and portfolio alpha = IR₀². Rescale alphas by IR/IR₀.
- **Benchmark neutralization**: `α_n ← α_n − β_n·α_B`. Benchmark- and cash-neutral: `α⁺ = α − c_B·V·h_B − c_C·V·h_C` with c_B, c_C chosen so α⁺_B = 0 and α⁺_C = 0.
- **Differential risk aversion**: `VA = α_P − λ_CF·x_PAᵀ·F·x_PA − λ_SP·h_PAᵀ·Δ·h_PA` (higher specific-risk aversion shrinks single-name bets and reduces dispersion).
- **Missing alphas** (N₁ = stocks with forecasts, N₀ without): `H₁ = Σ_{N₁} h_B,n`, `ᾱ = Σ_{N₁} h_B,n·α_n / H₁`; set `α_n − ᾱ` for N₁ and `0` for N₀ (benchmark-neutral). Stocks outside the benchmark: add them with zero benchmark weight.
- **Annualized transaction cost** = round-trip cost / holding period (years).
- **Marginal contribution to value added**: `MCVA_n = α_n − 2·λ_A·ψ_P·MCAR_n`; hold (no trade) while `−SC_n ≤ MCVA_n ≤ PC_n`.
- **Loss from misestimated market volatility** (cash vs. market, true σ, estimate σ̂): `Loss/VA* = (1 − σ²/σ̂²)²` (reconstructed from the setup; matches the book's statements). General: `Loss = (1/4λ_A)·(U⁻¹α − V⁻¹α)ᵀ·V·(U⁻¹α − V⁻¹α)` for estimated covariance U vs. true V.
- **Factor/specific alpha split**: `H = (XᵀΔ⁻¹X)⁻¹XᵀΔ⁻¹` (factor portfolios, H·X = I); factor alphas `α_F = H·α`; `α_CF = X·α_F`, `α_SP = α − α_CF` (= residual of a regression of α on X weighted by 1/specific variance). To impose target active factor exposures x*: `α′ = 2λ_A·V·Hᵀ·x* + α_SP`.
- **Tracking-error bound with costs** (relative to the zero-cost optimum Q): `ψ ≤ √(TC / (2λ_A))`, TC = cost of moving from initial portfolio to Q.
- **Expected dispersion** of N accounts each with tracking error ψ vs. the composite: `E{max − min} ≈ 2·ψ·Φ⁻¹(0.5^(1/N))`.

## Key Concepts
- **Implementation**: efficient translation of research into portfolios; good implementation can't save poor research, poor implementation can foil good research.
- **Alpha scale**: cross-sectional std of alphas, ≈ volatility × IC.
- **Neutralization**: removing undesired bets (benchmark, cash, industry, factor) from alphas; not uniquely defined.
- **No-trade band**: alpha range of width PC + SC within which new information doesn't justify trading.
- **Amortization of costs**: comparing point-in-time costs with annual alpha/risk over the expected holding period.
- **Dispersion**: max − min return across separately managed accounts in the same strategy; client-driven (constraints) or manager-driven (inattention, costs).
- **Dual-benchmark optimization**: penalizing risk vs. both benchmark and composite; reduces dispersion only by sacrificing return.
- **Diversified portfolio**: minimum risk for its factor exposures; any such portfolio is a combination of factor portfolios (h = Hᵀx).

## Mental Models
- Think of constraints as **alpha shrinkers**: any constraint set equals a smaller, reshaped alpha vector. Ask whether you'd knowingly make that alpha change.
- Use QP when risk estimates are good; the optimizer exploits noise in the covariance matrix ("error maximization"), so invest in a good risk model rather than abandoning optimization.
- Think of transactions costs as adding a second knob: without costs, alpha scale errors can be offset by risk aversion; with costs, alpha scale must be right in absolute terms.
- Revise whenever new information arrives *if* costs and alphas are correctly specified; infrequent revision is a crude cure for overreactive alphas and underestimated costs.

## Anti-patterns
- **Hiding alpha problems inside optimizer constraints** instead of scaling/trimming/neutralizing alphas directly.
- **Choosing neutralizations by backtest** (trying all, keeping the best): decide a priori.
- **Underestimating costs + noisy fast alphas + daily rebalancing**: churn, higher costs, lower alpha.
- **Underestimating volatility in optimization**: errors > 3 points hurt significantly; estimating 12% when true is 17% gives *negative* value added.
- **Alternative risk measures** (semivariance, downside, shortfall) for institutional stock selection: higher-moment forecasts are mostly noise; much more effort, no better (often worse) portfolios. Exception: options/currency protection.
- **Driving dispersion to zero**: costs return; some dispersion is optimal.

## Reference Tables
Muller (1993) test — identical alphas (IC = 0.1, 500 stocks ⇒ target IR = 0.1·√500 = 2.24), no costs, ex post IRs over 4 rebalances × 3 risk levels:

| Method | Average | Std dev | Max | Min |
|---|---|---|---|---|
| Screen I (top N equal-weight, N = 50/100/150) | 0.86 | 0.27 | 1.43 | 0.50 |
| Screen II (top N cap-weight) | 1.10 | 0.79 | 2.24 | −0.53 |
| Strat (top J per BARRA industry, 55 ind.) | 1.27 | 0.89 | 2.82 | 0.33 |
| QP (full investment, no shorts, ≤10% per name) | 1.88 | 0.40 | 2.51 | 0.98 |

QP is highest and most consistent; none reach 2.24 because constraints shrink effective IC.

Other numbers: QP on 500 stocks needs 500 volatilities + 124,750 correlations. Beta range 0.9–1.1 across accounts with a 35% market year ⇒ 7% dispersion. Observed real-world dispersion: 23% over a year among five accounts; +15% vs. −9% vs. S&P in one year. Simulated convergence of five 100-name, ~3%-TE accounts took roughly 5 years.

## Worked Example
**Amortizing transactions costs.** Both stocks earn 4%/yr alpha; round-trip $0.75 each way on $100.
- Stock 1 (+$2 in 6 months, repeat 4 times over 2 years): costs $0.75+$1.50+$1.50+$1.50+$0.75 = $6; gain $8; profit $2 ⇒ 1%/yr. Annualized cost = 3%.
- Stock 2 (+$8 over 24 months): costs $1.50; profit $6.50 ⇒ 3.25%/yr. Annualized cost = 0.75%.
Rule: annualized cost = round-trip cost / holding period in years (6 months ⇒ ×2; 24 months ⇒ ÷2).

**Tracking-error bound.** λ_A = 0.10, round-trip cost 2%, 10% turnover to reach Q ⇒ TC = 0.2 (percent): ψ ≤ √(0.2/0.2) = 1%.

## Key Takeaways
1. Preprocess alphas: neutralize (a priori choices), then scale to volatility × IC and trim at 3× scale.
2. Set active risk aversion from intuition: λ_A = IR/(2ψ_P).
3. Amortize costs over the expected holding period; trade only when MCVA leaves the [−SC, PC] band; rebalance to the band edge.
4. Quadratic programming with a good risk model beats screens, stratification and LPs on average and in consistency.
5. Mean/variance is sufficient for institutional active management; alternative risk measures add effort and error.
6. Control factor/beta-driven dispersion across accounts; accept asset-level dispersion; with changing alphas, accounts converge over time.

## Connects To
- **Ch 5**: λ = IR/(2ω) optimality link between IR, risk aversion and risk.
- **Ch 3**: structural risk models — quality of V drives QP performance.
- **Ch 10–11**: alpha = volatility · IC · score sets the natural alpha scale; inferring missing factor alphas.
- **Ch 13**: half-life vs. costs drives after-cost IR (Fig. 14.1).
- **Ch 15**: the long-only constraint as the most consequential constraint.
- **Ch 16**: estimating and reducing transactions costs; turnover.
- **External**: Michaud "error maximization"; Black–Litterman / shrinkage alphas; Leland no-trade regions.
