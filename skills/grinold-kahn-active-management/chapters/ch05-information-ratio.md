# Chapter 5: Residual Risk and Return: The Information Ratio

## Core Idea
The **information ratio** (IR, the best achievable ratio of alpha to residual risk) defines your opportunity set, a straight-line **residual frontier**. **Residual risk aversion** λ_R defines your preferences. Together they fix your optimal aggressiveness, ω* = IR/(2λ_R), and your value added, VA* = IR²/(4λ_R). Every investor, whatever their risk aversion, wants the highest-IR strategy. They differ only in how aggressively they implement it.

## Frameworks Introduced
- **The "can opener" (assume success)**: posit an ex ante IR, then work out how to use it. Calibrate it from hope and humility: target residual risk of 5–6% and hoped-for alpha of 3–4% give IR ≈ 3.5/5.5 = 0.64 (range 0.5–0.8).
- **Residual frontier**: α_P = IR·ω_P, a line through the origin (benchmark and cash sit at the origin). IR is the manager's "budget constraint". Along the frontier you can raise alpha only by raising residual risk.
- **Preferences meet opportunities**: maximize VA = α_P − λ_R·ω_P² subject to α_P ≤ IR·ω_P, so VA(ω) = IR·ω − λ_R·ω². The optimum is the tangency of the frontier with a parabolic iso-VA curve.
  - When to use: to set the target tracking error for a product, or to back out an implied risk aversion from a chosen risk level.
  - How: ω* = IR/(2λ_R). Or invert it: λ_R = IR/(2ω*).
  - Why it works: IR is independent of aggressiveness. Scaling active positions scales α and ω together (until constraints such as no-shorting bind; see Ch 15).
- **Classification**: IR = 0.50 **good** (top quartile), 0.75 **very good**, 1.00 **exceptional**. λ_R = 0.05 aggressive, 0.10 moderate, 0.15 restrained.
- **Forecast alphas directly**: you don't need to forecast total returns and then decompose them. Recipe: sort stocks into 5 bins (strong buy / buy / hold / sell / strong sell) with alphas +2%, +1%, 0, −1%, −2%. Compute the benchmark alpha α_B = Σ h_B·α. If it isn't zero, set α_n ← α_n − β_n·α_B, which makes the alphas **benchmark-neutral** and leads an unconstrained optimizer to β = 1.
- **Alpha scaling check**: IR is linear in the alphas. If the raw alphas imply IR₀ = 2.46, multiply them by 0.75/2.46 to get a believable IR = 0.75 before optimizing (λ ≈ 0.10).

## Key Formulas
- Ex post regression: r_P(t) = α_P + β_P·r_B(t) + ε_P(t). Residual θ_P(t) = α_P + ε_P(t).
- Ex ante alpha: α_n = E{θ_n}. It has the portfolio property: α_P = Σ h_P(n)·α_n = Σ h_PA(n)·α_n. Benchmark-neutral means α_B = 0, and α_F = 0.
- IR_P = α_P/ω_P. **IR = max_P IR_P** (achieved by portfolio Q, or equivalently A).
- Ex post: IR ≈ t-stat(α)/√Y, where Y is years of data.
- Horizon: IR_T = IR_annual·√T. The quarterly IR is ½ the annual IR and the monthly IR is 1/√12 of it. Always quote annual IR.
- Value added: VA = α_P − λ_R·ω_P² (also called the certainty-equivalent return).
- **Optimal residual risk**: ω* = IR/(2λ_R).
- **Optimal value added**: VA* = IR²/(4λ_R) = IR·ω*/2.
- Implied aversion: λ_R = IR/(2ω*). For IR = 0.64 and ω = 5.5%, λ_R = 0.058 and VA* = 1.77%.
- **Portfolio A** (characteristic portfolio of α): h_A = V⁻¹α/(αᵀV⁻¹α), with α_A = 1, β_A = 0, and σ_A = ω_A = 1/IR. So **IR = √(αᵀV⁻¹α)**.
- **α = IR·MCRR_Q**: in the optimal portfolio, alphas are proportional to marginal contributions to residual risk. (Check: if IR = 0.5, each MCRR should be about 2× its alpha, i.e. alpha ≈ ½·MCRR.)
- IR_P = IR·Corr{θ_P, θ_Q}. Any portfolio's IR is the maximum IR times its residual correlation with Q.
- h_Q = β_Q·h_B + α_Q·h_A. SR_Q² = SR_B² + IR² [reconstructed from f = β·f_B + α and αᵀV⁻¹β = 0]. IR = SR_Q·ω_Q/σ_Q.
- Optimal holdings: h_P* = β_P·h_B + (IR/(2λ_R))·(h_A/ω_A). Beta is irrelevant to the residual problem.
- β = 1 frontier: σ_P² = σ_B² + ((f_P − f_B)/IR)². The benchmark is its hinge (minimum-risk β = 1 portfolio).
- With no active cash and no active beta: optimal residual h_PR = (IR/(2λ_R))·h_Y, where Y combines A and C's residual. IR_Y ≤ IR, so constraints reduce opportunity. Portfolio A has zero cash only if α_C = 0.

## Key Concepts
- **Alpha (ex ante)**: a forecast of residual return. **Alpha (ex post)**: the average realized residual return, for keeping score.
- **Information ratio**: annualized residual return per unit of annualized residual risk. Ex post it measures achievement, ex ante opportunity (Treynor–Black's "appraisal ratio").
- **Residual frontier**: the line α = IR·ω that bounds the attainable (ω, α) pairs.
- **Residual risk aversion λ_R**: converts residual variance into lost alpha. At ω = 5%, the loss is 3.75%, 2.5%, and 1.25% for λ_R = 0.15, 0.10, 0.05.
- **Aggressiveness**: the chosen level of residual risk ω*.
- **Benchmark-neutral alphas**: alphas whose benchmark-weighted sum is zero.
- **Portfolio A / Q / Y**: max-IR zero-beta alpha portfolio; max-Sharpe fully invested portfolio (= benchmark + A); max-IR portfolio with zero active cash and zero active beta.

## Mental Models
- Think of **IR as the slope of your opportunity line and λ_R as the curvature of your preferences**. Risk is set where they touch.
- **Doubling IR doubles the right risk and quadruples value added. Doubling λ_R halves both.**
- Use the **MCRR ∝ alpha check** on any hand-built portfolio. If marginal contributions to residual risk don't line up with alphas, the portfolio doesn't express your views.
- Think of the **IR distribution as centered on zero** (active management is a zero-sum game before fees). Aim for top quartile (0.5) ex ante.
- **Precision isn't the point**: alphas need only point the right way. Crude 5-bin alphas are fine.

## Anti-patterns
- **Using raw, over-optimistic alphas**: they imply IRs of 2+ and push the optimizer to extreme positions. Rescale them to a sensible IR.
- **Non-benchmark-neutral alphas**: they cause unintended beta or benchmark-timing bets.
- **Comparing IRs across horizons without annualizing**: monthly IRs look √12 times smaller.
- **Low IR plus restrained implementation**: IR = 0.5 with λ_R = 0.15 gives ω* = 1.67% and VA = 0.42%, which probably doesn't justify an active fee.
- **Setting risk from a mis-estimated IR**: overstating IR makes you over-aggressive, and the VA lost can wipe out the gain (see example).
- **Assuming IR stays constant at any risk level under tight constraints**: no-short and other constraints erode IR as aggressiveness rises.

## Reference Tables
Generic ex ante IR distribution (before fees):

| Percentile | IR |
|---|---|
| 90 | 1.0 |
| 75 | 0.5 |
| 50 | 0.0 |
| 25 | −0.5 |
| 10 | −1.0 |

Optimal residual risk ω* = IR/(2λ_R):

| IR \ λ_R | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| 1.00 | 10.00% | 5.00% | 3.33% |
| 0.75 | 7.50% | 3.75% | 2.50% |
| 0.50 | 5.00% | 2.50% | 1.67% |

Value added VA* = IR²/(4λ_R):

| IR \ λ_R | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| 1.00 | 5.00% | 2.50% | 1.67% |
| 0.75 | 2.81% | 1.41% | 0.94% |
| 0.50 | 1.25% | 0.63% | 0.42% |

Empirical IRs (Kahn–Rudd; US equity mutual funds 1991–93, n = 300; institutional 1995–96, n = 367):

| Pctl | Eq MF before | Eq MF after | Eq Inst before | Eq Inst after | Bond MF before | Bond MF after | Bond Inst before | Bond Inst after |
|---|---|---|---|---|---|---|---|---|
| 90 | 1.33 | 1.08 | 1.25 | 1.01 | 1.14 | 0.50 | 1.81 | 1.29 |
| 75 | 0.78 | 0.58 | 0.63 | 0.48 | 0.50 | −0.22 | 0.89 | 0.38 |
| 50 | 0.32 | 0.12 | −0.01 | −0.15 | −0.11 | −0.86 | 0.01 | −0.57 |
| 25 | −0.08 | −0.33 | −0.56 | −0.72 | −0.61 | −1.50 | −0.62 | −1.37 |
| 10 | −0.47 | −0.72 | −1.03 | −1.25 | −1.22 | −2.21 | −1.50 | −2.41 |

Annual active risk:

| Pctl | Equity MF | Equity Inst | Bond MF | Bond Inst |
|---|---|---|---|---|
| 90 | 9.87% | 9.49% | 3.44% | 1.89% |
| 75 | 7.00% | 6.47% | 2.01% | 0.98% |
| 50 | 4.76% | 4.39% | 1.33% | 0.61% |
| 25 | 3.66% | 2.85% | 0.96% | 0.41% |
| 10 | 2.90% | 1.93% | 0.74% | 0.26% |

The median equity active risk is 4–5%. Bond managers take far less risk, and fees hurt bond IRs much more (Kahn 1998: bond managers need to take more risk).

## Worked Example
**Four-stock example** (benchmark 25% each; alphas +1.50%, −2.00%, +1.75%, −1.25%). Portfolio P has active weights +10%, −15%, +15%, −10%, so α_P = 0.15 + 0.30 + 0.2625 + 0.125 = 0.84% with active risk 2.04% (IR ≈ 0.41). Portfolio L has active weights 1.5× those of P (+15%, −22.5%, +22.5%, −15%), so α ≈ 1.26% and risk ≈ 3.06%. **The IR is unchanged.**

**Sizing and the cost of misjudging IR**: with λ_R = 0.12 and IR = 0.60, ω* = 0.6/0.24 = 2.5% and VA* = 0.36/0.48 = 0.75%. If the true IR is 0.30, running at 2.5% gives VA = 0.30·2.5 − 0.12·6.25 = 0. The correct ω* = 1.25% gives VA = 0.09/0.48 ≈ 0.19%, so about 0.19%/yr is lost.

**Implied aversion**: IR = 0.5 with target ω = 4% gives λ_R = 0.5/8 = 0.0625. A required VA of 2.5% at λ_R = 0.10 needs IR = √(4·0.1·2.5) = 1.0 and ω = 5%.

**Iso-VA check**: at λ_R = 0.10, (α = 4.1%, ω = 4%) gives 4.1 − 0.1·16 = 2.5%, equal to a riskless 2.5%.

## Key Takeaways
1. IR is the single key number in active management. Everyone prefers the highest-IR strategy.
2. Set tracking error to ω* = IR/(2λ_R). With IR around 0.5–0.75 and λ_R around 0.1, that is about 2.5–4%.
3. Value added grows with IR². Raising IR matters far more than turning up risk.
4. A top-quartile IR is about 0.5 before fees. 0.75 is very good, 1.0 exceptional. Be skeptical of claims above 1.
5. Scale and benchmark-neutralize alphas before optimizing. Check alpha ∝ MCRR afterwards.
6. Quote annual IRs, since IR grows with √horizon.
7. Constraints (no active cash, no shorting) lower the effective IR.

## Connects To
- **Ch 2**: portfolios A, Q, and C and the characteristic-portfolio machinery. SR_Q² = SR_B² + IR².
- **Ch 4**: the value-added objective (stock selection term).
- **Ch 6**: the Fundamental Law breaks IR down into IC·√BR.
- **Ch 12**: information analysis, i.e. estimating ex ante IR and IC from data.
- **Ch 14**: optimizer calibration with λ_R and alpha scaling.
- **Ch 15**: long/short, and the IR lost to the no-short constraint.
- **Ch 17**: ex post IR, t-statistics, and style analysis.
- **Treynor–Black (1973)**: the appraisal ratio and active/passive mixing.
