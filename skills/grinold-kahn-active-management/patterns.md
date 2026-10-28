# Patterns & Techniques — Active Portfolio Management

## Forecast Refinement (α = Volatility · IC · Score)
**When to use**: Turning any raw signal into an alpha usable by an optimizer (Ch 10).
**How**: Standardize the signal to a score z = (g − mean)/std. Estimate the IC (or use 0.05). Set α = ω · IC · z, with ω the residual volatility. For buy/sell lists, α = ±IC·ω. For a stock tip, α ≈ 20% × IC × score.
**Trade-offs**: Gives equal risk per asset. Flat alphas (e.g. ±1%) instead push the optimizer into low-volatility names. Use the cross-sectional variant α = c·IC·z_CS when the signal's dispersion doesn't scale with volatility (Ch 11).

## Multi-Signal Combination
**When to use**: Several correlated signals (Ch 10, 11).
**How**: Two signals: IC*₁ = (IC₁ − ρIC₂)/(1 − ρ²). Combined IC² = (IC₁² + IC₂² − 2ρIC₁IC₂)/(1 − ρ²). K signals: φ = ω·ICᵀρ⁻¹z. Or build an information-efficient portfolio per signal and weight them to maximize IR.
**Trade-offs**: Correlated signals add little: two signals with correlation γ give IC² ×2/(1+γ). Use marginal R² to keep the model parsimonious.

## Alpha Analysis Before Optimization
**When to use**: Every rebalance, before portfolio construction (Ch 14).
**How**: (1) Benchmark-neutralize: α − β·α_B. (2) Neutralize any factors you do not intend to bet on, using H = (XᵀΔ⁻¹X)⁻¹XᵀΔ⁻¹. (3) Fill missing alphas: covered names α − ᾱ_cap, uncovered names 0. (4) Scale: multiply by IR_target/IR₀, where IR₀ = √(αᵀV⁻¹α). (5) Trim outliers beyond ~3× the alpha scale.
**Trade-offs**: Alpha adjustment does the work of constraints more transparently. Every constraint is equivalent to a modified alpha.

## Mean-Variance Portfolio Construction (QP)
**When to use**: The default construction method (Ch 14).
**How**: Maximize α_P − λ_A·ψ_P² − TC. λ_A = IR/(2ψ_target). Check optimality: α = 2λψ·MCAR, and hold an asset while −SC ≤ MCVA ≤ PC.
**Trade-offs**: In the Muller test QP captured the most IR (1.88, vs 1.27 stratification and 0.86–1.10 screens). Screens are simple but leak IR.

## Information Analysis (Signal Backtest)
**When to use**: Deciding whether a signal has value (Ch 12).
**How**: (1) Turn the signal into an information-efficient factor portfolio each period: long/short, zero beta, zero net, factor-neutral. (2) Measure its IR, IC, t-statistic and decay. Compare the long and short legs to see where the information lies.
**Trade-offs**: Guard against data mining with intuition, restraint, sensibleness and out-of-sample tests. With 20 worthless tests, P(at least one t > 2) = 64%.

## Event Study
**When to use**: Signals that fire on discrete events: earnings surprises, CEO changes, buybacks (Ch 12).
**How**: Regress post-event standardized residual returns θ/ω on event attributes. Then IC ≈ √R² and α = ω · fitted value. Account for event frequency (breadth) and decay.
**Trade-offs**: Rare events mean low breadth, so a high IC per event is required.

## Signal Blending Across Horizons (Now/Later)
**When to use**: A signal decays, and you hold both a fresh and a lagged version (Ch 13).
**How**: Let ρ be the score autocorrelation and γ the IR decay. If ρ < γ, blend. If ρ > γ, short the lagged score. Weight w_Now = (1−ργ)/((1+γ)(1−ρ)).
**Trade-offs**: A slowly decaying signal permits lower turnover. Value-added half-life is half the IR half-life.

## Dividend Discount / Implied Growth
**When to use**: Valuation-based alphas (Ch 9).
**How**: α_DDM = yield + g − (i_F + β·f_B). Shrink analyst growth toward implied growth g*: g′ = mean(g*) + [std g*/std g]·(g − mean g). Use a three-stage DDM for realism.
**Trade-offs**: Raw growth estimates are far too dispersed (std ~23% vs ~3% implied). Without shrinkage, alphas are noise. IRR mode assumes the mispricing persists; NPV mode assumes it corrects in one year.

## Value-Added / Turnover Frontier
**When to use**: Choosing trading intensity (Ch 16).
**How**: Pick the turnover where the frontier's slope equals the round-trip cost. Bound: VA(TO) ≥ VA_I + ΔVA_Q·x(2−x) with x = TO/TO_Q. Halving turnover keeps at least 75% of VA.
**Trade-offs**: Use stock-specific costs (≈30% cost reduction). Rebalance to the edge of the no-trade band, not to the target.

## Market-Impact Model
**When to use**: Estimating trade cost (Ch 16).
**How**: Cost ≈ commission + spread/2 + c·σ·√(trade/ADV). Trading one day's volume costs about one day's volatility, so total cost ∝ size^1.5.
**Trade-offs**: Realized costs are censored, because unexecuted trades are opportunity cost. A risk-averse trader front-loads execution; an impact-dominated one trades evenly.

## Performance Attribution
**When to use**: Evaluating a manager or strategy (Ch 17).
**How**: Jensen regression r_P = α + β·r_B + ε, with t_α = α√T/ω. Decompose by factor and specific contributions, and split systematic return into expected beta, beta surprise and timing. Use portfolio-based analysis when holdings are available.
**Trade-offs**: SE(IR) ≈ 1/√years: proving IR 0.5 takes 16 years. Watch for survivorship bias.

## Benchmark Timing
**When to use**: Only if you have a market-timing edge (Ch 19).
**How**: β_PA* = Δf_B/(2λ_BT·σ_B²), or β = κ·S with κ = IC/(2λσ_B). Implement with futures, not by tilting stocks.
**Trade-offs**: Breadth is only ~4/yr, so you need IC ~0.25 for IR 0.5. Tilting stocks for a 0.1 active beta forces at least 1.6% residual risk.

## Tactical Asset Allocation
**When to use**: Active bets across 3–20 asset classes or countries (Ch 18).
**How**: Time-series regression forecasts on standardized predictors. Refined forecast = consensus + vol·IC·score. Test out of sample with an expanding, then rolling, window.
**Trade-offs**: Breadth is low, so the skill hurdle is high. Handle currency consistently using a composite-currency consensus.
