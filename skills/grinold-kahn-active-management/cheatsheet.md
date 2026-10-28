# Cheatsheet — Grinold & Kahn Decision Rules

## Core equations
| What | Formula |
|---|---|
| Fundamental Law | IR = IC · √BR |
| Alpha from a signal | α = ω · IC · z |
| Value added | VA = α_P − λ_R·ω_P² → optimum ω* = IR/(2λ_R), VA* = IR²/(4λ_R) |
| Implied risk aversion | λ = IR/(2ω) (e.g. IR 0.5, ψ 5% → 0.05) |
| Combine sources/managers | IR_total² = Σ IR_i² |
| Optimal active beta | β_PA = Δf_B/(2λ_BT·σ_B²) |
| Ex post skill | IR ≈ t/√years; SE(IR) ≈ 1/√years |

## Thresholds & defaults
| Quantity | Weak | Good | Very good | Exceptional / suspicious |
|---|---|---|---|---|
| IR (annual, ex ante) | <0.3 | 0.5 (top quartile) | 0.75 | 1.0 / >2 = distrust |
| IC | <0.02 | 0.05 | 0.10 | 0.15 world-class / >0.20 = bad backtest |
| λ_R | 0.05 aggressive | 0.10 moderate | 0.15 restrained | |
| Active risk ψ | | median equity 4–5% | | |
| Stock residual risk ω | | typical 20–35% | | |

- IR percentiles (generic): 90th 1.0, 75th 0.5, 50th 0, 25th −0.5.
- A hit rate h implies IC = 2h − 1: 51% → 0.02, 55% → 0.10.
- Scale IR to horizon by √T: quarterly IR = annual/2.
- Turnover cost: annualized = round-trip cost ÷ holding period in years. Calibrate to ~2% round trip.
- Trim alphas beyond 3× the alpha scale. Winsorize raw attributes at ±3σ.

## Decision rules
- **Forecasts equal consensus, or IC = 0?** → α = 0 → hold the benchmark. Mathematics can't overcome ignorance.
- **Raising IR: IC or breadth?** → Doubling IR needs 4× breadth or 2× IC. +50% breadth ≈ +22% IC. Prefer many small independent bets (e.g. 500 stocks quarterly).
- **Benchmark timing?** → Only with IC ≈ 0.25+ at quarterly frequency. Otherwise keep β_P = 1 and use futures if you time at all.
- **Several signals measure the same theme?** → Count them as one bet. Combine with the correlation adjustment, not by summing.
- **Constraint or alpha tweak?** → Prefer modifying alphas. Every constraint equals an implied alpha change.
- **Alphas ignore volatility (flat ±1%)?** → Rescale to ω·IC·z, or the optimizer piles into low-vol names.
- **Long-only?** → Expect to keep only ~50% of the unconstrained IR (more at low ψ, fewer names). The case for long/short is strongest with many names, low-vol assets, high active risk.
- **Trade?** → Hold while −sell cost ≤ MCVA ≤ purchase cost. Rebalance to the band edge, not the target.
- **Choose turnover** → where the slope of the VA/turnover frontier equals round-trip cost. Halving turnover keeps ≥75% of VA.
- **Signal decays (ρ vs γ)?** → ρ < γ: blend fresh and lagged scores. ρ > γ: short the lagged score.
- **Lagged second score?** → Diversify if IC₂ > ρ·IC₁, hedge otherwise.
- **Volatility estimate off by >3 points?** → Fix the risk model first. Underestimation can make VA negative.
- **Valuation alphas?** → Shrink analyst growth to implied-growth dispersion first. Each 1% of growth = 1% of alpha.
- **Expected return from history?** → Never. SE(mean) = σ/√T, e.g. 35% vol over 5 years → ±16%.
- **Fee-worthy?** → IR 0.5 at λ 0.15 gives VA ≈ 0.42%, probably not worth a typical fee.

## Tells & smells
- Backtest IR > 2 or IC > 0.2 → data mining, look-ahead, or survivorship.
- Many tests, one t > 2 → coincidence: P = 1 − 0.95^K.
- Monthly rebalancing on an annual view → one bet, not twelve.
- All picks in one industry or theme → one factor bet disguised as N stock bets.
- Highest-alpha names don't have the highest marginal contribution to risk → the portfolio isn't using its alphas.
- "Everyone agrees with my idea" → it's already in the price (Treynor test).
- Past IR predicts little. Breadth proxies (number of holdings, turnover) predict better.
- One-year results: even an IR 0.5 manager beats the benchmark only ~69% of the time.
