# Chapter 13: The Information Horizon

## Core Idea
Every signal has a shelf life, best measured as the **half-life** of its forecasting power (of its information ratio). The half-life is an intrinsic property of the strategy: optimally mixing current and lagged portfolios (or signals), and even past returns, can raise the IR, but it never changes the horizon. "The latest is not necessarily the greatest" — old information can diversify or hedge the noise in new information.

## Frameworks Introduced
- **Macroanalysis of the information horizon (lagged-implementation test)**: measure the strategy's realized IR when implementation is delayed 1, 2, … months; fit an exponential decay `IR(l) = γ^l · IR(0)`; the half-life is when IR halves.
  - When to use: to check whether a strategy is *temporally efficient* using only its returns, without knowing its internals.
  - How: compute IR_Now and IR_Later (lagged portfolio), decay γ = IR_Later/IR_Now, and correlation ρ of their active returns. Compare ρ to γ.
  - Decision rule: **ρ < γ → hold both (diversify); ρ > γ → long Now, short Later (hedge); ρ = γ → current portfolio already uses all the information.**
- **Manager Now vs. Manager Later**: Later trades last month's portfolio of Now. The optimal blend makes the correlation between consecutive mixed portfolios equal to the IR decay rate.
- **Microanalysis (optimal mixing of old and new scores)**: treat the lagged score as a second signal and apply Chap. 10's two-signal combining rule.
  - Diversification (add a fraction of the old score) when `IC₂ > ρ·IC₁` — strong residual signal, low correlation.
  - Hedging (subtract a fraction of the old score) when `IC₂ < ρ·IC₁` — weak residual signal, high correlation; removes shared noise.
  - Ignore the old score when `IC₂ = ρ·IC₁`.
- **"Settling old scores"**: use last period's realized return (standardized) as a third predictor. A realized alpha that matches the forecast does *not* mean the old information is exhausted — the move may have been incidental. The rule adjusts the *previous* score for the part of the past return it predicted.
- **Gradual decline model**: IC with the return n periods ahead = `IC · γ^n`; mixed signals are optimal when their autocorrelation equals γ.

## Key Formulas
- **Half-life**: `HL = ln(0.5) / ln(γ)` periods, where γ = per-period IR (or IC) decay factor. Example: monthly γ = 0.7937 ⇔ HL = 3 months.
- **Value-added half-life = ½ × IR half-life** (value added ∝ IR²).
- **Optimal weight on current portfolio** (decay γ, active-return correlation ρ, equal active risk):
  `w_Now = (1 − ρ·γ) / ((1 + γ)(1 − ρ))`, `w_Later = 1 − w_Now`.
- **Combined IR**: `IR*² = IR_Now² · (1 − 2ργ + γ²) / (1 − ρ²)`. No gain exactly when ρ = γ.
- **General optimal mix of lagged portfolios**: `h*(t) ∝ Σ_l γ^l · [h(t−l) − ρ·h(t−l−1)]` — a γ-weighted average of *innovations* (h(t−l) − ρ·h(t−l−1) is the residual of regressing the l-lag portfolio on the next older one). Then `Corr{θ*(t), θ*(t−1)} = γ` and `h*(t) = γ·h*(t−1) + innovation`.
- **Fundamental law with time**: `IR = IC(Δt) · √(1/Δt)`; breadth = arrivals per year (monthly signal ⇒ BR = 12).
- **Two-period shelf life**: `IC₁&₂ = (IC₁ + IC₂)/√2` (correlation of score with the 2-period return).
- **Combining current score s(0) and lagged score s(−Δt)** (correlation ρ between them):
  `α = ω·√Δt · [IC₁′·s(0) + IC₂′·s(−Δt)]`, `IC₁′ = (IC₁ − ρ·IC₂)/(1 − ρ²)`, `IC₂′ = (IC₂ − ρ·IC₁)/(1 − ρ²)`;
  combined `IC² = (IC₁² + IC₂² − 2ρ·IC₁·IC₂)/(1 − ρ²)`.
- **IC versus return horizon** (n = t/Δt periods): `IC(t) = IC · (1 − γⁿ) / ((1 − γ)·√n)`. Peak predictive power when the return horizon ≈ 1.8–2 × the signal half-life (`(1−e^(−x))/√x` max 0.638 at x = 1.257, x = n·ln(1/γ)).

## Key Concepts
- **Information horizon / shelf life**: the half-life of a signal's forecasting ability (IR decay under delayed implementation).
- **Just-in-time signal**: valuable only until the next signal arrives (interarrival time = shelf life).
- **Decay factor γ**: per-period ratio IR(lag l+1)/IR(lag l), assumed exponential.
- **Mixture strategy**: weighted combination of current and lagged portfolios; its IR decays at the same γ (proposition in the appendix) and its risk is independent of lag.
- **Innovation**: the new information in a portfolio/signal relative to its predecessor, h(t) − ρ·h(t−1).
- **Diversify vs. hedge**: old score adds signal (diversify) or cancels shared noise (hedge).
- **Settled score**: the old score corrected for the part of the subsequent realized return it explained.
- **Overlapping intervals**: measuring IC period-by-period ahead avoids their double counting.

## Mental Models
- Think of information like food: vegetables (fresh is best), fruit (needs ripening), wine/cheese (improves with age), sherry (best as a blend of vintages). Ask which one your signal is.
- Use the Now/Later test as a cheap audit: if blending last period's portfolio with the current one raises the IR, you are leaving information on the table — and you can apply the same weights to inputs instead of outputs.
- Think of the half-life as a fingerprint: time-averaging or differencing can't change it, only the IR level.
- Use past returns as an extra (weak) predictor even when they have no direct forecasting power — combined with the scores they can still help.

## Anti-patterns
- **Assuming the latest signal is all you need**: when ρ ≠ γ, older signals improve the forecast.
- **Declaring an alpha "realized" because the stock moved as forecast**: the move may be incidental; settle the old score statistically.
- **Expecting smoothing to lengthen a strategy's horizon**: mixing raises IR, not half-life.
- **Ignoring feedback from returns to scores**: momentum signals rise after high returns, value signals fall. Fix by residualizing new scores on prior returns.
- **IC₂ > IC₁**: if the score predicts period 2 better than period 1, investigate return autocorrelation rather than trusting it.

## Reference Tables
Now/Later example (IR_Now = 1.5, IR_Later = 1.2 ⇒ γ = 0.8; both 4% active risk):

| Corr ρ(Now, Later) | w_Now | w_Later | Combined IR |
|---|---|---|---|
| 0.50 | 67% | 33% | 1.59 |
| 0.70 | 81.5% | 18.5% | ≈1.51 |
| 0.80 (= γ) | 100% | 0% | 1.50 (no gain) |
| 0.85 | 118.5% | −18.5% | ≈1.51 |

Example strategy (Fig. 13.1): IR half-life 1.2 years ⇒ value-added half-life 0.6 year — you can delay trades more than 6 months and still keep 50% of value added (a long-horizon strategy).

Binary-model illustration (monthly residual vol 6%, forecast std 4%, information over next 2 months): IC of g(t) = 0.125, IC of g(t−1) = 0.0833; optimal combination of g(t) and g(t−1) ⇒ IC = 0.1334; adding last month's return r(t−1) ⇒ IC = 0.1335 (tiny gain, even though r(t−1) alone is useless).

## Worked Example
**Now/Later at ρ = 0.5, γ = 0.8, IR = 1.5.**
`w_Now = (1 − 0.5·0.8) / ((1.8)(0.5)) = 0.6/0.9 = 0.667`; `w_Later = 0.333`.
`IR*² = 1.5² · (1 − 0.8 + 0.64)/(1 − 0.25) = 2.25 · 0.84/0.75 = 2.52` ⇒ `IR* = 1.59`.
At ρ = 0.85: `w_Now = (1 − 0.68)/(1.8·0.15) = 0.32/0.27 = 1.185` → 118.5% long Now, 18.5% short Later.

**Half-life from a delay test.** Monthly signal, IR = 1, falls to 0.75 when delayed one quarter: γ_quarter = 0.75 ⇒ HL = ln 0.5/ln 0.75 = 2.41 quarters ≈ 7.2 months; value-added half-life ≈ 3.6 months.

**Two-period IC.** IC₁ = 0.15, IC₂ = 0.075 ⇒ IC₁&₂ = 0.225/√2 = 0.159. With ρ = 0.5 = IC₂/IC₁, the lagged score gets zero weight.

## Key Takeaways
1. Define a strategy's horizon as the half-life of its IR under delayed implementation; value added decays twice as fast.
2. Compare the correlation of current vs. lagged active returns (ρ) with the IR decay (γ): ρ < γ blend, ρ > γ hedge, ρ = γ you are efficient.
3. Optimal temporal mixing makes consecutive portfolios correlate at exactly γ; it raises IR without changing the half-life.
4. Treat older forecasts like forecasts from a different source and combine them with Chap. 10 machinery; past realized returns can add a little more.
5. Match the return horizon you evaluate a signal on to its half-life — IC peaks at roughly twice the half-life.

## Connects To
- **Ch 10–11**: combining two correlated signals; refined forecasts; alpha = volatility · IC · score.
- **Ch 12**: event-study decay (IC(j) = IC(1)·δ^j) and information analysis machinery.
- **Ch 14 & 16**: after-cost IR depends on half-life vs. transaction costs; slow signals tolerate infrequent rebalancing.
- **Ch 6**: breadth as arrival rate (BR = 1/Δt).
- **External**: exponential smoothing / AR(1) innovations; Kalman filtering of signals.
