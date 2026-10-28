# Chapter 15: Long/Short Investing

## Core Idea
The real benefit of long/short (equity market-neutral: beta 0, equal long and short value) investing is the removal of the **long-only constraint**, which is surprisingly costly: for a typical U.S. strategy (500 assets, 4.5% active risk) it cuts the information ratio roughly in half, induces a negative size bias, and even limits *long* positions because overweights must be financed by underweights.

## Frameworks Introduced
- **Three arguments for long/short — only one holds**:
  1. *Diversification* (long and short legs each with IR, uncorrelated ⇒ IR·√2): **misleading** — it applies equally to the active portfolio of any long-only fund.
  2. *Short-side inefficiency* (long-only dominance leaves overpriced stocks): plausible but hard to prove; shorting is costly.
  3. *Constraint relief* (efficiency gained by loosening the long-only constraint): **the critical argument**, relevant to long-only managers too.
- **Information boundary (direct effect)**: in a simple model (N assets, equal-weighted benchmark, identical uncorrelated residual risk ω, zero active beta), short sales bind when a stock's score falls below a threshold; any information below it is wasted.
  - When the boundary matters most: **large universes, low asset volatility, high active risk.**
- **Knock-on (indirect) effect**: full investment means overweights must be funded by underweights; scarce underweight room (small benchmark weights) shrinks overweights too. In a 1000-asset simulation, long-only effectively assigns the same negative alpha to every asset after the top ~300 and also handles positive alphas less efficiently.
- **Gulliver and the 100 Lilliputians**: the benchmark capitalization distribution drives long-only efficiency. One stock at 99%, 100 at 0.01% each; 50 Lilliputians at +3.73% alpha, 50 at −3.73%.
  - Long/short: Gulliver gets the same ±0.79% active position as any Lilliputian with the same alpha; portfolio alpha 3.00%, active risk 2.00% either way.
  - Long-only, Gulliver alpha positive: only tiny positions (~±0.01%) and a few basis points of alpha.
  - Long-only, Gulliver alpha negative: Gulliver −1.55% funds the Lilliputian overweights (+0.04%).
  - With a 50/50 chance of ±3.73% on Gulliver, average Gulliver active weight = −0.77% ⇒ systematic **underweighting of big stocks (negative size bias)**.
  - Leverage alternative (short cash to fund overweights, sell benchmark forward): alpha 1.53%, active risk 1.42%, cash −39% of unlevered value.
- **Lognormal capitalization model + simulation** to estimate the efficiency loss: 900 random long-only optimizations per case; N ∈ {50, 100, 250, 500, 1000}; desired risk 1–8% by 1%, then to 20% by 2%; residual risk 25%, IR 1.5, full investment, no costs.

## Key Formulas
- **Unconstrained active position** (equal-weight benchmark, uncorrelated residuals): `h_PA,n = α_n / (2·λ_R·ω²)`.
- **Alphas from the fundamental law**: `α_n = ω·IC·z_n`, `IC = IR/√N`, z_n ~ mean 0, std 1.
- **Active risk**: `ψ_P = IR / (2·λ_R)`.
- **Position vs. desired risk**: `h_PA,n = ψ_P · z_n / (ω·√N)`.
- **Short-sale boundary** (benchmark weight 1/N): binding when `1/N + h_PA,n < 0` ⇔ `z_n < −ω / (ψ_P·√N)`.
- **Capitalization model**: order assets largest (n = 1) to smallest (n = N); `p_n` = rank-based probabilities from ≈1 down to ≈0 (exact form garbled in source; e.g., `p_n = (N − n + ½)/N`); `y_n = Φ⁻¹(p_n)`; `cap_n ∝ exp(c·y_n)`. Choose c to match the Gini coefficient or Lorenz curve (or set c = std dev of log cap). Use **c = 1.55** for analysis.
- **Lorenz curve / Gini**: cumulative benchmark weight of the n largest assets vs. n; Gini = 2 × (area under curve − area under equal-weight line); 0 = equal weight, → 1 = single asset.
- **Shrinkage factor**: `SF(N, ψ) = α_long-only(ψ) / (IR·ψ)` — the fraction of IR (and IC) retained long-only. The book fits a concave empirical frontier α(ψ, N) [Eq. 15.11, not recoverable from the source text]; it scales linearly with IR, so conclusions don't depend on IR. Anchors: SF ≈ 0.49 at N = 500, ψ = 4.5%; SF ≈ 0.71 at N = 500, ψ = 2%.
- **Risk-aversion/risk link** under the constraint (Eq. 15.14): derived by maximizing `U = α(ψ) − λ·ψ²` with the empirical frontier; achieving high active risk long-only requires far lower risk aversion than the unconstrained `λ = IR/(2ψ)`.

## Key Concepts
- **Long/short (equity market-neutral) strategy**: zero beta, zero net investment; a method of *implementation*, not a source of information.
- **Long-only constraint**: no negative holdings; an inequality constraint, so effects must be simulated, not derived.
- **Information boundary**: minimum z score below which long-only cannot act.
- **Knock-on effect**: limited underweights constrain overweights ("overweights require underweights").
- **Size bias**: long-only active portfolios tilt toward small stocks (only large stocks offer big underweight room).
- **Shrinkage factor**: retained fraction of IR/IC under the long-only constraint.
- **Portable alpha**: long/short active return can be attached to any benchmark with futures (e.g., sell S&P 500 futures, buy FTSE 100 futures).
- **Enhanced indexing**: low-active-risk long-only strategies, where the constraint costs little.

## Mental Models
- Every long-only portfolio *contains* a long/short portfolio (its active holdings); the question is only how badly the constraint distorts it.
- Use long/short when **N is large, asset residual risk is low, and target active risk is high**; stay long-only for low-risk (enhanced index) mandates, where long/short's extra costs (borrowing, uptick rule) may dominate.
- Think of benchmark weights as your **short-selling budget**: the smaller the weights (large N, cap-skewed index), the smaller the budget.
- Long/short is pure active management: consensus expected return is zero, so it suits the most skillful managers and investors best able to identify them; no hiding behind a benchmark.

## Anti-patterns
- **Selling long/short on the √2 diversification argument**: it is wrong; the case rests on constraints.
- **Ignoring the long-only size bias**: a −0.65 size exposure (typical at 500 stocks, 4.5% risk) lost 98 bp in Oct 1997–Sep 1998 (size factor +1.5%) and 2.35% over Sep 1988–Sep 1998 (size factor +3.61% cumulative).
- **Pushing a long-only strategy to high active risk**: efficiency falls steadily as desired risk rises.
- **Paying active fees for a passive core**: long-only fees cover benchmark-like holdings; long/short fees pay only for active positions.

## Reference Tables
Long-only z-score boundary example: N = 500, ψ = 5%, ω = 25% ⇒ boundary z = −25/(5·√500) = −0.22 ⇒ information wasted for ~41% of stocks (normal scores).

Shrinkage factor sensitivities (long-only, IR retained):

| Case | Shrinkage factor |
|---|---|
| 500 assets, 4.5% active risk (typical U.S.) | 0.49 |
| 500 assets, 2% active risk (enhanced index) | 0.71 |
| 500 assets, 4.5%, c from 1.2 to 1.6 | 0.49–0.51 |
| 250 assets, 4.5%, residual risk 20%→35% | 0.54→0.65 |
| 500 assets, 4.5%, size/log-vol correlation −0.6 (U.S. large caps ≈ −0.51 to −0.57) | 0.63 |

Capitalization distribution (MSCI etc., Sept 1998):

| Index | Assets | Gini | c |
|---|---|---|---|
| Frank Russell 1000 (U.S.) | 1000 | 0.71 | 1.55 |
| MSCI U.S. | 381 | 0.66 | 1.38 |
| MSCI U.K. | 135 | 0.63 | 1.30 |
| MSCI Japan | 308 | 0.65 | 1.35 |
| MSCI Netherlands | 23 | 0.64 | 1.38 |
| Freedonia equal-weight | 101 | 0.00 | 0.00 |
| Freedonia cap-weight (Gulliver) | 101 | 0.98 | 11.15 |

(c typically 1.30–1.60 across countries.)

Empirical: 14 U.S. long/short strategies (1990s to Mar 1998), percentiles:

| Percentile | History (mo) | Volatility | Beta | Corr w/ S&P 500 | IR |
|---|---|---|---|---|---|
| 90 | 96 | 10.90% | 0.10 | 0.23 | 1.45 |
| 75 | 86 | 6.22% | 0.06 | 0.15 | 1.23 |
| 50 | 72 | 5.50% | 0.02 | 0.04 | 1.00 |
| 25 | 50 | 4.12% | −0.03 | −0.07 | 0.69 |
| 10 | 28 | 3.62% | −0.16 | −0.20 | 0.44 |

Risk levels similar to long-only active risk (SE of mean risk 0.64%); betas statistically indistinguishable from zero — market neutrality is achievable. Small, possibly unrepresentative sample.

## Worked Example
**How often does long-only waste information?** 500 stocks, equal-weighted benchmark (0.2% each), ω = 25%, target ψ = 5%.
Active position per unit score: `h = ψ·z/(ω√N) = 0.05·z/(0.25·22.36) = 0.894%·z`. The constraint binds when `0.2% + 0.894%·z < 0` ⇒ `z < −0.22`. With normal scores, P(z < −0.22) ≈ 41%: two-fifths of the universe's negative information can't be fully used — and the knock-on effect then trims the overweights too.

## Key Takeaways
1. The case for long/short is constraint relief, not diversification.
2. The long-only constraint roughly halves IR for typical U.S. strategies (SF ≈ 0.49 at 500 names, 4.5% risk); at 2% risk it keeps ~71%.
3. Constraint costs grow with universe size, lower asset volatility, higher target risk, and more cap-concentrated benchmarks.
4. Long-only portfolios carry an incidental negative size bias (≈ −0.65 std at typical settings); monitor and hedge it.
5. Long/short offers portable alpha, fees only for active bets, and low market correlation — but no benchmark to hide behind.

## Connects To
- **Ch 5**: active frontier, IR and risk aversion (ψ = IR/(2λ)); typical active risk levels (Table 5.8).
- **Ch 6**: IC = IR/√N used to generate alphas.
- **Ch 10**: alpha = volatility · IC · score.
- **Ch 12**: long and short legs of factor portfolios reveal upside vs. downside information.
- **Ch 14**: constraints as alpha shrinkers; modified alphas.
- **Ch 16**: implementation costs of shorting.
- **Ch 19**: futures to transport alpha / benchmark exposure.
- **External**: 130/30 and other relaxed-constraint products (Clarke–de Silva–Thorley "transfer coefficient").
