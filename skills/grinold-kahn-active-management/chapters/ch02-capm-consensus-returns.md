# Chapter 2: Consensus Expected Returns: The Capital Asset Pricing Model

## Core Idea
Any return splits into a market (beta) part and a residual part, and this needs no theory. The CAPM adds one claim: expected residual return is zero. That makes CAPM returns the **consensus expected returns**, the ones that make the market portfolio mean/variance optimal. Active management pays only to the extent that your forecasts beat this consensus.

## Frameworks Introduced
- **Separation of return (beta decomposition)**: r_P = β_P·r_M + θ_P, with θ_P uncorrelated with r_M.
  - When to use: always, as the first step in analyzing any portfolio. It requires only a good beta forecast.
  - How: forecast beta (better from fundamentals/risk model than from 60-month regression), then split risk into systematic and residual.
  - Why it works: beta is defined so that θ is orthogonal to the market. This lets beta-neutral managers ignore the market forecast μ_M entirely.
- **CAPM**: E{θ_P} = 0 for every stock and portfolio, so μ_P = β_P·μ_M.
  - When to use: as the *consensus* benchmark for expected returns. It is not a forecast you should act on as an active manager.
  - Why it works: market risk is the "hot potato" someone must hold, so it is rewarded. Residual risk is self-imposed and avoidable, so it is not rewarded. Overweighting relative to the market is a zero-sum game against an opposite-side investor (the "greater fool" argument).
  - Failure mode: investors actually have different information, which is the opening for active management.
- **Expected returns ↔ portfolios (one-to-one)**: feed CAPM returns into a mean/variance optimizer and you get market + cash. Conversely, assume any portfolio is optimal and you can back out the expected returns that justify it: they are proportional to beta with respect to that portfolio.
- **Characteristic portfolios (technical appendix)**: for any attribute vector a, h_a is the minimum-risk portfolio with unit exposure to a. Named ones: C (min variance, attribute e), B (benchmark, attribute β), q/Q (max Sharpe ratio, attribute f), A (attribute α).
  - When to use: to turn any signal (E/P, alpha, beta) into a portfolio, and to reason about which portfolio a set of returns "explains".
  - How: a portfolio based on alphas with beta 1, full investment, and the desired risk is a linear combination of portfolios A, B, and C.

## Key Formulas
- β_P = Cov{r_P, r_M} / Var{r_M}. Market β = 1, risk-free β = 0.
- Regression: r_P(t) = α_P + β_P·r_M(t) + ε_P(t). These are the "realized/historical" alpha and beta.
- σ_P² = β_P²·σ_M² + ω_P², where ω_P is residual risk (Std of θ_P).
- CAPM / security market line: E{R_P} = i_F + β_P·μ_M.
- Sample error of a mean return: SE = σ/√Y (Y = years of data, independent of sampling frequency). With σ ≈ 35%, SE ≈ 16% after 5 years. This is why historical averages make poor expected returns.
- **Characteristic portfolio** of attribute a: h_a = V⁻¹a / (aᵀV⁻¹a); σ_a² = 1/(aᵀV⁻¹a); asset betas relative to h_a equal a: a = V·h_a / σ_a².
- Cov of two characteristic portfolios: σ_{a,d} = a_d·σ_a² = d_a·σ_d² (a_d = exposure of h_d to a, etc.).
- Scaling: the characteristic portfolio of κ·a is h_a/κ. Linearity: if a = κ_d·d + κ_f·f, then h_a is the matching combination of h_d and h_f.
- Portfolio C: h_C = V⁻¹e/(eᵀV⁻¹e). Every asset has β = 1 relative to C. For any fully invested P, Cov{r_P, r_C} = σ_C².
- Portfolio B: the benchmark is the characteristic portfolio of β, i.e. the minimum-risk portfolio with β = 1. β_C = σ_C²/σ_B².
- Sharpe ratio SR_P = f_P/σ_P. Portfolio q (characteristic of f) has the maximum SR: SR_q = √(fᵀV⁻¹f), and SR_P = ρ_{P,q}·SR_q.
- Portfolio Q = q rescaled to full investment (requires f_C > 0). Then f = β(wrt Q)·f_Q: **expected excess returns are proportional to betas relative to Q**.
- Alpha: α = f − β·f_B. Portfolio A (characteristic of α) has β_A = 0, uncorrelated with B.
- Efficient frontier (fully invested): h_P = w·h_C + (1−w)·h_Q. Variance rises quadratically in (f_P − f_C) [reconstructed: σ_P² = σ_C² + (σ_Q² − σ_C²)·((f_P − f_C)/(f_Q − f_C))²].
- **CAPM theorem**: if all investors have mean/variance preferences, all assets are included, everyone knows f and agrees on V, and there are no costs or taxes, then Q = M and f = β·f_M.

## Key Concepts
- **Consensus expected returns**: the returns that make the market (consensus portfolio) mean/variance optimal, i.e. the CAPM returns.
- **Excess return**: total return minus the risk-free return over the same period.
- **Residual return θ**: return uncorrelated with the market. Its cap-weighted sum is exactly zero.
- **Security market line**: expected return vs beta, with intercept i_F and slope μ_M. The ex post version (the "insecurity market line") slopes down when the market underperforms cash.
- **Efficient markets (weak/semistrong/strong)**: no outperformance using price/volume data, then public information, then any information. Consistent with the CAPM but not identical to it.
- **Humility principle**: success is hard, so the burden of proof is on the active manager.
- **Characteristic portfolio**: minimum-risk portfolio with unit exposure to an attribute. It may be levered and long/short.
- **Beta mean reversion**: high historical betas tend to fall and low betas tend to rise toward 1. Fundamental betas forecast better than 60-month historical ones.

## Mental Models
- Think of the CAPM as the **null hypothesis for alpha**: a consensus residual return of zero is a convenient starting point. Your job is to forecast deviations from it.
- Use **beta control to drop market timing**: with β_P = 1, μ_M doesn't affect active return, so research can focus only on residual returns.
- Think of **any portfolio held as an implied view**: holding P means believing returns ∝ β relative to P. This is useful for reverse-engineering what a portfolio "believes".
- Think of an **active position as a bet against an opposite-side investor**: someone must hold the mirror-image residual position, so you need a reason you are not the greater fool.

## Anti-patterns
- **Using historical average returns as expected returns**: huge sample error (about 16% SE after 5 years) plus a changing universe and changing firms.
- **Treating the CAPM as a risk model**: the CAPM is about expected returns, not risk. The single-factor risk model is a different thing.
- **Being a "CAPM true believer" and an active manager**: that requires a schizophrenic or cynical stance. A pure CAPM view gives active management a 50/50 chance.
- **Confusing luck with skill**: the efficient-markets burden of proof should cut out ideas that can't pass scrutiny.

## Reference Tables
Historical (60-month) vs BARRA predicted betas vs S&P 500, Dec 1992 (selected MMI stocks; pairing in the source is partially garbled):

| Stock | Historical β |
|---|---|
| American Express | 1.21 |
| Chevron | 0.46 |
| Exxon | 0.46 |
| Eastman Kodak | 0.60 |
| IBM | 0.64 |
| General Electric | 1.30 |
| Disney | 1.23 |
| Procter & Gamble | 1.00 |

Predicted betas cluster more tightly around 1 (about 0.66–1.30), which shows the regression toward the mean.

Characteristic portfolios:

| Characteristic | Portfolio |
|---|---|
| f (expected excess return) | h_q |
| e_q·f | h_Q (if f_C > 0) |
| β | h_B (benchmark) |
| e (ones) | h_C (min variance) |
| α = f − β·f_B | h_A |

## Worked Example
- **American Express vs S&P 500, Jan 1988–Dec 1992 (monthly)**: historical β = 1.21 (SE 0.24). Realized residual = −78 bp/month (SE 96 bp), not significant at 95%. Residual monthly SD = 7.05%, R² = 0.31.
- **SML**: with i_F = 5% and μ_M = 7%, portfolios with β = 0.8, 1.15, and 1.3 have expected returns of 10.6%, 13.05%, and 14.1%.
- **Problems**: Sears β = 1.05. If the S&P underperforms bills by 5%, the expected excess return is −5.25%. With a 7% long-run μ_M, it is 7.35%. For stocks A (β 1.15, σ 35%) and B (β 0.95, σ 33%) with σ_M = 20%: ρ_AB = 1.15·0.95·400/(35·33) ≈ 0.38. Residual risks are ω_A = √(1225 − 529) ≈ 26.4% and ω_B = √(1089 − 361) ≈ 27.0%, so B has more residual risk.
- **Frontier**: Q has σ = 20% and f = 7%. C has σ = 12%, so f_C = f_Q·σ_C²/σ_Q² = 7·144/400 = 2.52%.

## Key Takeaways
1. Separating market and residual return is free of theory, so do it always, using good (fundamental) beta forecasts.
2. The CAPM gives you the consensus. Your alpha is only your deviation from it, and its expected value under the consensus is zero.
3. Running beta near 1 removes the need to forecast the market and keeps research focused on residuals.
4. Never use raw historical mean returns as expected returns.
5. Every set of expected returns implies an optimal portfolio, and vice versa. Use characteristic portfolios to move between them.
6. Hold the market if you lack superior information. Otherwise, be ready to defend your edge.

## Connects To
- **Ch 3**: beta and residual risk forecasts come from structural risk models.
- **Ch 4**: alpha is defined as expected residual return relative to the benchmark. The CAPM gives zero consensus alpha.
- **Ch 5**: portfolio A/Q machinery, and the information ratio as an analog of the Sharpe ratio.
- **Ch 7**: the APT as an alternative return model (useful to active managers, not as a consensus).
- **Black–Litterman / reverse optimization**: backing out implied returns from a held portfolio.
- **Fama–French (1992)**: a challenge to the CAPM. See Black (1993) and Grinold (1993) for the reply.
