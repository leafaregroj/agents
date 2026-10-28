# Chapter 12: Information Analysis

## Core Idea
Judge any signal — quantitative or judgmental — by a two-step process: (1) turn the information into portfolios, (2) evaluate those portfolios' performance, with the **information ratio** as the summary of investment value. Information analysis measures the unfettered value of a signal *before* backtesting adds turnover, tradability and cost considerations — and it is only as honest as your defenses against data mining.

## Frameworks Introduced
- **Two-step information analysis**: Step 1 — information into portfolios; Step 2 — performance evaluation of those portfolios.
  - When to use: any time you ask "does this predictor contain information about alphas?" — buy/sell lists, scores, ranks, or full alphas.
  - How: generate a portfolio set every period (period should match the information's own refresh/horizon: daily info → daily portfolios, quarterly info → quarterly). Then compute cumulative returns, regression alphas/betas, t statistics, IR, IC; optionally turnover, upside vs. downside (long vs. short leg), up- vs. down-market behavior.
  - Why it works / failure mode: it ignores *how* the recommendations were produced and looks only at outcomes. It is extremely sensitive to **price errors**: a bad price one month and its correction the next looks like a realized opportunity.
- **Six procedures for Step 1 (information → portfolios)**:
  1. Buy/sell lists → equal- or value-weight the buy group and the sell group.
  2. Scores → one portfolio per score category (e.g., quintiles), equal- or value-weighted.
  3. Alphas → split at average alpha; weight each stock by its distance from the average.
  4. Alphas → rank into quintiles/deciles; equal- or value-weight within each.
  5. **Factor portfolio (beta-controlled)**: long and short portfolios of equal value and equal beta; long has a unit (e.g., 1 std dev) bet on the signal relative to short; long tracks short as closely as possible.
  6. **Factor portfolio with controls**: as 5, but also matched on prespecified control variables (industry, sector, size, risk-model factors).
  - Recommendation: **use procedure 5 or 6** — they set up a controlled experiment and isolate the *new* information in the signal from incidental bets (quintile portfolios carry hidden market/size/industry bets).
- **Event studies**: for information that arrives episodically (earnings announcements, new CEO, dividend changes, splits, insider trades).
  - How: collect (a) an event description (0/1 or a magnitude, e.g., earnings surprise = (announced − consensus)/forecast dispersion; dividend change as yield change), (b) post-event **residual** (or specific) returns — to avoid confounding calendar time with event time, (c) conditioning variables at event time (e.g., inside vs. outside CEO, predecessor retired/fired, leverage, institutional ownership). Regress standardized post-event residual returns on the conditioning variables; examine coefficients by window (week 1, week 2, …).
- **Four guards against data mining**: **intuition, restraint, sensibleness, out-of-sample testing.**

## Key Formulas
- **Regression for Step 2**: `r_P(t) = α_P + β_P · r_B(t) + ε_P(t)` (excess returns). `t_α = α̂ / SE(α̂)`; |t| > 2 ⇒ < 5% chance luck produced the returns (normality assumed).
- **IR vs. t statistic**: `IR ≈ t / √T` (T = years of observation); more exact as observations grow. Equivalently `SE(IR) ≈ 1/√T`.
- **Fundamental law back-out of breadth**: `IR = IC · √BR` ⇒ `BR = (IR/IC)²`.
- **Event-study regression** (event n, window 1..j): `θ_n(1,j) / ω_n(1,j) = Σ_k c_k · X_nk + ε_n`, where θ = cumulative residual return, ω = residual risk over that window, X = conditioning variables. LHS has ex ante mean 0, std 1, so in-sample `IC ≈ √R²`.
- **Event study → alpha**: `α_n(1,j) = ω_n(1,j) · Σ_k ĉ_k X_nk` — this *is* volatility · (IC · score). If the event happened j₁ periods ago, use the coefficients and ω estimated for the window (j₁+1, j₁+j).
- **Signal decay**: `IC(j) = IC(1) · δ^(j−1)`; half-life (days) = `ln 0.5 / ln δ`.
- **Event arrivals**: geometric; P(next event in j days) = `p(1−p)^(j−1)`; mean gap = `1/p`.
- **Annual IC from daily IC** (J trading days): `IC(J) = IC(1) · (1 − δ^J) / ((1 − δ)·√J)`.
- **Event-based IR** (N similar assets, uncorrelated residuals) — reconstructed from the appendix derivation; verify against the book before relying on exact form: `IR² ≈ N · J · IC(1)² · p / (1 − (1−p)·δ²)`. Effective breadth `N* = IR² / IC(J)²`: N* → N when events are frequent and information persistent; N* ≪ N for rare events (new CEO ≈ every 7 years ⇒ p ≈ 0.00056/day).
- **Multiple-testing**: P(at least one |t|>2 in K independent worthless tests) = `1 − 0.95^K`; K = 20 ⇒ 64%.

## Key Concepts
- **Predictor**: any data that may help forecast alphas; = signal + noise. Information analysis estimates the signal-to-noise ratio.
- **Information classifications**: primary vs. processed; judgmental (irreproducible) vs. impartial; ordinal (groups/order) vs. cardinal (meaningful numbers); historical vs. contemporary vs. forecast.
- **Information coefficient (IC)**: correlation between the data item and subsequent realized alpha; 0 = pure noise, 1 = pure signal, negative = perverse; bounded in [−1, +1].
- **t statistic vs. IR**: t measures *statistical significance*; IR measures *value added* (risk-reward per year). IR 0.5 over 5 years is more significant than over 1 year, but has the same value added.
- **Factor portfolio (information-efficient portfolio)**: minimum-risk portfolio with unit exposure to the signal, subject to constraints (zero beta, zero net investment, zero exposure to risk-model factors). Its long and short legs reveal upside vs. downside information.
- **Event study**: cross-event (rather than cross-sectional) analysis of residual returns following an event type.
- **Data mining**: finding patterns in history that don't exist; "the bane of information analysis".
- **Statistics of coincidence**: improbable from a narrow perspective, likely from a broad one (lottery double win: 17 trillion:1 for her, ~30:1 for someone).

## Mental Models
- Think of information analysis as a **controlled experiment**: the factor-portfolio procedures hold beta/industry/size fixed so that any performance is attributable to the signal alone.
- Use the **broad perspective** on any backtest: count every variant you tried, not just the winner. "After 19 false starts, the 20th works with t = 2" is noise with 64% probability.
- Think of an event signal's value as three knobs: **event rate (p), initial forecasting power (IC(1)), decay (δ / half-life)**. Rare events with fast decay yield little breadth regardless of IC.
- Use event studies when information arrives at different dates for different assets; use cross-sectional analysis when it arrives for the whole universe at once.

## Anti-patterns
- **Comparing only top vs. bottom quintile**: quintiles embed incidental bets (market, size, industry) that muddy the conclusion — the book-to-price quintiles and factor portfolio told different stories on the same data.
- **Nonintuitive signals** (sunspots, skirt lengths, Super Bowl winners): may fit history, fail in implementation.
- **Endless variant testing** (current vs. lagged price, sector subsets…): each variant drifts further from intuition toward the specific historical sample. Map out variants *before* testing.
- **Believing IR > 2** on simple public data in an efficient market: it signals an analysis error, not insight. Only ~10% of realized IRs exceed 1.
- **Using calendar-time total returns in event studies**: confounds market conditions with the event; use residual (or specific) returns.
- **Trusting uncleaned prices**: one bad price manufactures a fake "realized opportunity".

## Reference Tables
Book-to-price information analysis, S&P 500, Jan 1988–Dec 1992 (monthly):

| Portfolio | Alpha (mo) | t(α) | Beta | IR |
|---|---|---|---|---|
| Ex. A Highest B/P quintile | 0.03% | 0.14 | 1.02 | 0.06 |
| High | −0.06% | −0.05 | 0.92 | −0.21 |
| Mid | −0.14% | −0.97 | 0.92 | −0.45 |
| Low | −0.15% | −1.10 | 1.04 | −0.51 |
| Lowest B/P quintile | 0.31% | 1.15 | 1.12 | 0.53 |
| Ex. B Long (procedure 5) | −0.02% | −0.10 | 1.08 | −0.05 |
| Short | −0.11% | −0.76 | 1.08 | −0.35 |
| Net (long − short) | 0.08% | 0.59 | 0.00 | 0.27 |

None of the alphas are significant at 95%. Value underperformed growth 1989–1990; the net factor bet worked to early 1989, vanished 1990–91, revived 1992.

Data-mining guardrails: IR > 1 ≈ top 10% of realized IRs; scrutinize any IR > 2.

## Worked Example
**Backing out breadth from book-to-price.** Net factor portfolio IR = 0.27; IC of book-to-price signal = 0.01.
`BR = (IR/IC)² = (0.27/0.01)² = 729` → "a little over 700 independent bets per year". The raw data supply 500 assets × 12 months = 6,000 items/year, so most items are *not* independent. Lesson: BR is harder to measure than IR or IC; infer it.

**Multiple testing.** 20 independent worthless strategies, each with 5% chance of |t| > 2: P(≥1 "significant") = 1 − 0.95²⁰ = 0.64.

## Key Takeaways
1. Evaluate information by building portfolios from it and measuring their performance; the IR is the single best summary of value.
2. Prefer beta-neutral (and factor-controlled) long/short factor portfolios over quintile sorts — they isolate the signal.
3. Translate between statistics: `IR ≈ t/√T`, `SE(IR) ≈ 1/√T`, `BR = (IR/IC)²`.
4. For episodic information, run a standardized-residual-return event study; the fitted values give alpha = volatility · IC · score directly.
5. Event value depends on frequency, initial IC and half-life together — rare events deliver little effective breadth.
6. Defend against data mining with intuition, restraint (pre-specify variants), sensibleness (distrust IR > 2) and out-of-sample tests (later period, or alternate months).

## Connects To
- **Ch 6**: Fundamental law `IR = IC·√BR` used to back out breadth.
- **Ch 10–11**: refined forecasts, alpha = volatility · IC · score; IC is the input to combining signals.
- **Ch 13**: information horizon / half-life of signals, introduced here via event decay.
- **Ch 14**: backtesting → portfolio construction; turnover and transactions costs.
- **Ch 17**: full performance analysis; standard error of IR.
- **External**: Fama–MacBeth / cross-sectional regression (factor portfolios ≡ regression factor returns); multiple-hypothesis testing (Bonferroni); event-study methodology (MacKinlay).
