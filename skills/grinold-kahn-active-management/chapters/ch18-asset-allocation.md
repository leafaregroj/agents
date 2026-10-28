# Chapter 18: Asset Allocation

## Core Idea
Tactical asset allocation (TAA) obeys the same active-management principles as stock selection but with only 3–20 assets, so the Fundamental Law sets a high skill hurdle; research it in three steps—time-series return forecasts, benchmark-anchored (Bayesian) portfolio construction, and out-of-sample testing—and handle currencies so consensus returns are consistent across numeraires.

## Frameworks Introduced
- **Strategic vs. tactical asset allocation**: strategic = choosing the target mix (analogous to choosing a benchmark; not covered); tactical = active deviations around it (analogous to active management within a market). Domestic TAA: stocks/bonds/cash; global TAA: ~20 equity and bond markets plus currencies.
  - Why pursue it despite low breadth: new return sources across borders, diversification, and control of the largest determinant of total return (global allocation is the largest source of performance differences among global portfolios).
- **The Three-Step Process**:
  1. *Forecast asset-class returns* with a separate **time-series** model per asset class (not a cross-sectional model): asset classes have low correlations relative to stocks' shared market factor, and explanatory variables are series-specific and not comparable across countries.
  2. *Build optimal portfolios* by refining raw forecasts toward benchmark-implied consensus returns (Bayesian, Ch 10) instead of trading raw excess return against total risk.
  3. *Test out-of-sample*: cumulative excess and active returns, Sharpe ratios, IR, t stats, turnover, up/down market behavior, hit rates, and per-market contribution (dominance by one market is a warning).
  - Key point: time-series forecasts still add value only through **relative** performance—betting on A vs. B this month, not A this month vs. last month.
- **Rolling/expanding estimation for honest in-sample testing**: fit on months 1–30, forecast month 31; expand the window to 60 months, then roll a 60-month window. Measure forecasting ability by IC on these quasi-out-of-sample forecasts (8 years of data → 5.5 years of forecasts), choose variables, then validate on a true out-of-sample period. Needed because a model fitted through T₂ embeds future information into all earlier fitted values.
- **Sneaking the benchmark into expected returns**: keep traditional total-risk mean/variance optimization but feed it refined forecasts `consensus + volatility·IC·score`, where consensus = benchmark-implied (β · f_B). Markets with insignificant IC revert to consensus, so the optimal portfolio stays near the benchmark.
  - Failure mode it fixes: raw forecasts + highly correlated asset classes → extreme, unstable weights far from benchmark (the problem Black-Litterman also addresses).
- **Composite currency (COM/BSK) approach** (Grinold 1996) for international consensus returns: define a basket currency BSK (e.g., 40% $, 30% ¥, 20% £, 10% DM) that is both numeraire (COM) and risk-free asset; presume portfolio Q efficient from the COM perspective; translate to each local perspective. Moving from Q to Q_US changes **only the currency position**—partially hedging the basket.
- **Currency semantics**: "currency" is both a numeraire (how you keep score) and an asset (a short-term default-free bill in that country).

## Key Formulas
- **Forecast model** (18.1): `r(t) = Σ_j b_j · x_j(t) + ε(t)`, x_j known at start of period t. Only `x_j b_j` has return units—standardize x_j (z-scores) to compare coefficient magnitudes.
- **Refined forecast** (18.2–18.3): `E{r | g} = E{r} + Cov{r, g}·Var{g}⁻¹·(g − E{g})` → rule of thumb `refined = consensus + volatility · IC · score`, with a per-market (possibly time-varying) IC.
- **Benchmark-implied consensus** (18.4): `f_n = β_n · f_B` (β vs. benchmark; f_B = consensus benchmark expected excess return).
- **Cross-currency return** (18.5): `R_BHP(US|0,t) = R_BHP(UK|0,t) · [($/£)(t) / ($/£)(0)]`.
- **Cross-perspective excess returns** (18.6–18.7, via Itô): `r_n(US) = r_n(UK) − r_$(UK) + Cov{r_n, r_£}` (covariance from the US perspective) — excess return from US view = UK-view excess return, minus UK-view excess return on US bills, plus covariance of asset with sterling bills. Taking expectations links expected excess returns; requires both return and risk forecasts.
- **Consistency check (Siegel-paradox form)**: applying 18.7 to the two bills, `f_DM(US view) + f_$(German view) = σ²_FX` (exchange-rate variance).
- **Q_US from Q**: subtract fraction `1 − σ_Q/SR_Q` of the portfolio long the currency basket and short 100% dollars. With SR_Q ≈ 0.35, σ_Q ≈ 12% → ≈ 65% hedge (12% is lower than all-equity Q's 15–20% because of bond exposure).

## Key Concepts
- **Tactical asset allocation**: active variation around a strategic target mix.
- **Home bias**: assuming a ~20%-foreign US pension mix is efficient makes implied foreign returns artificially low; reweighting to 60% foreign makes them too high.
- **Composite currency (COM) / basket (BSK)**: single-currency fiction used to derive consistent consensus returns.
- **Q_US**: the portfolio that, with ordinary domestic CAPM from the US perspective, reproduces the international consensus returns.
- **Expanding/rolling window IC**: forecasting-ability measure free of look-ahead.
- **Explanatory variables used**: predicted dividend yield, short rate, short-rate spread to the US, exchange rate (US$ per local unit), January dummy (last two dropped for US).

## Mental Models
- Think of TAA as stock selection with tiny breadth: IR = IC·√BR punishes 4–20 bets, so demand high IC or frequent independent forecasts.
- Use benchmark-implied consensus as the Bayesian prior; let IC decide how far forecasts move from it.
- Think of a currency forecast inconsistency as an implied exchange-rate volatility: if cross-forecasts sum to more than realistic σ²_FX, they're incoherent.
- When two explanatory variables are highly correlated with offsetting signs (short rate vs. rate spread), drop one.

## Anti-patterns
- **Total-risk optimization on raw forecasts**: extreme sensitivity, weights far from benchmark, concentration in few assets.
- **Testing a time-series model only in-sample**: coefficients estimated through T₂ leak future information.
- **Naive consensus returns from a home-biased "efficient" portfolio**: inconsistent and unrealistic across currencies.
- **Treating currency only as numeraire or only as asset**: muddles hedging decisions.
- **Performance driven by a single market**: likely luck, not a robust process.

## Reference Tables
**Model results (as of Jan 1993, 60-month window, German/Japanese/UK/US equity, 1985–1992)**: average adjusted R² ≈ 9% (≈0 for US, >20% Germany)—good for a return model. Selected coefficients (monthly):
| Effect | Value | Interpretation |
|---|---|---|
| US dividend yield | +4.0 | +1% predicted yield → +4.0%/mo expected excess return; DY coefficient positive in every market |
| US short rate | −0.56 | +1% short rate → −0.56%/mo |
| UK rate spread to US | −1.92 | spreads always negative |
| Germany exchange rate | −0.58 | only significant FX coefficient; DM falling $0.59→$0.58 → +0.58%/mo |
| Germany January dummy | +2.89%/mo | January effect positive for Germany, UK |
(Full Table 18.1 garbled in source; values above are those quoted in text.)

**Consensus currency returns, naive (Table 18.2) vs. COM approach (Table 18.3)**
| Pair | Naive | COM-consistent |
|---|---|---|
| US investor holding DM bills | 1.93% | 0.41% |
| German investor holding $ bills | 4.80% | 1.08% |
| Implied DM/$ volatility | 25.9% | 12.2% |
Realized DM/$ volatility over sizable periods: 11–13%. Presumed efficient portfolio: 60/40 stocks/bonds in each country; country weights US 40%, Japan 30%, UK 20%, Germany 10%; risk from 1970–1995.

## Worked Example
- **Currency translation**: UK investor sees BHP +3.5%; $/£ moves 1.50 → 1.52. US return = 1.035 × 1.52/1.50 − 1 = **4.88%**.
- **Consistency check**: naive 1.93% + 4.80% = 6.73% = σ²_FX → σ_FX = 25.9% (implausible). COM: 0.41% + 1.08% = 1.49% → σ_FX = 12.2% (matches history).
- **Hedge fraction**: 1 − 0.12/0.35 ≈ 0.66 → Q_US ≈ Q minus 65% of (long basket, short dollars).

## Key Takeaways
1. TAA follows the book's principles but faces a breadth handicap; skill requirements are high.
2. Forecast with per-asset-class time-series models, but remember value comes from relative calls.
3. Use expanding/rolling windows and IC to test forecasts honestly; then validate out-of-sample.
4. Avoid total-risk optimization on raw forecasts—refine toward benchmark-implied consensus (`volatility·IC·score`).
5. For global portfolios, derive consensus returns from a composite-currency efficient portfolio; local efficient portfolios differ only by a partial currency hedge (~65%).
6. Check currency forecasts for consistency via implied exchange-rate variance.
7. Inspect out-of-sample results for turnover, up/down behavior, and concentration in one market.

## Connects To
- **Ch 2**: CAPM consensus expected returns `f = β·f_B`.
- **Ch 4 / Ch 19**: benchmark timing is the one-asset limit of allocation.
- **Ch 6**: Fundamental Law explains the breadth hurdle.
- **Ch 10**: Bayesian refinement, `α = volatility · IC · score`.
- **Ch 12**: IC/out-of-sample information analysis.
- **Ch 17**: performance analysis of the out-of-sample results.
- **External**: Black-Litterman model; Black's universal hedging; Siegel's paradox; Solnik's international return predictability.
