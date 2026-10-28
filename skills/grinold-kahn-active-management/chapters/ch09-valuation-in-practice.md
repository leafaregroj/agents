# Chapter 9: Valuation in Practice

## Core Idea
Practical equity valuation is ad hoc. There are three families:
- **dividend discount models (DDMs)**, which are only as good as their growth inputs ("g in, g out");
- **comparative valuation**, which prices a firm's attributes relative to similar firms;
- **returns-based analysis**, which skips prices and regresses returns directly on attributes in an APT-like factor model.

Keep in mind the humility principle (the market may be right) and the Fundamental Law (you need to be right only a little more than 50% of the time).

## Frameworks Introduced
- **Modigliani–Miller ground rules**: dividend policy is "pay you now or pay you later", and financing policy does not change firm value.
  - Split equity value into *operational value* (the "widget side": revenue minus operating costs minus capital costs) and *financial value* (capital surplus minus debt).
  - Be suspicious of any valuation scheme that relies on "magic" from dividends or leverage. Accounting numbers such as EPS, EPS growth, E/P and B/P can be manipulated through dividend and debt policy.
- **Constant-growth (Gordon–Shapiro) DDM** and its alpha form.
  - **The Golden Rule of the Dividend Discount Model: "g in, g out."** Each extra 1% of growth input adds 1% to alpha.
- **Implied growth rates**: assume the stock is fairly priced (α = 0) and solve for g. Use them to:
  1. sanity-check analyst growth (consensus growth runs too high because Wall Street sells stocks);
  2. flag prices that embed unrealistic growth;
  3. anchor growth forecasts.
- **Three ways to fix unrealistic growth inputs**:
  1. *Sector rescaling*: linearly transform raw growth forecasts within each sector to match the mean and standard deviation of that sector's implied growth rates. Caveat: this pushes sector alphas toward 0 and so deletes sector-timing information.
  2. *Skill-based shrinkage*: g′_n = g\*_n + c·(g_n − g\*_n), where c depends on growth-forecasting skill (the IC, see Ch 10). With no skill, c = 0.
  3. *Three-stage DDM*: short-run growth g_IN for stage 1 (0–T1), linear interpolation in stage 2 (T1–T2), equilibrium growth g_EQ from T2 on. Payout ratios are interpolated the same way. It is just an elaborate smoother and "is not magic".
- **Two ways to convert a DDM into alphas**:
  - *IRR mode*: solve for the internal rate y_n that matches dividends to price. The implied benchmark f_B = Σ h_B,n·y_n − i_F. Then α_n = y_n − (i_F + β_n·f_B). This assumes the mispricing *persists forever* (δ = 1 in Ch 8 terms, collected as a yield).
  - *NPV mode*: discount at the fair rate i_F + β_n·f_B to get p_fair. Calibrate f_B so that aggregate fair value equals aggregate market value. Then α_n ≈ (p_fair − p_mkt)/p_mkt. This assumes the mispricing *disappears within 1 year* (δ = 0).
- **Comparative valuation**: cross-sectionally regress price on attributes within groups of similar firms, e.g. p_n = c1·e_n(1) + c2·b_n(0) + ε_n, fit by OLS, GLS or pooled regression. The pricing error is the alpha signal.
  - Arbitrage interpretation: build "megacompanies" OV (overvalued) and UV (undervalued) with identical attributes. UV is cheaper, which is a quasi-arbitrage.
  - Failure mode: an omitted attribute such as brand value appears as persistent "overvaluation". Compare the current error to the firm's historical average error.
- **Returns-based analysis**: r_n(t) = Σ_k X_n,k(t)·b_k(t) + u_n(t), where the X include both alpha attributes and risk-control factors. It is APT-shaped, but it may also forecast specific returns.
  - Separate benchmark from residual by one of two choices:
    - include predicted beta and benchmark-neutralize every other attribute (benchmark-weighted average = 0); or
    - use industry/sector dummies, which form an intercept.
  - Choose one of beta and intercept. If you use both, beta-adjust the sector exposures.

## Key Formulas
- **Equity value (9.1)**: value = operational value + financial value.
- **DDM (9.2–9.3)**: p(0) = Σ_t d(t)/(1+y)^t. Under certainty, y = i_F.
- **Constant-growth DDM (9.5)**: p(0) = d(1)/(y − g).
- **Return decomposition (9.7)**: y = i_F + f = d/p + g. Expected total return = yield + expected capital appreciation, and capital appreciation equals dividend growth equals earnings growth.
- **Growth model (9.13–9.15)**:
  - g = (1 − κ)·ρ, where κ = payout ratio, 1 − κ = reinvestment rate and ρ = return on reinvested capital.
  - If e(1) = ρ·b(0), then ρ is the ROE and book value also grows at g.
- **DDM alpha (9.19)**: α_n = d_n/p_n + g_n − (i_F + β_n·f_B).
- **Implied growth (9.20)**: g\*_n = i_F + β_n·f_B − d_n/p_n.
- **Sector rescaling (9.21–9.22, reconstructed)**: g′_n = a + b·g_n, with b = Std{g\*}/Std{g} and a chosen so that Mean{g′} = Mean{g\*}. Equivalently, g′_n = Mean{g\*} + b·(g_n − Mean{g}).
- **Skill shrinkage (9.23)**: g′_n = g\*_n + c·(g_n − g\*_n).
- **Three-stage DDM (9.24–9.26)**:
  - Growth in stage 2: g(t) = g_IN + (g_EQ − g_IN)·(t − T1)/(T2 − T1).
  - Terminal value: p(T2) = d(T2+1)/(y_EQ − g_EQ), with y_EQ = i_F + β·f_B.
  - Solve for y in p(0) = Σ_{t≤T2} d(t)/(1+y)^t + p(T2)/(1+y)^{T2}.
- **NPV-mode alpha (9.33–9.34)**: α_n = (1+y_n)·(p_fair − p_mkt)/p_mkt ≈ (p_fair − p_mkt)/p_mkt. For monthly alphas the two versions differ by a factor of about 1.01.
- **Clean surplus (9.37)**: b(t) = b(t−1) + e(t) − d(t).
- **Exceptional earnings (9.38–9.39)**: e\*(t) = e(t) − y·b(t−1). They decay as e\*(t+1) = δ·e\*(t), with δ < 1.
- **Price as earnings + book (9.41, reconstructed, Ohlson form)**: p(0) = e(1)/(1+y−δ) + b(0)·(1−δ)/(1+y−δ). In general, p = c1·e(1) + c2·b(0).
- **Comparative-valuation alpha (9.44)**: α_n = (p_fitted,n − p_mkt,n)/p_mkt,n. This presumes the price moves to fitted value over the alpha horizon.
- **Benchmark-neutralizing attribute k**: X_n,k = Z_n,k − z_B,k, or the variant X_n,k = Z_n,k − β_n·z_B,k, where z_B,k = Σ h_B,n·Z_n,k.

## Key Concepts
- **Humility principle**: the market may be right and you may be wrong.
- **Golden Rule of the DDM**: alpha moves one-for-one with the growth input ("garbage in, garbage out").
- **Implied growth rate**: the growth that makes the current price fair given the yield, beta and the benchmark forecast.
- **IRR mode vs. NPV mode**: these assume the mispricing persists forever vs. vanishes in one year.
- **Clean surplus relation**: book value changes only through earnings minus dividends.
- **Comparative valuation**: a multidimensional, cross-sectional "multiples" model (a formalized investment-banker comps analysis).
- **Returns-based analysis**: a direct regression of (excess or residual) returns on attributes. The factor returns are returns to minimum-risk factor portfolios.
- **Fractile analysis in regression form**: indicator exposures (high/average, with low = −1, −1 when neutralized) test nonlinearity while controlling for sector or benchmark.

## Mental Models
- Treat a DDM as a *growth-forecast amplifier*. The model adds discipline, record-keeping and equal treatment of assets, but no information beyond g.
- Use comparative valuation when you trust that the attribute set fully determines value within a peer group. Build separate models for banks, miners, utilities, industrials, and so on.
- Use returns-based analysis when you care about predicting residual returns directly. It is the empirical, non-efficient-markets cousin of the APT.
- A DDM process gives a beneficial institutional side effect: compare the DDM portfolio to the actual portfolio to measure the effect of judgmental overrides and implementation costs.

## Anti-patterns
- **Feeding raw consensus growth into a DDM**: it is upward-biased and wildly dispersed. MMI 5-year EPS growth ranged from −37.5% to +23%.
- **Believing the three-stage DDM fixes bad growth forecasts**: it only smooths them.
- **Rescaling growth by sector when you are trying to time sectors**: it strips out sector alphas.
- **Least-squares on raw attributes with outliers**: pull outliers in to ±3 standard deviations so that one or two suspect observations do not drive the fit.
- **Mixing beta and an intercept naively**: the benchmark return is split ambiguously between them.
- **Over-relying on one valuation approach**: "total devotion to the DDM" is tunnel vision.

## Reference Tables
**Table 9.1 – Implied growth rates, MMI stocks, Dec 1992** (i_F = 3.1%, f_B = 6%; g\* = 3.1 + 6β − yield)

| Stock | Yield | Beta | g\* |
|---|---|---|---|
| American Express | 4.00% | 1.21 | 6.36% |
| AT&T | 2.60% | 0.96 | 6.26% |
| Chevron | 4.70% | 0.45 | 1.10% |
| Coca-Cola | 1.30% | 1.00 | 7.80% |
| Disney | 0.50% | 1.24 | 10.04% |
| Dow Chemical | 4.50% | 1.11 | 5.26% |
| DuPont | 3.70% | 1.09 | 5.94% |
| Eastman Kodak | 4.90% | 0.60 | 1.80% |
| Exxon | 4.70% | 0.47 | 1.22% |
| General Electric | 2.90% | 1.31 | 8.06% |
| General Motors | 2.50% | 0.90 | 6.00% |
| IBM | 9.60% | 0.64 | −2.66% |
| International Paper | 2.50% | 1.16 | 7.56% |
| Johnson & Johnson | 1.80% | 1.15 | 8.20% |
| McDonald's | 0.80% | 1.07 | 8.72% |
| Merck | 2.30% | 1.09 | 7.34% |
| 3M | 3.20% | 0.74 | 4.34% |
| Philip Morris | 3.40% | 0.97 | 5.52% |
| Procter & Gamble | 2.10% | 1.01 | 7.06% |
| Sears | 4.40% | 1.04 | 4.94% |

(The columns were realigned from garbled text. Every row checks against the formula.) Reasonable growth rates average about **5.5%**, with a cross-sectional standard deviation of about **3.1%**.

**Table 9.2 – Raw 5-year historical EPS growth for the MMI**: mean ≈ 4.5% but standard deviation ≈ 23%. Rescaling compresses these to the implied-growth distribution before computing alphas. The per-stock values are too garbled to reproduce.

**Three-stage DDM sensitivities** (illustrative case):
- Inputs: T1 = 5, T2 = 15 years; payout 0 → 0.3; f_B = 5.5%; β = 1.2; i_F = 4%, so the fair return is 10.6%; EPS(1) = $4.80; price = $50.
- Implied g_IN = 12.62%.
- Alpha ≈ 0.54 × (g_IN − 12.62%), i.e. "g in, 0.54·(g − 12.62%) out".
- About 30 bp of alpha per $0.20 of EPS(1).

**Popular U.S. equity signals (Merrill Lynch survey)**: EPS variability, ROE, foreign exposure, beta, P/E, P/S, size, estimate revisions, neglect, rating revision, dividend yield, EPS momentum, projected growth, low price, duration, relative strength, EPS torpedo, estimate dispersion, D/E, EPS surprise, DDM, P/CF, P/B. Most managers use 5–7 of them.

## Worked Example
**Constant-growth DDM, fair-pricing check (Fig. 9.2)**: i_F = 4%, f_B = 6%, β = 1.2, so the required return is 4 + 7.2 = 11.2%. With a yield of 2.5%, the stock is fairly priced (α = 0) at g = 11.2 − 2.5 = **8.7%**. A growth input of 9.7% gives α = +1%.

**End-of-chapter problems**
- *Problem 3*: β = 1.2, no dividend, i_F = 6%, f_B = 6%. Then g\* = 6 + 7.2 − 0 = **13.2%**.
- *Problem 5*: P/E = 15, payout = 0.5, expected return 12%. Yield = 0.5/15 = 3.33%, so g = 12 − 3.33 = **8.67%**.

## Key Takeaways
1. α_DDM = yield + g − (i_F + β·f_B). Every 1% of growth error is 1% of alpha error.
2. Compute implied growth rates first and use them to discipline (rescale or shrink) analyst growth.
3. Decide the horizon assumption explicitly. IRR mode assumes the mispricing persists, and NPV mode assumes it is gone in a year.
4. Comparative valuation and returns-based analysis need only current information and avoid DDM growth problems, but they risk omitted-attribute bias.
5. In returns-based regressions, benchmark-neutralize attributes, control the benchmark with beta *or* industries, and winsorize at ±3σ.
6. Signals can be *consistent* (always-positive factor returns, so tilt) or *thematic* (timed, e.g. growth or size). Even consistent signals can need timing, e.g. momentum is often negative in January because of window dressing and tax-loss selling.

## Connects To
- **Ch 7 (APT)**: returns-based analysis has the APT form.
- **Ch 8**: the theory these ad hoc methods approximate, including the κ/δ horizon logic behind IRR vs. NPV mode.
- **Ch 10**: the skill-shrinkage of growth, g′ = g\* + c·(g − g\*), is the basic forecasting formula.
- **Ch 12 / Ch 17**: evaluating the IRs of signals and of the combined strategy.
- **External**: Gordon growth model; Ohlson residual-income / clean-surplus model; EVA.
