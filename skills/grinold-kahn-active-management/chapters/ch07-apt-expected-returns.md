# Chapter 7: Expected Returns and the Arbitrage Pricing Theory

## Core Idea
The APT says every stock's expected excess return is a linear function of its factor exposures (f = X·m), but it does not tell you the factors or the factor forecasts. Finding a *qualified* factor model is easy (any good risk model works). The *factor forecasts* are the hard part, and they need skill plus factors you have intuition about. That flexibility makes the APT a poor model of consensus returns and a good model for a manager's own active views.

## Frameworks Introduced
- **Arbitrage Pricing Theory (APT)**: postulate a K-factor model of excess returns, then rule out quasi-arbitrage. Expected excess returns must then be explained by factor exposures times factor forecasts m_k.
  - When to use: as a structure for a manager's *own* expected-return forecasts, not as the consensus (the CAPM plays that role).
  - How: (1) pick a factor model (X, F, Δ); (2) check that it is qualified; (3) forecast m_k for each factor; (4) f_n = Σ_k X_n,k·m_k; (5) alpha = APT forecast − CAPM forecast (β_n·f_B).
  - Why it works: if f ≠ X·m, you could build an active position with zero factor exposure (so almost no risk), positive expected return and zero net investment. That is a *quasi-arbitrage opportunity*. Failure mode: the theory is silent on the factors and on m. It has been called the "arbitrary pricing theory". Its rigor gives no guarantee that your m_k are right.
- **Qualified model (portfolio Q diversified w.r.t. the factor model)**: a factor model explains expected returns *if and only if* portfolio Q has minimum risk among all portfolios with Q's factor exposures (Proposition 1, technical appendix).
  - Practical test: highly diversified portfolios should have almost all of their variance explained by the factors. *Any factor model that is good at explaining the risk of a diversified portfolio should be (nearly) qualified as an APT model.*
- **"The Easy Part / The Hard Part"**: qualifying a model is easy. Factor forecasts are hard, and structure helps.
  - How: build the model *opportunistically* from your strengths. Use factors you can forecast (e.g. bond beta means you are forecasting interest rates, and a growth factor means you are forecasting growth stocks). Then round the model out with industries to capture the risk. "There is no use building the world's greatest (most qualified) APT model if we cannot come up with the factor forecasts."
- **Six implementation archetypes**:
  - *Structural*, preferred by practitioners because the factors link to intuition:
    - (1) Given exposures, estimate factor returns (BARRA style, with cross-sectional regression).
    - (2) Given factor returns (NYSE, gold, bonds, FX basket, commodities), estimate exposures by time-series regression of (r_n − r_NYSE) on the other factors. NYSE exposure is set to 1.
    - (3) A hybrid of 1 and 2.
  - *Statistical*, preferred by academics because it avoids prejudgment:
    - (1) Principal components. With 50 stocks over 200 months, about 20 components explain 90% or more of the risk. The factors are abstract, so the only defensible m_k is the historical mean.
    - (2) Maximum-likelihood factor analysis. With 500 stocks × 60 months = 30,000 returns, it estimates 5,000 exposures and 600 factor returns. Exposures are assumed constant for 5 years.
    - (3) The dual of (2), after Connor–Korajczyk. Use the T×T covariance instead of N×N (because N ≫ T), which swaps the roles of exposures and factor returns.

## Key Formulas
- **Factor model (Eq. 7.1)**: r_n = Σ_k X_n,k·b_k + u_n
  - X_n,k = exposure (factor loading), known at the start of the period. b_k = factor return. u_n = specific (idiosyncratic) return.
  - Assumes Cov{b_k, u_n} = 0. Uncorrelated specific returns across stocks are convenient but not required.
- **APT expected return (Eq. 7.2)**: f_n = E{r_n} = Σ_k X_n,k·m_k, where m_k = factor forecast. In matrix form, f = X·m. This implies E{u_n} = 0 under the APT.
- **CAPM as a special case**: K = 1, X_n,1 = β_n, m_1 = f_M (expected market excess return).
- **Covariance**: V = X·F·Xᵀ + Δ, where F = K×K factor covariance and Δ = specific covariance (usually diagonal).
- **Portfolio exposures**: x_P = Xᵀ·h_P.
- **Link to portfolio Q** (from Ch 2): f = (f_Q/σ_Q²)·V·h_Q. A first guess is m* = (f_Q/σ_Q²)·F·Xᵀ·h_Q, which leaves a residual term (f_Q/σ_Q²)·Δ·h_Q. That residual vanishes exactly when Q is diversified w.r.t. (X, F, Δ).
- **Intercept**: the model contains an intercept if e = X·g for some g (e.g. it has industry dummies). Then the fraction invested in risky assets is readable from factor exposures alone. A model without an intercept admits a fully invested portfolio with zero exposure to every factor, which is strange.

## Key Concepts
- **Factor forecast (m_k)**: the expected excess return attached to one unit of exposure to factor k.
- **Quasi-arbitrage**: a zero-investment position with positive expected return and negligible (not zero) risk. Ruling these out yields the APT.
- **Qualified model**: a factor model for which some m exists such that f = X·m.
- **Diversified with respect to a factor model**: minimum risk among all portfolios with the same factor exposures.
- **Portfolio Q**: the fully invested portfolio with the highest Sharpe ratio. Every expected return is proportional to the asset's beta with respect to Q. The CAPM is the claim "Q = the market".
- **Structural vs. statistical APT models**: structural factors have names and meaning, so they can be forecast with intuition and sanity-checked. Statistical factors are abstract, which leaves almost no latitude for forecasting.
- **Standardized exposure**: (attribute − market average) / cross-sectional std. Mean is 0, about 2/3 of stocks fall in [−1, +1], and about 5% fall beyond ±2.
- **Backcast**: using the historical average of factor returns as the forecast. This implicitly assumes stationarity.

## Mental Models
- Treat the APT as a *template for active views*. If every manager shared the same APT information, it would be worthless.
- Factor-level history beats stock-level history, because stocks "change their stripes". Knowing how growth stocks have done says more than the past returns of a stock that is only *now* a growth stock.
- Reducing 1,000 stock forecasts to 10 factor forecasts is not a free lunch. By the Fundamental Law, you need about √(1000/10) = 10× the skill (IC) per factor forecast to keep the same value added.
- Stories, implementations and theory are three different things. Macro "stories" do not mean the APT requires macro factors.

## Anti-patterns
- **Using the APT as the consensus model**: it is too flexible. Two equally informed modelers will build different models.
- **Building a statistically perfect model you cannot forecast**: qualification is cheap, and forecastability is what adds value.
- **Forecasting statistical (PCA) factors away from their historical mean**: there is no intuition to justify it.
- **Tilting toward high-beta stocks to time the market**: in 1983–84 high-beta stocks had negative residual returns while the market rose. Futures-based benchmark timing avoids the residual bet.
- **Treating the CAPM as gospel or dismissing it**: it is a guideline. The market still plays the central role in expected returns.

## Reference Tables
**CAPM failure, Dec 1982 + 24 months (S&P 500 beta deciles)**: predicted beta vs. realized beta had slope 0.93 and R² = 0.89, so beta forecasts were accurate. Alphas were nonetheless significantly nonzero and *perversely* (negatively) related to beta, most likely because of interest-rate and inflation changes.

**Table 7.3 – Unexplained variance of index portfolios in the BARRA U.S. Equity model (Dec 1992)**

| Portfolio | # Assets | Unexplained variance |
|---|---|---|
| MMI | 20 | 2.51% |
| OEX | 100 | 0.99% |
| S&P 500 | 500 | 0.30% |
| FR1000 | 1000 | 0.21% |
| FR3000 | 3000 | 0.18% |
| ALLUS | ~6000 | 0.17% |

(Row alignment reconstructed from garbled text. The FR3000 is 99.82% factor-explained.) The implication is that there is almost no room to cut risk while keeping the same exposures, so the risk model is qualified.

## Worked Example
**Illustrative APT forecasts for Major Market Index stocks (Dec 1992)**: the factors are industry, growth, bond beta, size (ln cap) and ROE, all standardized over 1,000+ stocks.
- Factor forecasts: growth +2.0%, bond beta +2.5%, size −1.5%, ROE 0.0%. Industries: chemical 8%, all others 6%. The CAPM uses f_M = 6%.
- Most APT forecasts came in *below* CAPM forecasts. A notable case is Sears: APT 1.42% vs. CAPM 6.60%, giving a residual of −5.18%. APT forecasts are not required to average to the CAPM forecasts.

**End-of-chapter Problem 5 (GE alpha)**
- Forecasts: heavy electrical industry 6.0%, growth 2.0%, bond beta −1.0%, size −0.5%, ROE 1.0%. Benchmark expected excess return is 6.0%.
- GE data, as best reconstructed from the garbled text: growth −0.24, bond 0.13, size 1.56, ROE 0.15, β = 1.10.
- APT forecast = 6.0 + 2.0(−0.24) − 1.0(0.13) − 0.5(1.56) + 1.0(0.15) = 6.0 − 0.48 − 0.13 − 0.78 + 0.15 = **4.76%**
- CAPM forecast = 1.10 × 6.0 = 6.60%, so **α ≈ −1.84%**. The exposure values are uncertain because of PDF extraction.

## Key Takeaways
1. f = X·m. Expected returns come from exposures times factor forecasts. The theory guarantees m exists but not how to find it.
2. Any risk model that explains more than 99% of the variance of diversified portfolios is effectively a qualified APT model.
3. Spend effort on factor forecasts, and pick factors you have real insight into (rates, growth, commodities, etc.).
4. Simple practical forecast: the trailing average of factor returns (Haugen–Baker use a 12-month trailing mean). This is a backcast, valid only under stationarity.
5. Alpha from an APT model = APT forecast − β·f_B (consensus).
6. A constant tilt, i.e. a constant active exposure c to a factor, implies a constant positive forecast for that factor.

## Connects To
- **Ch 2 (CAPM)**: portfolio Q and the proportionality of expected returns to beta with respect to Q.
- **Ch 3 (Risk)**: the APT factor model has the same structure as the structural risk model (Eq. 3.16).
- **Ch 6 (Fundamental Law)**: fewer forecasts need proportionally more skill.
- **Ch 9**: returns-based analysis is an APT-form model that can also carry specific-return signals.
- **Ch 11**: if you forecast one factor, the other factors should get nonzero forecasts too.
- **External**: Ross (1976); Fama–French (1992); Chen–Roll–Ross macro factors; Connor–Korajczyk asymptotic PCA.
