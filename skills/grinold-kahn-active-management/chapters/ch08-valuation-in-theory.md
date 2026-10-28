# Chapter 8: Valuation in Theory

## Core Idea
Value = the risk-adjusted expectation of future cash flows, discounted at the *risk-free* rate. The risk adjustment uses market-wide "value multiples" that depend only on portfolio Q's return. This unifies option pricing, the CAPM and the APT. Misvaluation turns into alpha according to how big it is (κ) and how fast the market corrects it (δ).

## Frameworks Introduced
- **Modern theory of valuation (risk-adjusted expectations)**: p(0) = Σ_t E*{cf(t)}/(1+i_F)^t, where E* uses risk-adjusted (martingale) probabilities π*(t,s) = π(t,s)·υ(t,s).
  - When to use: as the yardstick for judging ad hoc valuation models (Ch 9), and to convert a perceived mispricing into an expected-return forecast.
  - How: specify states s, probabilities π and cash flows cf(t,s). Apply the value multiples υ(t,s), which are positive, have E{υ} = 1, and are a function of portfolio Q's return. Then discount at i_F.
  - Why it works: the multiples exist whenever there is no arbitrage (proof by LP duality). They are *market-dependent, not stock-dependent*, so the same υ and i_F value IBM stock, GM puts or the S&P 500.
- **Five properties a valuation formula must have**:
  1. Nonnegative cash flows give a nonnegative value.
  2. Scaling cash flows scales value.
  3. Additivity, so values combine into portfolios and cash-flow streams can be split.
  4. It reduces to plain PV discounting when cash flows are certain.
  5. It agrees with market prices.
  - Property 5 is where the active manager and the economist part company. The manager hunts for stocks where it fails, accepting it only "on average" or within groups.
- **Misvaluation-to-alpha framework (κ, δ)**:
  - κ = the percentage gap between fitted (fair) and market price today.
  - δ = the fraction of the misvaluation that persists after one year.
  - Alpha is positive only if κ ≠ 0 and δ < 1. If the market never learns, there is no profit.
  - Half-life of misvaluation = −0.69/ln(δ) years. A "fast idea" has δ near 0. A "slow idea" has δ near 1.
- **Risk-adjusted discount rate vs. risk-adjusted expectations**: the practical alternative, discounting E{cf} at i_F + β·f_Q (the *equity cost of capital*), is simple but fails for cash flows with zero expected value. For example, a ±$100,000 coin toss always values to 0. The risk-adjusted-expectations approach has no such failure.

## Key Formulas
- **Certain cash flows (8.1–8.2)**: p(0) = Σ_t cf(t)/(1+i_F)^t.
- **Naive approach fails (8.4–8.5)**: Σ_t E{cf(t)}/(1+i_F)^t usually *exceeds* the price because it ignores risk.
- **Risk-adjusted discount rate (8.6–8.7)**: E{cf(1)}/p(0) = 1 + i_F + β·f_Q, so p(0) = E{cf(1)}/(1 + i_F + β·f_Q).
- **Risk-adjusted expectation (8.8)**: E*{cf(t)} = Σ_s π(t,s)·υ(t,s)·cf(t,s) = E{υ(t)·cf(t)}.
- **Valuation formula (8.9)**: p(0) = Σ_t E*{cf(t)}/(1+i_F)^t.
- **Covariance form (8.14)**: E*{cf} = E{cf} + Cov{υ, cf}. The covariance is usually negative and is the explicit risk penalty.
- **Expected-return form (8.16)**: E{R_n} − (1+i_F) = f_n = −Cov{υ, R_n}. Under the modified measure, E*{R_n} = 1 + i_F for every stock (8.17).
- **Value multiple in terms of Q (Prop. 2, reconstructed)**: υ = 1 − (f_Q/σ_Q²)·(r_Q − f_Q).
  - It decreases as the market rises. Good-times cash flows are worth less and bad-times cash flows are worth more.
  - With f_Q/σ_Q² ≈ 1.5–2.0 (use 1.75), υ < 0 only if r_Q > f_Q + 0.57, i.e. about a 63% excess market return, which is more than a 3σ event.
- **Portfolio S**: the portfolio (risky plus cash) with minimum second moment E{R_P²}.
  - S is a mix of F (cash) and Q (Prop. 4).
  - Expected returns satisfy f_P = −Cov{R_P, R_S}/E{R_S} (Prop. 3, reconstructed).
  - V·h_S = −E{R_S}·f.
  - υ = R_S/E{R_S} (Prop. 5, reconstructed from the facts that υ ∝ R_S and E{υ} = 1).
  - S exists even if f_C ≤ 0, so it needs fewer assumptions than Q.
- **Misvaluation alpha (8.19–8.22, reconstructed and verified against Table 8.1)**:
  - Market price p_mkt(0) = p_fair(0)/(1+κ). Next year the market price is p_fair(1)/(1+δ·κ).
  - **α = (1+i_F)·κ·(1−δ)/(1+δ·κ)**. Strictly, the factor multiplying κ is the stock's fair expected total return. The table uses 1 + i_F = 1.06.
  - Special cases: δ = 0 gives α = (1+i_F)·κ, the full gain plus interest. δ = 1 or κ = 0 gives α = 0.

## Key Concepts
- **Value multiples υ(t,s)** (also called valuation multipliers or state-price deflators): positive, mean-one weights that rescale cash flows. Technically they are a Radon–Nikodym derivative.
- **Risk-adjusted (martingale) probabilities π\***: π·υ. Under π\*, every asset earns the risk-free rate.
- **Arbitrage (multi-period)**: a self-financing strategy with nonpositive initial wealth, nonnegative wealth in every terminal state, and positive wealth in at least one.
- **Self-financing condition**: the portfolio value before rebalancing equals the value after rebalancing.
- **Complete market**: every state-contingent payoff can be replicated. Only then are the value multiples *unique*.
- **Equity cost of capital**: i_F + β·f_Q.
- **Portfolio S**: the minimum-second-moment portfolio. Its total return is proportional to υ.
- **Misvaluation κ / persistence δ / half-life**: the size of the pricing error, the fraction surviving one year, and −0.69/ln δ.

## Mental Models
- Think of υ as a "scarcity price" of a dollar. A marginal dollar is worth more when dollars are scarce (market down), so assets that pay off in bad times are priced higher.
- Only systematic (portfolio-Q) risk is priced. Stock-specific features enter only through the cash flows cf(t,s), never through the discounting.
- Expected-return models (CAPM/APT) and valuation models are two sides of one coin. Each implies the other.
- Every alpha view has a size (κ) and a clock (δ). A large mispricing that never corrects is worthless.

## Anti-patterns
- **Discounting expected cash flows at the risk-free rate**: this overvalues risky assets. In the example it gives 50.75 vs. a true price of 50.
- **Risk-adjusting only the discount rate for odd payoffs**: it breaks when E{cf} ≈ 0, as with long/short or hedged claims.
- **Ignoring persistence**: a mispricing view without a horizon or decay assumption gives no usable alpha.

## Reference Tables
**Table 8.1 – One-year alpha for misvaluation κ and persistence δ (i_F = 6%)**

| κ \ δ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| 1% | 1.06% | 0.85% | 0.63% | 0.42% | 0.21% |
| 5% | 5.30% | 4.20% | 3.12% | 2.06% | 1.02% |
| 10% | 10.60% | 8.31% | 6.12% | 4.00% | 1.96% |
| 25% | 26.50% | 20.19% | 14.45% | 9.22% | 4.42% |
| 50% | 53.00% | 38.55% | 26.50% | 16.31% | 7.57% |

(The table was rebuilt from the formula above. All entries match the surviving digits.)

Half-life lookup: δ = 0.8 gives about 3.1 years, δ = 0.6 about 1.35 years, δ = 0.4 about 0.75 years, and δ = 0.2 about 0.43 years.

## Worked Example
**One-month binomial stock**
- p = 50. Payoffs are 49 or 53, each with probability 0.5. E{cf} = 51, std = 2. Annual i_F = 6%, so the monthly R_F = 1.06^(1/12) = 1.00487.
- Naive approach: 51/1.00487 = 50.75, which is wrong.
- Solve 50 × 1.00487 = 49·π\*_DN + 53·π\*_UP with π\*_DN + π\*_UP = 1. This gives π\*_UP ≈ 0.31 and π\*_DN ≈ 0.69.
- The value multiples are therefore υ_UP = 0.62 and υ_DN = 1.38. They average to 1.
- E\*{cf} = 0.69·49 + 0.31·53 = 50.24, and 50.24/1.00487 ≈ 50.00 ✓.
- The risk penalty is Cov{υ, cf} = 50.24 − 51 = −0.76, with present value −0.756. The rescaled cash flows are 1.38 × 49 = 67.62 and 0.62 × 53 = 32.86.
- A call struck at 50 pays 3 in UP and 0 in DN, so its value is 0.31 × 3/1.00487 ≈ 0.93.

**Problem 4 (misvaluation to alpha)**
- The stock is 25% undervalued and the half-life is 3.1 years, so δ = exp(−0.69/3.1) ≈ 0.80.
- α = 1.06 × 0.25 × 0.2/(1 + 0.8 × 0.25) = 0.053/1.2 ≈ **4.4%** for the next year.

## Key Takeaways
1. Value = Σ E\*{cf}/(1+i_F)^t. Risk adjustment goes into the probabilities, not the discount rate.
2. The adjustment is market-wide: υ depends only on portfolio Q (or S) returns. Only systematic risk is priced.
3. Valuation and expected-return models are equivalent. CAPM/APT ⇔ valuation by υ.
4. To turn a valuation view into alpha, you need both κ (size) and δ (persistence): α ≈ (1+i_F)·κ·(1−δ)/(1+δκ).
5. Slow ideas (δ near 1) deliver small annual alphas even for large mispricings. A 25% mispricing with δ = 0.8 gives only about 4.4% per year.
6. Valuation multiples are unique only in complete markets, but they exist whenever there is no arbitrage.

## Connects To
- **Ch 2 (CAPM)** and **Ch 7 (APT)**: portfolio Q drives both expected returns and the value multiples.
- **Ch 9**: the practical, ad hoc valuation methods (DDM, comparative valuation) judged against this theory.
- **Ch 13 (Information Horizon)**: δ and the half-life generalize to signal decay.
- **External**: Black–Scholes / Cox–Rubinstein binomial pricing; Harrison–Kreps martingale pricing; stochastic discount factor (SDF) asset pricing.
