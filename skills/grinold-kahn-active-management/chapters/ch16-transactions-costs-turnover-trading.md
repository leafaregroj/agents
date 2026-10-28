# Chapter 16: Transactions Costs, Turnover, and Trading

## Core Idea
Transactions costs are the "death by a thousand cuts" that explain most of active managers' average 1–2%/yr underperformance; control them by (1) forecasting costs stock-by-stock, (2) cutting turnover intelligently along the concave value-added/turnover frontier (≥75% of value added for 50% of turnover), and (3) treating trade scheduling as its own optimization that swaps market impact for short-term risk.

## Frameworks Introduced
- **Market microstructure view of costs**: liquidity suppliers charge for (a) the risk that you are an *informed trader* (inferred from trade size and urgency) and (b) *inventory risk* while they wait for an offsetting trade. Both imply market impact rises with trade size and speed.
  - When to use: explaining why aggressive/large trades cost more; justifying patient trading.
- **Implementation shortfall** (Treynor, embellished by Perold): compare returns of the *paper portfolio* (desired portfolio executed instantly at decision prices, zero cost) with the actual portfolio.
  - When to use: ex post measurement of total trading cost.
  - How: record decision prices for every intended trade; difference captures commissions + spread + market impact + **opportunity cost** of trades never executed (Wagner: opportunity costs often dominate).
  - Failure mode of alternatives: **VWAP** comparison measures impact crudely, ignores unfilled orders entirely, and traders can game it.
- **Inventory risk model of market impact**: time to clear → inventory risk → price concession proportional to risk. Yields the square-root law of impact.
  - How: estimate time to clear from trade size vs. daily volume; convert stock volatility to that horizon; multiply by a market-set price of risk c. Industrial versions (BARRA market impact model) use submodels for volatility, volume/trade intensity (size and frequency, intraday/holiday patterns), and **elasticity** (how buyers vs. sellers respond as price moves from equilibrium), with separate prices of risk for buys/sells and exchange vs. OTC.
  - Why it works: structural decomposition sidesteps the censored, noisy, sparse tick data problem and extends to illiquid/new assets.
- **Value-added/turnover frontier VA(TO)**: maximum value added achievable with turnover ≤ TO; increasing and concave (diminishing marginal value of turnover).
  - When to use: setting turnover limits, sanity-checking cost assumptions, retaining value from a high-turnover signal.
  - How: compute portfolio Q (best unconstrained-turnover portfolio), TO_Q, ΔVA_Q; apply the lower bound; schedule the highest alpha-per-cost trades first to beat it.
- **Implied transactions cost**: the slope of the frontier at a turnover limit is the round-trip cost that limit implicitly assumes. Reconcile an ad hoc "≤20% turnover per quarter" policy with your actual cost estimate.
- **Trading as portfolio optimization** (distinct from portfolio construction): choose a path of intermediate portfolios from current to target, maximizing short-term alpha − short-term risk penalty − market impact, benchmarked to immediate execution.

## Key Formulas
- **Time to clear** (16.1): `τ_clear ≈ |V_trade| / V̄_daily` (days); V̄ = average (or forecast) daily volume.
- **Inventory risk** (16.2): `σ_inv = σ · √(τ_clear / 250)`; σ = annual volatility, 250 trading days.
- **Price concession** (16.3): `impact ≈ c · σ_inv`, measured from the bid (sells) or offer (buys); c = liquidity supplier's risk/return trade-off, set competitively.
- **Cost model** (16.4): `Cost (return units) ≈ commission + c_tc · √(|V_trade| / V̄_daily)`; c_tc bundles volatility, price of risk, annual→daily conversion.
  - Rule of thumb: **it costs about one day's volatility to trade one day's volume** (≈ σ/√250 at V_trade = V̄; reconstruction of the implied c_tc).
  - Impact per share ∝ (size)^½ → total cost ∝ (size)^{3/2} (matches Loeb 1983 block-bid data; BARRA found exponent ½ fits best).
  - Simple calibration: pick c_tc so typical-size trades cost ~2% round trip, or fit to the manager's own trade history; linearize/quadratize near expected trade sizes if the optimizer requires.
- **Value added**: `VA_P = α_P − λ_A · ψ_P²`; `ΔVA_Q = VA_Q − VA_I`.
- **Turnover**: purchase `TO⁺ = Σ max(0, h'_n − h_n)`, sales `TO⁻ = Σ max(0, h_n − h'_n)`; **TO = min(TO⁺, TO⁻)** (excludes contributions/withdrawals).
- **Lower bound** (16.10), equality-constrained choice set containing I:
  `VA(TO) ≥ VA_I + ΔVA_Q · (TO/TO_Q) · (2 − TO/TO_Q)`
  achieved by prorating every I→Q trade by fraction TO/TO_Q. At TO = ½TO_Q → ¾ of ΔVA_Q; effective IR retained ≈ √0.75 ≈ 87%.
- **Optimal turnover with uniform cost** (16.11): maximize `VA_P − TC · TO_P` → choose TO* where `SLOPE(TO*) = TC`; if `TC > SLOPE(0)`, don't trade.
- **Trading utility** (16.12): `U = short-term α − short-term risk adjustment − market impact`, relative to immediate execution. One-stock example (reconstructed, h(t) = stock fraction, h(0)=0, h(T)=1):
  - active return `∫₀ᵀ (h(t) − 1) f dt`; active risk `∫₀ᵀ (h(t) − 1)² σ² dt`; impact `∫₀ᵀ χ (dh/dt)² dt` (quadratic in trading speed, no memory).
  - Zero alpha: impact-dominated `(gT)² ≪ 1` → straight line `h(t) = t/T` (uniform trading); risk-dominated `(gT)² ≫ 1` → `h(t) ≈ 1 − e^{−gt}` (front-loaded), with g ∝ √(risk aversion · σ² / impact coefficient). Other regimes: `sT² ≫ 1` (positive alpha dominates impact → trade early), `sT² ≪ −1` (negative alpha → trade late), `|s/g²|` compares alpha vs. risk aversion.

## Key Concepts
- **Commissions**: per-share broker charge; smallest and most measurable cost.
- **Bid/ask spread**: approximately the cost of trading one share.
- **Market impact**: incremental cost of trading many shares vs. one; unobservable a priori (the "Heisenberg" problem—every trade alters the market).
- **Opportunity cost**: return lost on trades never completed while waiting for price.
- **Informed trader**: a trader whose size/urgency signals private information, prompting larger concessions.
- **Inventory risk**: liquidity supplier's risk of holding the position until it clears.
- **Censored data**: tick data show trades done, not orders withheld because cost was too high → realized costs underestimate expected costs.
- **Choice set CS**: closed convex set of allowed portfolios (full investment, β=1, no shorts...).
- **Implied transactions cost**: SLOPE(TO_R) of the VA/turnover frontier at a turnover restriction.
- **Tightness / depth / resiliency** (see Ch 21): spread / impact / recovery speed.

## Mental Models
- Think of turnover as a resource with diminishing marginal returns: the first half of turnover buys three-quarters of the value.
- Use the implied-cost check when a turnover cap and a cost estimate disagree: if SLOPE(TO_R) = 4.5% but you believe costs ≈ 2%, either raise the cost estimate, loosen the cap, or shrink alphas—or a bit of each.
- In trade scheduling: risk pushes toward immediate execution, market impact toward even spacing, alpha toward early (or late) execution.
- Cost dispersion is an opportunity: better discrimination among stocks (by alpha *and* cost) is what lets you beat the lower bound.

## Anti-patterns
- **Ignoring costs because "the stock will double"**: forecasts are often wrong; you pay costs on winners and losers alike. A top-quartile (IR 0.5) manager can lose roughly half his returns to costs.
- **Judging traders by VWAP**: gameable, misses unfilled orders and true impact.
- **Uniform cost assumption**: forgoes the ~30% cost savings from distinguishing high- vs. low-cost stocks.
- **Dismissing high-turnover strategies outright**: an 80%/month strategy may add substantial value capped at 10–40%.
- **Heavy use of limit orders**: they give the market free options and can fill asymmetrically in big moves (all sells fill, no buys → unintended market exposure). Use sparingly, for the highest-impact stocks, priced near the market (value managers may legitimately use them to *sell* liquidity).
- **Counting realized tick-data costs as expected costs**: censoring biases them low.

## Reference Tables
**Table 16.1 (S&P 100 example; Q: α=3.2%, ψ=4%, λ_A=0.1 → VA_Q=1.60; I: 20 random equal-weighted stocks, α=0.07%, ψ=5.29% → VA_I=−2.73)**

| % of TO_Q | Lower bound | Excess (scheduling) | Total VA | % of ΔVA_Q | Implied round-trip TC |
|---|---|---|---|---|---|
| 0 | −2.73 | 0.00 | −2.73 | 0.0% | — |
| 10 | −1.91 | 0.87 | −1.03 | 39.2% | 8.66% |
| 20 | −1.17 | 1.00 | −0.18 | 59.0% | 5.12% |
| 30 | −0.52 | 0.88 | 0.36 | 71.4% | 3.40% |
| 40 | 0.04 | 0.70 | 0.74 | 80.2% | 2.50% |
| 50 | 0.52 | 0.51 | 1.03 | 86.8% | 1.90% |
| 60 | 0.91 | 0.34 | 1.24 | 91.8% | 1.43% |
| 70 | 1.21 | 0.19 | 1.40 | 95.5% | 1.02% |
| 80 | 1.43 | 0.09 | 1.51 | 98.0% | 0.67% |
| 90 | 1.56 | 0.02 | 1.58 | 99.5% | 0.33% |
| 100 | 1.60 | 0.00 | 1.60 | 100% | 0.00% |

(Column alignment reconstructed from a garbled extraction; the lower-bound column checks exactly against Eq. 16.10.) Scheduling excess peaks near 20% of TO_Q. Realistic ~2% round-trip costs imply modest turnover; very tight or very loose caps imply unrealistic costs.

## Worked Example
- **Lower bound check** (Table 16.1): ΔVA_Q = 1.60 − (−2.73) = 4.33. At 50% of TO_Q: −2.73 + 4.33 × 0.5 × 1.5 = 0.52 (bound). Actual with best-trades-first: 1.03 → 86.8% of incremental VA for half the turnover (rule of thumb is conservative).
- **Scheduling**: I→Q requires buying 10% each of assets 1, 2 (α = 5%, 3%) and selling 10% each of 3, 4 (α = −3%, −5%); TO_Q = 20%. With a 10% cap, rather than 5% of each swap, do 8% of the 1-for-4 swap and 2% of 2-for-3.
- **Differing costs**: set costs to 1.42% for half the stocks and 2.26% for the rest (average unchanged at the 1.90% implied at 50% TO_Q) → alphas and risk barely change, costs fall ~30%.
- **Problem 3 logic**: 200 bp VA at 200% turnover → ≥150 bp at 100% turnover. For 100 bp: x(2 − x) = 0.5 → x ≈ 0.29 → ~59% turnover.
- **Trading schedule** (Fig. 16.5, T = 5 days): risk-dominated case reaches 75% of target within 2 days (40% of the period); impact-dominated case trades evenly.

## Key Takeaways
1. Average active U.S. equity underperformance of 1–2%/yr vs. 2–3% targeted active return (≈0.5% fees) is essentially a transactions cost story.
2. Measure costs with implementation shortfall; forecast them with an inventory-risk (square-root) model.
3. Impact ∝ √(trade size / daily volume); total cost ∝ size^{3/2}. Trading one day's volume costs ≈ one day's volatility.
4. With half the turnover you keep ≥75% of incremental value added (≥87% of IR); smart scheduling and cost discrimination do better.
5. Set turnover where frontier slope = round-trip cost; use implied costs to audit turnover policies.
6. If the starting portfolio is outside the choice set, the first chunk of turnover is spent just getting back in (e.g., 4% of a 10% budget).
7. Optimal trading trades short-term risk for lower impact; high risk aversion → trade fast, high impact → trade evenly.

## Connects To
- **Ch 14**: transactions costs as an input to portfolio construction and alpha amortization.
- **Ch 5 / Ch 6**: VA ∝ IR², which converts the 75% VA rule into ~87% IR retention.
- **Ch 3**: structural risk models reused for inventory-risk estimation.
- **Ch 17**: performance attribution; implementation shortfall is a performance measure.
- **Ch 21**: open questions on tightness, depth, resiliency, and when to trade.
- **External**: Almgren–Chriss optimal execution; square-root impact law; Perold (1988) implementation shortfall.
