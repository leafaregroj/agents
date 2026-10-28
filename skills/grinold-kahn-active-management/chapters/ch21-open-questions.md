# Chapter 21: Open Questions

## Core Idea
The theory of active management is not seamless: the hard, unresolved problems are mostly *interactions*—changing alphas, risks, and costs over time—plus a few domains (transactions costs, liabilities, nonlinear size effects, taxes, behavioral finance) where no accepted model yet exists. Treat these as places where the book's static machinery is an approximation.

## Frameworks Introduced
- **Six open-question areas for the *process* of active management** (not for finding specific signals, which is every manager's perpetual open question):
  1. **Dynamics**: managing over time with flows of information and noise, time-varying volatilities and exposures, and costs that depend on trading speed.
     - Open: *When should we trade?* With zero costs, trade whenever alpha changes; with real costs, the answer requires a dynamic analysis.
     - Open: how to choose a dynamic strategy and what it does to the return distribution—dynamic strategies (e.g., portfolio insurance) create skewed, option-like payoffs even from normal single-period returns.
  2. **Transactions costs**: no "CAPM or Black-Scholes" of trading. A complete model must cover **tightness** (bid/ask spread), **depth** (market impact), and **resiliency** (how fast impact decays back to equilibrium after a block trade).
     - Open: how to model all three; how costs in one stock affect costs in others (cross-impact).
  3. **Liabilities, asset allocation, and risk over time**: liabilities are uncertain and contingent (inflation, mortality).
     - Open: accurate liability modeling; managing assets against fixed-horizon liabilities (glide path into and past retirement).
  4. **Nonlinearities**: (a) empirically nonlinear, multidimensional behavior along **size** (small vs. midcap vs. large vs. megacap), which pervades benchmarks, market averages, and long-only constraints; (b) whether nonlinear dynamic models (chaos, catastrophe theory) have predictive power.
  5. **After-tax investing**: dynamics plus wash-sale rules, holding-period-dependent rates, and calendar effects (respond differently to alphas received in January vs. December). Never solvable exactly; current practice is one-period optimization with a penalty on net realized capital gains.
     - Open: how good are these simple approaches, and how much better can we do?
  6. **Behavioral finance**: investors are irrational in systematic, cataloged ways; so far this mainly explains why known anomalies (e.g., residual reversal) persist.
     - Open: does it have *predictive* power—can psychology's first principles generate new strategies?

## Key Formulas
None new. The relevant static tools this chapter flags as incomplete:
- One-period after-tax objective (current practice, schematic): `maximize VA − tax penalty × net realized capital gains`.
- Static trading-cost model (Ch 16): `cost ≈ commission + c_tc·√(trade/daily volume)` — captures depth only, not resiliency or cross-stock effects.

## Key Concepts
- **Tightness**: the bid/ask spread dimension of liquidity.
- **Depth**: the market impact dimension—price concession for size.
- **Resiliency**: speed with which prices recover after a trade's impact.
- **Dynamic strategy**: a rule that changes holdings in response to market moves; can produce option-like return distributions.
- **Wash sale rule**: tax rule that disallows a loss when the same security is repurchased within a short window; complicates tax-loss harvesting.
- **Contingent liabilities**: obligations whose size depends on uncertain events (inflation, survival).
- **Size nonlinearity**: returns/risk not linear in log-cap; behavior differs across size segments.

## Mental Models
- Treat every static optimization in the book (alpha vs. risk vs. cost in one period) as a myopic approximation to a dynamic problem.
- Think of liquidity as three-dimensional (tightness, depth, resiliency), not a single cost number.
- Use behavioral finance as an explanation for why anomalies survive, not (yet) as a proven source of new forecasts.

## Anti-patterns
- **Assuming the framework is complete**: it sweeps aside dynamics, cost interactions, and taxes.
- **Trading every time alpha changes**: correct only with zero costs.
- **Assuming normal single-period returns imply normal strategy returns**: dynamic rules create skew.
- **Treating size as a linear factor**: its effects are nonlinear and multidimensional.
- **Ignoring calendar and holding-period tax effects** for taxable clients.

## Reference Tables
| Area | Open question(s) |
|---|---|
| Dynamics | When to trade given dynamic returns, risks, costs; choosing dynamic strategies and their return distributions |
| Transactions costs | Modeling tightness, depth, resiliency; cross-stock cost effects |
| Liabilities / risk over time | Liability modeling; allocation approaching and after retirement |
| Nonlinearities | Capturing nonlinear size behavior; predictive power of chaos/catastrophe models |
| After-tax investing | Quality of one-period capital-gains penalty approaches; size of possible improvement |
| Behavioral finance | Predictive (not just explanatory) power |

## Worked Example
The authors give no numeric example. Illustration of the "when to trade" problem using Ch 16 tools: if a new alpha implies a trade whose round-trip cost (~2%) exceeds the marginal value added from the frontier slope at zero turnover, the static rule says don't trade—but if the alpha is expected to persist and costs to fall (more volume tomorrow), a dynamic rule might trade partially now and more later. Resolving this properly requires a joint model of alpha decay (Ch 13), risk, and cost resiliency.

## Key Takeaways
1. The search for superior information is always open for each manager; the open questions here concern the *process*.
2. Dynamics—especially *when to trade*—is the most important unsolved interaction.
3. Transaction cost modeling needs tightness, depth, and resiliency, plus cross-stock effects.
4. Liability-relative and horizon-dependent allocation remain poorly modeled.
5. Size effects are nonlinear; after-tax management is only approximately solved.
6. Behavioral finance explains anomalies; whether it predicts new ones is unknown.

## Connects To
- **Ch 13**: information horizon and alpha decay—inputs to dynamic trading.
- **Ch 14**: one-period portfolio construction being generalized.
- **Ch 16**: static cost models and trade scheduling.
- **Ch 18**: strategic asset allocation and liabilities (not addressed in the book).
- **Ch 20**: behavioral finance as a reason to believe in active management.
- **External**: Hilbert's problems (inspiration for the list); Kyle (1985) market depth/resiliency; Garleanu-Pedersen dynamic trading; tax-aware portfolio management.
