# Chapter 6: The Fundamental Law of Active Management

## Core Idea
The information ratio comes from two things: **skill** (information coefficient, IC) and **breadth** (BR, the number of independent bets per year). **IR ≈ IC·√BR.** Play often and play well. A tiny edge, applied many times independently, produces a top-decile IR.

## Frameworks Introduced
- **Fundamental Law of Active Management**: IR = IC·√BR.
  - When to use: to choose between research programs, compare strategy designs (stock picking vs timing), size expectations, and scale alphas. It is a strategic guide, not an operational tool.
  - How: estimate IC as the correlation of each forecast with the realized residual return, and BR as the number of *independent* forecasts per year. Doubling IR needs 2× IC or 4× BR. A 50% rise in BR equals a 22% rise in IC (√1.5 ≈ 1.22).
  - Why it works: independent bets diversify residual risk, so expected alpha grows ∝ BR while risk grows ∝ √BR.
  - Assumptions / failure modes: (1) **the manager gauges information accurately and uses it optimally** ("competence", the most critical assumption); (2) forecasts are independent; (3) IC is equal across bets. The approximation ignores the risk reduction from knowing the signal, which is negligible for IC < 0.1.
- **Additivity of squared IRs**: IR² = Σ_i IC_i²·BR_i across stock groups, decision types (stock selection, market timing, currency, country), or managers, provided the combination is implemented optimally. Value added is the area under the "skill curve" when sources are sorted by IC.
  - When to use: to attribute value added to parts of a strategy, or to judge whether to add a new universe or signal.
  - Failure mode: indexing a new segment instead of exploiting it adds nothing (IR stays IR₁).
- **Dependence audit of forecasts**: quantify analysts' recommendations and regress them on firm attributes (industry, E/P, momentum). Common patterns reveal *single* bets disguised as many. The regression residuals are the truly independent stock-specific forecasts. Over time, monthly rebalancing on an annual view is **one** bet made 12 times, not 12 bets.
- **Investment style implication**: the law favors an **eclectic** style (use any independent information). Business pressure for a clear style works against this. In practice "managers prepare the ingredients and the sponsor makes the stew".

## Key Formulas
- **IR = IC·√BR**.
- Optimal aggressiveness: ω* = IR/(2λ_R) = IC·√BR/(2λ_R).
- Optimal value added: VA* = IR²/(4λ_R) = IC²·BR/(4λ_R). It grows linearly in breadth and with the square of skill.
- Directional forecasts (x, y = ±1): IC = 2·(N₁/N) − 1, where N₁ is the number of correct calls. So hit rate = (1 + IC)/2.
- Additivity: IR² = IC₁²·BR₁ + IC₂²·BR₂ + …
- Combining managers (optimal allocation, independent active risks): IR_total² = Σ IR_n².
- Two correlated signals, each with skill IC and inter-signal correlation γ: IC²(combined) = 2·IC²/(1 + γ).
- Simulation test: α = IC²·θ + IC·√(1 − IC²)·ω·z, with z ~ N(0,1). This gives Corr{α, θ} = IC and Var{α} = IC²·ω².
- Appendix sketch: θ = A·x (orthonormal residual basis), z = E·y (orthonormal signal basis), P = Corr{x, y}. Then α(z) = E{θ|z} = Cov{θ,z}·Var⁻¹{z}·z. The conditional residual variance is independent of z. IR² = E{yᵀPᵀ D P y} with D = (I − PPᵀ)⁻¹ ≈ I, so IR² ≈ Tr(PᵀP) = Σ_j IC_j², which equals BR·IC² when all signals are equally valuable.

## Key Concepts
- **Breadth (BR)**: the number of independent forecasts of exceptional return per year. Hard to estimate because of hidden dependence.
- **Information coefficient (IC)**: the correlation between forecasts and realized (residual) returns. It measures skill. Realistic values are small: 0.02–0.10. About 0.1 is optimistic.
- **Skill curve**: IC by information source, sorted from high to low. The law replaces it with its average.
- **Independent forecast**: one not derived from information correlated with another forecast (e.g. "growth will lag" and "high yield will lead" are largely the same bet).
- **Competence (hypercompetence)**: knowing precisely what you know and don't know, and turning it optimally into portfolios.
- **Not the law of large numbers**: the law holds at BR = 10 as well as at BR = 1,000. Breadth diversifies residual risk. It does not shrink sampling error toward a true mean.

## Mental Models
- **Think like a casino**: a small, reliable edge times huge breadth gives an enormous IR.
- Use the law as a **research budget calculator**: before a project, compare the IR gain from more coverage or shorter horizons (BR) with that from better forecasts (IC).
- Think of **market timing as low-breadth**: about 4 independent bets per year means IC must be 10× a stock picker's (0.25 vs 0.025) to reach the same IR.
- Treat **squared IR as the currency of value**: it adds across sources and managers, so evaluate a new source by its IC²·BR contribution.

## Anti-patterns
- **Counting rebalances as bets**: monthly trading on a stale annual view is still one bet.
- **Repackaged signals**: several measures of the same theme (value via E/P, B/P, yield) inflate apparent breadth. Combined IC grows only by a factor of √(2/(1+γ)).
- **Single-theme stock picks**: if all recommendations sit in one industry or factor, you have made one factor bet, not N stock bets.
- **Spurious precision**: the law is approximate. Breadth especially can't be measured precisely.
- **Relying on a narrow style for marketing**: you forgo independent information that would raise IR.
- **Ignoring constraints and costs**: no-short constraints and turnover costs pull the realized IR below IC·√BR (e.g. 1.157 falls to 0.70 after 0.8% costs).

## Reference Tables
Strategies that reach the same IR = 0.50:

| Strategy | BR/yr | Required IC |
|---|---|---|
| Market timer, quarterly | 4 | 0.25 |
| Stock selector, 100 stocks quarterly | 400 | 0.025 |
| Specialist, 2 stocks × 200 revisions/yr | 400 | 0.025 |

IC vs directional hit rate: IC = 0.02 is about 51%, and IC = 0.0577 is about 52.9%. An IC of 0.02 on 200 stocks quarterly (BR = 800) gives IR ≈ 0.56.

Roulette (casino as the active manager):

| Setup | IC | BR | Exact IR | Law IR |
|---|---|---|---|---|
| 1/37 edge, 1 bet | 0.027 | 1 | 0.027038 | 0.027027 |
| 1/37 edge, 10⁶ bets | 0.027 | 10⁶ | 27.038 | 27.027 |
| 2/38 edge, 10⁶ bets | 0.0526 | 10⁶ | 52.70 | 52.63 |

International example (London-based, 4 countries, quarterly currency and country bets = 24/yr; stocks US 400, Japan 300, UK 200, Germany 100, annual = 1,000/yr):

| Source | IC | BR | IR² |
|---|---|---|---|
| Stocks | 0.02 | 1000 | 0.40 |
| Currency + country | 0.075 | 24 | 0.135 |
| Total | | | 0.535, so IR ≈ 0.73 |

Timing would need IC ≈ 0.129 to match the stock contribution.

## Worked Example
**200-stock binary-nugget model**: each stock's semiannual residual return is the sum of 300 independent ±1% nuggets (σ = √300 = 17.32% semiannual, 24.49% annual). You know nugget 1 only, so IC = 1/17.32 = 0.0577. BR = 200 × 2 = 400, so the law predicts IR = 0.0577·√400 = **1.154**.
- Implementation: benchmark = equal weight (0.5% each). Hold the ~100 "+1%" stocks at 1% each and none of the "−1%" stocks (active ±0.5%).
- Expected active return = 100·(1/200)·1% + 100·(−1/200)·(−1%) = 1% per 6 months. Active SD = √(200·(1/200)²·299) = 1.2227%.
- 6-month IR = 0.8179, so annual IR = 0.8179·√2 = **1.157** (slightly above the law because knowing a nugget also reduces risk).
- Turnover ≈ 100%/yr. At 0.80% round-trip cost, net alpha ≈ 1.2% on 1.729% risk, so **IR ≈ 0.70**.

**Additivity**: 200 stocks semiannually with IC = 0.04 gives IR² = 0.0016·400 = 0.64 (IR = 0.80). Adding 100 stocks semiannually at IC = 0.03 adds 0.0009·200 = 0.18, so IR² = 0.82, IR = 0.906, and value added rises **28%**.
**Stock + timing**: 400 stocks annually at IC = 0.03 gives IR = 0.60. Quarterly market calls at IC = 0.10 give IR = 0.20. Combined IR = √(0.36 + 0.04) = **0.63**.
**Managers**: IRs of 0.75, 0.50, and 0.30 combine (optimally) to √(0.5625 + 0.25 + 0.09) = **0.95**.
**Correlated sources**: source 1 knows nuggets 1 and 2, source 2 knows nuggets 2 and 3. Each has IC = √2/√300 = 0.0816 and γ = 0.5, so IC_com = √(2·0.00667/1.5) = **0.0942**.
**Problems**: a picker covering 500 stocks monthly (6,000 forecasts) with IC = 0.05 and realized IR = 1.0 implies only BR = (1/0.05)² = 400 independent bets, so his alphas are highly correlated. In the exercise model, a = IC² = 0.0001 means IC = 0.01, and over 500 stocks monthly IR = 0.01·√6000 ≈ 0.77.

## Key Takeaways
1. IR = IC·√BR. Play often (breadth) and play well (skill).
2. Realistic ICs are small (0.02–0.10). Success comes from applying modest skill broadly.
3. Market or benchmark timing can't compete without very high IC, because its breadth is tiny.
4. Squared IRs add across independent sources and managers. Use IC²·BR to rank research ideas.
5. Breadth counts only *independent* bets. Audit forecasts for hidden common factors and stale repeats.
6. Correlated signals add less than you think: IC²_com = 2IC²/(1+γ).
7. The law assumes you size and use information optimally. Costs, constraints, and mis-scaled alphas erode realized IR.

## Connects To
- **Ch 5**: IR → ω* = IR/(2λ_R), VA* = IR²/(4λ_R).
- **Ch 10–11**: forecasting basics. Refined forecast α = volatility·IC·score, and combining correlated signals.
- **Ch 12**: information analysis, measuring IC empirically.
- **Ch 13**: information horizon (shorter horizons raise breadth).
- **Ch 14**: alpha scaling to consistent IR. **Ch 15**: no-short constraint cost. **Ch 16**: turnover/transaction costs eroding IR.
- **Ch 19**: why benchmark timing rarely pays.
- **Grinold (1989), "The Fundamental Law of Active Management", JPM**: original source.
