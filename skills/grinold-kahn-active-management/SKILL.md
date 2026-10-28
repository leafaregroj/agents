---
name: grinold-kahn-active-management
description: "Knowledge base from \"Active Portfolio Management\" (2nd ed.) by Richard C. Grinold and Ronald N. Kahn. Use when applying Grinold & Kahn's frameworks for the information ratio, the Fundamental Law (IR = IC·√BR), alpha forecasting and refinement, factor risk models, portfolio construction, transaction costs, performance attribution, or quant equity research, studying the book, or referencing its concepts."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Active Portfolio Management (2nd ed.)
**Authors**: Richard C. Grinold, Ronald N. Kahn | **Pages**: ~613 | **Chapters**: 22 + appendices | **Generated**: 2026-10-27

## How to Use This Skill

- **Without arguments**: load the core frameworks below
- **With a topic**: e.g. `breadth`, `alpha scaling`, `market impact`, `long/short`. I find and read the relevant chapter
- **With a chapter**: e.g. `ch06`. I load that chapter file
- **Browse**: ask "what chapters do you have?"

For anything not covered in Core Frameworks, read the relevant `chapters/` file before answering. For quick numeric judgments, use [cheatsheet.md](cheatsheet.md).

Note: the source PDF's equations were partly lost in extraction. Chapter files rebuilt them from context and checked them against the book's worked numbers. Formulas that could not be confirmed are marked "[reconstructed]".

---

## Core Frameworks & Mental Models

### 1. Active management is forecasting
Start from a **benchmark** and **consensus expected returns** (CAPM: β·μ_B). If your forecasts equal the consensus, or carry no information, the machinery returns you to the benchmark. Portfolios depart from the benchmark only to the extent that you have *superior information*. "Mathematics cannot overcome ignorance." (Ch 1, 2, 22)

### 2. Return decomposition
E{R} = i_F (time premium) + β·μ_B (risk premium) + β·Δf_B (exceptional benchmark return, i.e. timing) + **α** (expected residual return). Keep stock selection (α) separate from benchmark timing (β_PA). Residual return θ = r − β·r_B. (Ch 2, 4)

### 3. Value added and the information ratio
Objective: **VA = α_P − λ_R·ω_P²**, the risk-adjusted residual return. The **information ratio IR = α/ω** is both the *figure of merit* and the *budget constraint*. The residual frontier is the line α = IR·ω, so:
- Optimal residual risk **ω\* = IR/(2λ_R)**, and optimal value added **VA\* = IR²/(4λ_R)**. Value scales with IR².
- Benchmarks: IR 0.5 = good (top quartile), 0.75 = very good, 1.0 = exceptional. λ_R 0.05/0.10/0.15 = aggressive/moderate/restrained.
- Use **residual risk, not total risk**. Total-risk utility implies absurd residual risk. (Ch 4, 5)

### 4. The Fundamental Law of Active Management
**IR = IC · √BR.** IC is skill: the correlation between forecasts and outcomes. BR is breadth: *independent* bets per year.
- Play often and play well: a small edge × large breadth beats a big edge × few bets.
- To double IR you need 2× IC or **4× breadth**. A quarterly market timer needs IC 0.25 for IR 0.5; a 100-stock quarterly picker needs 0.025.
- Squared IRs add across independent sources and managers: IR² = Σ IC_i²·BR_i.
- Breadth is easy to overstate. Correlated signals and rebalancing on stale views are *not* extra bets.
- It assumes **competence**: information is used optimally. (Ch 6)

### 5. Alpha = Volatility · IC · Score
The forecasting rule of thumb: refined forecast **α = ω · IC · z**, where z is the standardized signal. Alphas must control for **volatility, skill and expectations**. This gives roughly equal risk per name. IC realism: 0.05 good, 0.10 great, 0.15 world-class, >0.20 is suspicious. Combine signals with their correlations: IC²_comb = (IC₁² + IC₂² − 2ρIC₁IC₂)/(1−ρ²). Forecasts don't change risk materially (ω_post = ω·√(1−IC²)). (Ch 10, 11)

### 6. Structural risk models
V = X·F·Xᵀ + Δ (factor exposures, factor covariance, specific variance). Use standard deviation; alternatives add effort, not value. Attribute risk via marginal contributions. Check that your highest-alpha names carry the most marginal risk. Classify bets as inherent, intentional or incidental, and remove the incidental ones. (Ch 3)

### 7. Information analysis and data-mining discipline
Test a signal by building **information-efficient portfolios** (long/short, β = 0, factor-neutral) and measuring IR, IC, t-stats and decay. IR ≈ t/√T. Guard against data mining with **intuition, restraint, sensibleness and out-of-sample testing**. Twenty worthless tests give a 64% chance of a spurious t > 2. Signals have **half-lives**: blend fresh and lagged scores when their autocorrelation ρ < IR decay γ. (Ch 12, 13)

### 8. Implementation should subtract as little value as possible
- **Alpha analysis first**: benchmark-neutralize, neutralize unwanted factors, scale to target IR, trim beyond 3σ. Every constraint is equivalent to a modified alpha.
- **Mean-variance QP** beats screens and stratification at capturing IR.
- **No-trade band**: hold while −sell cost ≤ MCVA ≤ buy cost.
- **Turnover**: trade until the slope of the VA/turnover frontier equals cost. Halving turnover keeps ≥75% of VA.
- **Market impact ∝ √(trade/ADV)**. One day's volume costs about one day's volatility.
- **Long-only** wastes negative information and keeps only ~50% of the unconstrained IR. Long/short helps most with many names, low volatility and high active risk. (Ch 14, 15, 16)

### 9. Distinguishing skill from luck is difficult
SE(IR) ≈ 1/√years, so confirming IR 0.5 at t = 2 takes ~16 years. An IR-0.5 manager beats the benchmark only ~69% of single years. Winners repeat only ~55% of the time, and survivorship bias manufactures persistence. Breadth proxies (holdings, turnover) predict future IR better than past IR does. (Ch 17, 20)

### 10. Valuation and APT as alpha sources
APT: f = X·m. The "easy part" is qualifying the model; the "hard part" is forecasting factor returns. A DDM alpha equals yield + g − required return. **Shrink growth forecasts** toward implied-growth dispersion, applying the humility principle. Misvaluation κ with persistence δ yields α ≈ (1+i_F)·κ(1−δ)/(1+δκ). (Ch 7, 8, 9)

### 11. Treynor's consensus test
If everyone you ask agrees with your idea, it is already in the price. Superior information is lonely. (Ch 22)

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introduction.md) | Introduction | Active management process; three attitudes; benchmark-relative framework |
| [ch02](chapters/ch02-capm-consensus-returns.md) | Consensus Expected Returns: The CAPM | Separation r = βr_M + θ; characteristic portfolios; reverse optimization |
| [ch03](chapters/ch03-risk.md) | Risk | Structural factor model; marginal contribution; inherent/intentional/incidental bets |
| [ch04](chapters/ch04-exceptional-return-value-added.md) | Exceptional Return, Benchmarks, and Value Added | Four-component return; value-added objective; timing vs selection |
| [ch05](chapters/ch05-information-ratio.md) | Residual Risk and Return: The Information Ratio | Residual frontier; ω* and VA*; IR benchmarks; alpha scaling |
| [ch06](chapters/ch06-fundamental-law.md) | The Fundamental Law of Active Management | IR = IC·√BR; additivity of IR²; dependence audit |
| [ch07](chapters/ch07-apt-expected-returns.md) | Expected Returns and the APT | APT f = Xm; qualified models; factor forecasting |
| [ch08](chapters/ch08-valuation-in-theory.md) | Valuation in Theory | Risk-adjusted expectations; value multiples; misvaluation → alpha |
| [ch09](chapters/ch09-valuation-in-practice.md) | Valuation in Practice | DDM; implied growth; growth shrinkage; comparative valuation |
| [ch10](chapters/ch10-forecasting-basics.md) | Forecasting Basics | α = vol·IC·score; basic forecasting formula; signal combination |
| [ch11](chapters/ch11-advanced-forecasting.md) | Advanced Forecasting | Time-series vs cross-sectional scores; spillover; IC shrinkage |
| [ch12](chapters/ch12-information-analysis.md) | Information Analysis | Information-efficient portfolios; event studies; data-mining guards |
| [ch13](chapters/ch13-information-horizon.md) | The Information Horizon | Signal half-life; Now/Later blending; lagged scores |
| [ch14](chapters/ch14-portfolio-construction.md) | Portfolio Construction | Alpha analysis; constraints ⇔ alphas; QP vs screens; dispersion |
| [ch15](chapters/ch15-long-short-investing.md) | Long/Short Investing | Long-only cost; shrinkage factor; capitalization model; Gulliver |
| [ch16](chapters/ch16-transactions-costs-turnover-trading.md) | Transactions Costs, Turnover, and Trading | Implementation shortfall; √ impact model; VA/turnover frontier |
| [ch17](chapters/ch17-performance-analysis.md) | Performance Analysis | Skill vs luck; returns- vs portfolio-based attribution; t-stats |
| [ch18](chapters/ch18-asset-allocation.md) | Asset Allocation | TAA process; out-of-sample IC testing; currency consensus |
| [ch19](chapters/ch19-benchmark-timing.md) | Benchmark Timing | Optimal active beta; futures vs stocks; low breadth |
| [ch20](chapters/ch20-historical-record.md) | The Historical Record for Active Management | Survivorship; persistence; breadth predictors |
| [ch21](chapters/ch21-open-questions.md) | Open Questions | Dynamics; liquidity; taxes; behavioral finance |
| [ch22](chapters/ch22-summary.md) | Summary | Four themes; Treynor's test |
| [ch23](chapters/ch23-appendix-notation-and-statistics.md) | Appendix A & C: Notation, Statistics | Notation table; standard errors; regression basics |

## Topic Index

- **Active beta / benchmark timing** → ch04, ch19
- **Alpha scaling / trimming / neutralization** → ch05, ch14
- **APT / factor forecasts** → ch07, ch11
- **Asset allocation / TAA / currency** → ch18
- **Breadth** → ch06, ch12, ch19, ch20
- **CAPM / consensus returns / characteristic portfolios** → ch02
- **Data mining** → ch12, ch10
- **Dispersion across accounts** → ch14
- **Dividend discount model / implied growth** → ch09
- **Event studies** → ch12
- **Fundamental Law (IR = IC·√BR)** → ch06, ch01, ch22
- **Information coefficient (IC)** → ch06, ch10, ch12
- **Information horizon / decay / half-life** → ch13, ch12
- **Information ratio** → ch05, ch06, ch17
- **Long/short vs long-only** → ch15
- **Market impact / trading** → ch16
- **Notation** → ch23
- **Performance attribution / skill vs luck** → ch17, ch20
- **Portfolio construction / optimization** → ch14
- **Residual risk aversion / value added** → ch04, ch05
- **Risk models / tracking error** → ch03
- **Signal combination** → ch10, ch11
- **Survivorship bias / persistence** → ch17, ch20
- **Transaction costs / turnover** → ch14, ch16
- **Valuation theory / value multiples** → ch08

## Supporting Files

- [glossary.md](glossary.md): all key terms with definitions
- [patterns.md](patterns.md): techniques (alpha refinement, information analysis, QP, trading, attribution)
- [cheatsheet.md](cheatsheet.md): thresholds, decision rules, tells and smells

---

## Scope & Limits

This skill covers the book's content only (2nd ed., 2000; empirical numbers are from the 1990s). It is not personalized investment advice. Chapter formulas were reconstructed where the PDF extraction lost them; check flagged formulas against the book before relying on them in production.
