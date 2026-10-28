# Chapter 4: Exceptional Return, Benchmarks, and Value Added

## Core Idea
Split expected return into time premium, risk premium, exceptional benchmark return, and alpha. Then maximize **value added**: exceptional return minus *separate* penalties for benchmark-timing variance and residual variance. Plain total-risk utility (Markowitz) gives institutionally absurd residual risk, because clients bear benchmark risk while managers bear active risk (business risk).

## Frameworks Introduced
- **Four components of expected return**: E{R_n} = i_F + β_n·μ_B + β_n·Δf_B + α_n.
  - When to use: to sort any list of expected returns into consensus and exceptional parts before building a portfolio.
  - How (spreadsheet recipe): given E{R_n}, β_n, h_B, i_F, and μ_B: (1) f_B = Σ h_B(n)·(E{R_n} − i_F); (2) Δf_B = f_B − μ_B; (3) α_n = E{R_n} − i_F − β_n·f_B. Include cash (α = 0, β = 0) and the benchmark (α = 0, β = 1) as checks.
- **Value-added objective** (the book's standard objective): VA = [β_PA·Δf_B − λ_BT·β_PA²·σ_B²] + [α_P − λ_R·ω_P²].
  - When to use: as the objective for every active portfolio decision in the rest of the book.
  - How: choose active beta (benchmark timing) and residual positions (stock selection) *separately*. Each has its own risk aversion: λ_BT for timing and λ_R for residual.
  - Why it works: it captures the **double standard**. Sponsors accept 20% benchmark risk but are loath to accept 6.4% residual risk, because all managers share the benchmark "tide" and residual return decides who ranks last and is fired.
  - Generality: with benchmark = cash (F) and λ_T = λ_BT = λ_R, it reduces to ordinary mean/variance.
- **Utility decomposition (appendix)**: u = f_P − λ·σ_P² splits into four terms. (9) the benchmark term, all forecast and no action, a constant; (10) the cross term β_PA·(μ_B − 2λ_T·σ_B²), action without forecast, which is zero when λ_T is calibrated to hold the benchmark; (11) benchmark timing; (12) stock selection. Only 11 and 12 count as value added.
- **Separate benchmark timing from stock selection**: most of the book assumes β_P = 1, so active = residual. Reasons: most institutional managers don't time; it keeps the analysis simple; "the man who chases two rabbits will catch neither"; and timing has low breadth (Ch 6).

## Key Formulas
- Beta relative to the benchmark: β_n = Cov{r_n, r_B}/Var{r_B}.
- Active holdings: h_PA = h_P − h_B. Active cash = −(sum of active risky holdings).
- Active variance: ψ_P² = β_PA²·σ_B² + ω_P², with β_PA = β_P − 1 and ω_P² = σ_P² − β_P²·σ_B².
- Consensus expected excess return: β_n·μ_B. Expected excess return: f_n = β_n·μ_B + β_n·Δf_B + α_n.
- **Exceptional expected return** = β_n·Δf_B (benchmark timing) + α_n (stock selection).
- Total-risk utility: U = f_P − λ_T·σ_P².
- Calibrating λ_T so that you hold the benchmark under consensus: **λ_T = μ_B/(2σ_B²)**. With μ_B = 6% and σ_B = 20%, λ_T = 0.0075. With μ_B = 8% and σ_B = 16%, λ_T = 0.0156.
- Optimal beta under total utility (with cash): β_P = 1 + Δf_B/μ_B, so β_PA = Δf_B/μ_B.
- Optimal residual risk under total utility: ω* = IR/(2λ_T). For IR = 0.75 and λ_T = 0.01, ω* = 37.5%, which is far too aggressive.
- **Optimal active beta under VA**: β_PA* = Δf_B/(2·λ_BT·σ_B²). It goes to 0 if λ_BT is large or Δf_B = 0.
- Residual return θ_P = r_P − β_P·r_B. Active return r_PA = r_P − r_B = β_PA·r_B + θ_P. If β_P = 1, active = residual.
- Certainty equivalent: CE = f − λ·σ². With λ_T = 0.0075, f = 6%, and σ = 20%, CE = 6 − 3 = 3%.

## Key Concepts
- **Benchmark (bogey, normal portfolio)**: the sponsor-specified standard the manager is judged against (e.g. EAFE, Russell 2000). It replaces "the market".
- **Time premium i_F**: compensation for parting with money for a period. Known in advance.
- **Risk premium β_n·μ_B**: μ_B is a very-long-run (70+ years) estimate, typically **3–7% per year** for equity markets.
- **Exceptional benchmark return Δf_B = f_B − μ_B**: a near-term view that differs from the long-run premium. It averages zero over time.
- **Alpha α_n**: expected residual return, E{θ_n}. The benchmark's alpha is 0 by definition.
- **Active beta β_PA**: β_P − 1. The lever for benchmark timing.
- **Business risk vs investment risk**: the client bears benchmark (investment) risk. The manager's career rides on residual (active) risk.
- **Value added**: risk-adjusted exceptional return, ignoring the benchmark's own risk and return.

## Mental Models
- Think of the **benchmark as the tide that lifts all boats**. Managers can't affect it and are ranked only on residual return.
- Use **VA, not total utility**, whenever you're judged relative to a benchmark. Total-risk utility treats a little information as permission to take huge residual risk.
- Think of **λ_T as implied by holding the benchmark**: back it out from μ_B/(2σ_B²) instead of guessing.
- **No forecast means no action**: if Δf_B = 0 and α = 0, the optimum is the benchmark, with β_PA = 0 and ω = 0.

## Anti-patterns
- **Maximizing total mean/variance utility for an institutional mandate**: on MMI forecasts it gives ω ≈ 21%, about 3× typical institutional levels.
- **Treating benchmark and residual risk even-handedly**: this ignores who bears which risk.
- **Using historical alphas as forecasts**: expected returns vary wildly (from −12% to +29% across MMI stocks) because historical alphas are noisy.
- **Mixing benchmark timing into stock selection**: they are distinct decisions with distinct risk aversions. Judge timing by risk-adjusted value added, not by one lucky period ("I knew a fellow who was in puts in October 1987").
- **Endless style niches**: managers spawn new styles to avoid comparison, and benchmarks multiply.

## Reference Tables
MMI return breakdown, Dec 1992 (β = 0.96 vs S&P 500; i_F = 26 bp; S&P = 131 bp; long-run monthly excess = 50 bp, so benchmark surprise = 55 bp):

| Component | Return |
|---|---|
| Risk-free | 0.26% |
| Risk premium (0.96·0.50) | 0.48% |
| Exceptional benchmark (0.96·0.55) | 0.53% |
| Alpha | −1.87% |
| Total | −0.60% |

Total-utility optimal portfolio on MMI forecasts (λ_T = 0.0075): f_P = 27.29%, σ_P = 25.20%, β_P = 1.05, ω_P = 20.99%, α_P = 13.78%.

Selected MMI expected returns (i_F = 3.16%, μ_B = 6%, 60-month historical alphas): Coca-Cola α 20.03%, β 1.00, E{R} 29.19%; IBM α −19.04%, β 0.64, E{R} −12.04%; S&P 500 α 0, β 1, E{R} 9.16%; cash 3.16%.

| Risk aversion | Typical value |
|---|---|
| λ_T (total) | 0.0075–0.0156 (from μ_B/2σ_B²) |
| λ_R (residual), "moderate to aggressive" | 0.075 |

## Worked Example
**Total vs value-added investor** (problem 4). Benchmark f_B = 6% and σ_B = 20%. Both portfolios have β = 1.
- Portfolio 1: f = 10%, σ = 20.22%, so α = 4% and ω = √(408.85 − 400) ≈ 2.97%.
- Portfolio 2: f = 16%, σ = 25%, so α = 10% and ω = √(625 − 400) = 15%.
- Investor A (λ_T = 0.0075): U₁ = 10 − 0.0075·408.85 = 6.93 and U₂ = 16 − 0.0075·625 = 11.31, so A picks **Portfolio 2**.
- Investor B (λ_R = 0.075): VA₁ = 4 − 0.075·8.85 = 3.34 and VA₂ = 10 − 0.075·225 = −6.88, so B picks **Portfolio 1**.

**Decomposing McDonald's** (problem 1): i_F = 6%, f_B = 6.5%, μ_B = 6%, β = 1.07, E{R} = 15%. The parts are time premium 6%, risk premium 6.42%, exceptional benchmark 0.535%, and alpha = 15 − 6 − 6.42 − 0.535 ≈ 2.05%. Consensus return (12.42%) + exceptional return (2.58%) = 15%.

**Risk illusion**: a 21% total risk vs a 20% benchmark looks small, but with β = 1 it means ω = ψ = √(441 − 400) = 6.4%.

## Key Takeaways
1. Exceptional return = benchmark timing (β_n·Δf_B) + stock selection (α_n). Everything else is consensus.
2. Use the value-added objective with separate λ_BT and λ_R. Never use total-risk utility for benchmarked mandates.
3. Small differences in total risk can hide large residual risk (20% vs 21% total means 6.4% residual).
4. Set β_P = 1 unless you have a real timing forecast. Then active risk = residual risk.
5. Calibrate total risk aversion as λ_T = μ_B/(2σ_B²), which is about 0.0075 for typical equity numbers.
6. Value added is a certainty-equivalent return: expected return minus λ·variance.

## Connects To
- **Ch 2**: consensus μ = β·μ_B and the CAPM's zero consensus alpha.
- **Ch 3**: active/residual risk definitions and λ·variance as the cost of risk.
- **Ch 5**: maximizing the stock-selection term α_P − λ_R·ω_P² gives the information ratio and optimal ω* = IR/(2λ_R).
- **Ch 6**: why benchmark timing rarely adds much (low breadth).
- **Ch 18–19**: asset allocation and benchmark timing in detail.
- **Roll (1992), tracking-error mean/variance**: why tracking-error-only optimization is suboptimal in total terms.
