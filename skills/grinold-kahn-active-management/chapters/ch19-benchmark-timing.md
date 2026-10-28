# Chapter 19: Benchmark Timing

## Core Idea
Benchmark timing—varying active beta on forecasts of *exceptional* benchmark return—is a one-asset bet, so by the Fundamental Law it adds almost nothing unless you make many independent forecasts per year; implement it with futures (not high/low-beta stocks), and size it with `active beta ∝ IC · score`.

## Frameworks Introduced
- **Benchmark timing defined**: an active decision to vary portfolio beta relative to the benchmark—raise β when you expect the benchmark to do *better than usual*, cut it when *worse than usual*. In its pure form, choosing the mix of benchmark and cash (equivalent to trading benchmark futures/forwards), with no residual bets. Not asset allocation (which has 5–20 assets; benchmark timing has 1).
- **Fundamental Law applied to timing**: `IR_BT = IC_BT · √BR`. Stock selection diversifies cross-sectionally; timing can diversify only **serially** (more forecasts per year).
  - Quarterly forecasts (BR = 4): IR 0.5 needs IC 0.25—"extremely high."
- **Futures vs. stocks for implementation**: tilting toward high-/low-beta stocks has three drawbacks: (1) unavoidable residual risk, (2) beta estimation error (cash β = 0 and benchmark β = 1 exactly), (3) higher transactions costs. Without a futures contract on the benchmark, timing potential is very small.
- **Exceptional-return formulation** (reformulation to avoid forecasting μ_B and choosing λ_BT):
  1. Forecast exceptional benchmark return directly: `Δf_B = σ_B · IC · S`.
  2. Translate qualitative views into scores with mean 0, std 1 (Table 19.3).
  3. Set aggressiveness via κ (the active-beta-per-unit-score): κ = 0.06 ⇒ β within 0.94–1.06 two-thirds of the time, above 1.06 one time in six, below 0.94 one time in six. Too aggressive? Lower κ.
  - Realistic IC: 0.05 good, 0.10 very good, 0.15 world class/terrific. "Humility should enter": IC 0.10 ≈ calling better-than-average benchmark direction correctly 55% of the time.
- **Forecast frequency**: with T independent forecasts per year, value added scales with T. Repackaging one annual forecast into four quarterly ones adds no information—breadth requires new information each period (and knowledge of which signal components apply to which period).
- **Timing performance analysis**: returns-based (up- vs. down-market betas, Ch 17) or portfolio-based (requires ex ante betas): split active systematic return into expected active beta return, active beta surprise, and active benchmark timing, using μ_B and average active beta set ex ante (ideally agreed with the client), ex post, or hybrid.

## Key Formulas
- **Fundamental Law** (19.1): `IR_BT = IC_BT · √BR`.
- **Minimum residual risk for active beta via stocks** (19.2–19.3, 19A.1–19A.2): minimum-risk fully invested portfolio with β = β_BT is a mix of benchmark and portfolio C; its residual variance
  `ω²_min = β_PA² · σ_B² · β_C / (1 − β_C)`, with `β_C = σ_C² / σ_B²` (≡ β_PA²·σ_B²σ_C²/(σ_B² − σ_C²)).
- **Value added of timing, one decision/year** (19.4): `VA_BT = β_PA · Δf_B − λ_BT · β_PA² · σ_B²`.
- **Optimal active beta** (19.5): `β*_PA = Δf_B / (2 λ_BT σ_B²)`.
- **Optimal value added** (19.6): `VA* = Δf_B² / (4 λ_BT σ_B²)`.
- **Refined exceptional forecast** (19.7): `Δf_B = σ_B · IC · S`.
- **Score form** (19.8–19.9): `β*_PA = κ · S`, `κ = IC / (2 λ_BT σ_B)`; `VA(S) = IC² S² / (4 λ_BT) = κ · σ_B · IC · S² / 2`.
- **Unconditional VA** (19.10): `E{VA} = IC² / (4 λ_BT) = κ σ_B IC / 2`.
- **T forecasts per year** (19.11–19.17): period volatility `σ_B/√T`; `Δf_B(t) = (σ_B/√T) · IC · S(t)`; `β*_PA(t) = κ √T · S(t)` (betas double going annual→quarterly, all else equal, though IC may shrink at shorter horizons); `E{VA_annual} = T · IC² / (4 λ_BT)` — VA proportional to breadth.
- **Net VA when timing with stocks** (19A.13, form reconstructed): `E{VA} ≈ T·IC²/(4λ_BT) · [1 − (λ_R/λ_BT) · β_C/(1 − β_C)]` → positive only if residual risk aversion is well below timing risk aversion (β_C = 4/9 for σ_B = 18%, σ_C = 12%).
- **Ex ante timing attribution** (19.18): `β_PA(t) · θ_B(t) = β_PA(t) · μ_B Δt + β_PA(t) · [θ_B(t) − μ_B Δt]` = expected active return + benchmark timing.
- **Ex post attribution**: with sample averages β̄, θ̄: `β_PA θ_B = β̄ θ̄ + β̄ (θ_B − θ̄) + (β_PA − β̄) θ_B`; the timing term averages to the in-sample covariance of active beta and benchmark return.

## Key Concepts
- **Active beta (β_PA)**: β_P − 1; the decision variable.
- **Exceptional benchmark return (Δf_B)**: refined forecast f_B minus usual expected excess return μ_B.
- **Aversion to benchmark timing risk (λ_BT)**: high 0.14, medium 0.09, low 0.06.
- **Score (S)**: normalized signal, mean 0, std 1 over time.
- **κ**: active beta per unit score; a direct aggressiveness dial replacing λ_BT.
- **Portfolio C**: minimum-variance fully invested portfolio; β_C = σ_C²/σ_B².
- **Serial diversification**: breadth from repeated independent forecasts over time.

## Mental Models
- Think of timing as stock selection with one stock: breadth only comes from the calendar.
- Consider the gambler analogy: 65% on the Super Bowl once a year loses to 55% on each of 480 games.
- Use κ to sanity-check aggressiveness by asking how often beta should leave a ±κ band.
- Use futures whenever they exist; with only stocks, the residual-risk penalty can wipe out most of the gain.

## Anti-patterns
- **Annual timing bets expecting meaningful value**: IC 0.10 once a year ≈ 4 bp.
- **Implementing timing through stock tilts**: an active β of 0.1 forces ≥1.6% residual risk (≈0.25% VA penalty at λ_R = 0.1); shaves 2.9 of 4.2 bp in the base case.
- **Overestimating timing skill**: it's rare; assuming you have it is probably self-deception.
- **Counting split-up annual forecasts as multiple bets**: no new information, no added breadth.
- **Forecasting absolute benchmark returns**: work in exceptional (relative-to-usual) terms.

## Reference Tables
**Table 19.1 — optimal active beta (σ_B = 17%)**
| Δf_B | High λ 0.14 | Medium 0.09 | Low 0.06 |
|---|---|---|---|
| +4% | 0.05 | 0.08 | 0.12 |
| +2% | 0.02 | 0.04 | 0.06 |
| 0% | 0.00 | 0.00 | 0.00 |
| −2% | −0.02 | −0.04 | −0.06 |
| −4% | −0.05 | −0.08 | −0.12 |

**Table 19.2 — value added, bp/yr (one decision/yr)**
| Δf_B | High 0.14 | Medium 0.09 | Low 0.06 |
|---|---|---|---|
| ±4% | 9.9 | 15.4 | 23.1 |
| ±2% | 2.5 | 3.8 | 5.8 |
| 0% | 0 | 0 | 0 |

**Table 19.3 — view → score**: very positive (p 0.11) 1.73; positive (0.22) 0.87; no view (0.33) 0; negative (0.22) −0.87; very negative (0.11) −1.73.

**Table 19.4 — IC 0.10, σ_B 17%, λ_BT 0.06**
| Score | Forecast Δf_B | Active β | VA |
|---|---|---|---|
| ±1.73 | ±2.94% | ±0.09 | 0.12% |
| ±0.87 | ±1.47% | ±0.04 | 0.03% |
| 0 | 0 | 0 | 0 |

**Table 19.5 — κ by skill and aversion**
| Skill (IC) | High 0.14 | Medium 0.09 | Low 0.06 |
|---|---|---|---|
| Good 0.05 | 0.01 | 0.02 | 0.02 |
| Very good 0.10 | 0.02 | 0.03 | 0.05 |
| World class 0.15 | 0.03 | 0.05 | 0.07 |

**Table 19.6 — expected annual VA (bp), λ_BT = 0.09**
| IC | 1/yr | 4/yr | 12/yr | 52/yr |
|---|---|---|---|---|
| 0.10 | 2.78 | 11.11 | 33.33 | 144.44 |
| 0.05 | 0.69 | 2.78 | 8.33 | 36.11 |
| 0.02 | 0.11 | 0.44 | 1.33 | 5.78 |
| 0.01 | 0.03 | 0.11 | 0.33 | 1.44 |

## Worked Example
- **Stocks vs. futures**: σ_B = 18%, σ_C = 12% ⇒ β_C = 144/324 = 4/9. Active β 0.1 ⇒ ω² = 0.01 × 324 × (4/9)/(5/9) = 2.59 ⇒ ω ≈ 1.6%; at λ_R = 0.1 the penalty ≈ 0.26%.
- **Table 19.2 entry**: Δf_B = 4%, λ_BT = 0.06, σ_B = 17%: β* = 4/(2 × 0.06 × 289) = 0.115; VA* = 16/(4 × 0.06 × 289) = 0.23% = 23.1 bp.
- **Score-based**: IC 0.10, σ_B 17%: very positive view S = 1.73 ⇒ Δf_B = 17 × 0.10 × 1.73 = 2.94%; κ = 0.10/(2 × 0.06 × 17) ≈ 0.05 ⇒ β_PA ≈ 0.09; E{VA} = 0.01/(4 × 0.06) ≈ 0.042% = 4.2 bp/yr (with κ 0.05).
- **Problems**: λ_BT 0.09, Δf_B 5%, σ_B 17% ⇒ β* = 5/52.0 ≈ 0.096 ⇒ portfolio β ≈ 1.10. Bob (IC 0.05, once/yr, λ 0.06): VA = 0.0025/0.24 ≈ 1 bp/yr; active risk = κσ_B ≈ 0.025 × 17 ≈ 0.42%; at 12 bets/yr VA ≈ 12.5 bp.
- **Binary model**: annual exceptional return = sum of 400 independent ±1% elements (σ = 20%/yr, 10%/quarter); a single annual signal yields the same VA whether viewed annually or quarterly; receiving the same information in quarterly parcels quadruples breadth.

## Key Takeaways
1. Benchmark timing is hard and low value: one IC-0.10 bet per year ≈ 3–4 bp.
2. Value added scales with the number of *independent* forecasts per year: 52 weekly IC-0.10 calls ≈ 144 bp.
3. Forecast exceptional (not absolute) benchmark returns: `Δf_B = σ_B·IC·S`; active beta = κ·S.
4. Use futures; stock-based timing incurs unavoidable residual risk, beta error, and high costs.
5. Reasonable ICs: 0.05/0.10/0.15; be humble.
6. Measure timing via up/down betas or by attributing active systematic return into expected, surprise, and timing parts—but low-frequency timing needs many years to be statistically provable.

## Connects To
- **Ch 4**: original split of active management into benchmark timing and stock selection; timing utility.
- **Ch 6**: Fundamental Law—breadth drives the conclusions.
- **Ch 10**: forecasting rule of thumb, scores, IC.
- **Ch 17**: Henriksson-Merton timing regression and attribution of active systematic return.
- **Ch 18**: asset allocation as the multi-asset cousin.
- **Ch 2 / Ch 3**: portfolio C and beta.
- **External**: Henriksson-Merton (1981); Treynor-Mazuy; Sharpe (1975) "Likely Gains from Market Timing".
