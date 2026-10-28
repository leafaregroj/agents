# Chapter 17: Performance Analysis

## Core Idea
Performance analysis exists to separate skill from luck—and to show *where* the skill lies; cross-sectional rankings cannot do it, returns-based regressions do it crudely, and holdings-based attribution followed by time-series analysis of each attributed component does it best, provided the manager's goals were agreed ex ante.

## Frameworks Introduced
- **Skill vs. luck matrix** (Fig. 17.1): *blessed* (skill + luck), *doomed* (neither), *forlorn* (skill, no luck), *insufferable* (luck, no skill). We only observe the combination; blessed and insufferable both show positive histories.
  - Why it matters: of 1,000 managers, ~50 (5%) look exceptional by chance alone.
- **Three tiers of performance analysis**:
  1. *Cross-sectional comparison* (peer-universe percentile tables): identifies winners/losers only.
  2. *Returns-based analysis* (time series of returns): Jensen alpha regression, Sharpe ratio comparison, plus refinements—separates skill from luck in aggregate.
  3. *Portfolio-based analysis* (time series of holdings + returns): attributes return to many sources and tests each for skill.
- **Jensen returns regression**: regress portfolio excess returns on benchmark excess returns; intercept = realized alpha, slope = beta; test alpha's t statistic (t ≥ 2 ≈ skill at 95%).
- **Returns-based refinements**: statistical (Bayesian shrinkage of α and β using cross-manager priors—Vasicek; heteroskedasticity; autocorrelation, e.g. overlapping periods) and financial (benchmark timing term, a priori/forecast betas, valuation-based value added, conditioning on public information—Ferson-Schadt, Sharpe style analysis, Fama-French size/value controls, Carhart momentum). The last three are controversial: they credit the manager only for returns beyond what public information/styles deliver.
- **Style analysis** (Sharpe 1992): estimate nonnegative, fully invested style weights by minimizing *variance* of selection return.
  - When to use: large-sample studies of many managers without holdings data; rough risk estimates.
  - Failure mode: style weights have large standard errors (style indices are highly correlated); identifying style needs no fancy machinery anyway.
- **Portfolio-based performance analysis** = *performance attribution* (single period, factor model on beginning-of-period exposures) + *performance analysis* (time series of each attributed component: mean, t statistic, IR).
  - How:
    1. Each period, attribute active return: active systematic (β_PA·r_B) vs. active residual; residual split into common factors (industries, risk indices) and specific asset selection.
    2. Split active systematic return into expected active beta return (not active management), active beta surprise, and active benchmark timing.
    3. For each series compute mean, risk (blend of ex ante forecast and realized), IR and t stat.
    4. Rank **best and worst policies**; check that they match the manager's stated strategy (stock pickers → specific selection; value managers → value factors).
  - Why it works: it tests whether value added arose along the dimensions promised ex ante; diagnoses *incidental risk* (e.g., growth tilt bringing industry/volatility bets) and *incremental decision making* (a chain of individually sensible trades that ends up far riskier than intended).
- **Flexible attribution**: factors may include ex post factors (e.g., end-of-period IBES revisions → skill at forecasting revisions); specific return can be grouped by analyst or sector; part of return can be attributed to optimizer constraints via Lagrange multipliers (Grinold & Easton). Attribution is not unique.

## Key Formulas
- **Standard error of IR** (17.1): `SE(IR) ≈ 1/√Y` (Y = years) → `t ≈ IR · √Y`. More exact (accounting for error in estimated residual risk): `SE(IR) ≈ √[(1 + IR²·Δt/2) / Y]`, Δt = period length (1/12 for monthly) (reconstructed from Problem 3's setup).
  - Top quartile (IR = 0.5) at t = 2 requires **Y = 16 years**. Using monthly IRs doesn't help.
- **Probability of positive realized alpha over horizon H years**: `Φ(IR · √H)`. IR = 0.5: 1 month 56%, 1 year 69%, 5 years 87% (so 13% of skilled managers show negative 5-year alpha).
- **Return definitions**: compound `R_P(1,T) = Π(1 + R_P(t)) − 1`; geometric `g_P = [Π(1 + R_P(t))]^{1/T} − 1`; average log `z_P = (1/T)Σ ln(1 + R_P(t))`; arithmetic `a_P = (1/T)Σ R_P(t)`. Always `z_P ≤ g_P ≤ a_P`; `g ≈ a − σ²/2`. Be consistent.
- **Jensen regression** (17.7): `r_P(t) = α_P + β_P · r_B(t) + ε_P(t)`; residual `θ_P(t) = α_P + ε_P(t)`.
- **Alpha t statistic**: `t_α ≈ α_P · √T / ω_P` (per-period α, ω; T observations).
- **Sharpe test**: positive performance if `SR_P > SR_B`; SE of each ≈ `1/√N`, of the difference ≈ `√(2/N)` → significant if `SR_P − SR_B > 2√(2/N)` (per-period ratios). Dybvig-Ross: Sharpe-superior ⇒ positive Jensen α, but not conversely.
- **Benchmark timing regression** (17.11): `r_P(t) = α_P + β_P r_B(t) + γ_P · max{0, r_B(t)} + ε_P(t)`; down-market beta β_P, up-market β_P + γ_P; γ_P > 0 significant ⇒ timing skill (Henriksson-Merton).
- **Conditional (Ferson-Schadt) regression** (17.12, form reconstructed): `r_P = α + β₀ r_B + β₁ [y(t)·r_B(t)] + β₂ [i_F(t)·r_B(t)] + ε` — beta varies linearly with dividend yield y and short rate i_F.
- **Style analysis** (17.13–17.14): `r_P(t) = Σ_j h_Pj r_j(t) + u_P(t)`; choose h to minimize `Var{u_P}` s.t. `h_Pj ≥ 0, Σ h_Pj = 1` (penalizes variance, not mean, of selection return).
- **Fama-French** (17.15): `r_P(t) = α_P + β_P r_B(t) + s_P · SMB(t) + h_P · HML(t) + ε_P(t)`. SMB/HML from 2×3 sorts (NYSE median size; book-to-price 30/40/30).
- **Attribution** (17.16–17.17): `r_P(t) = Σ_j x_Pj(t) b_j(t) + u_P(t)`; return attributed to factor j = `x_Pj(t) · b_j(t)`.
- **Active residual split** (17.21–17.23): `x_PR = x_PA − β_PA x_B`, `h_PR = h_PA − β_PA h_B` → `r_PA = β_PA r_B + Σ_j x_PR,j b_j + u_PR`.
- **Active systematic decomposition** (17.24–17.27): `β_PA(t) r_B(t) = β̄_PA μ_B + β̄_PA [r_B(t) − μ_B] + [β_PA(t) − β̄_PA] r_B(t)` (β̄_PA = average active beta, μ_B = long-run expected benchmark excess return) = expected active beta return + active beta surprise + active benchmark timing.
- **Cumulative active return**: `R_P(1,T) − R_B(1,T)` (difference of compounded totals), *not* the compounding of period active returns. For attributed returns use the top-down approach (delete each factor from the cumulative total and observe the effect), which leaves a smaller cross-product term; allocate that back proportional to factor risk or return.
- **Blended risk estimate** (17A.12): with SE(variance) ≈ σ²√(2/T), inverse-variance weighting gives `σ̂² = (T₀ σ²_prior + T σ²_real) / (T₀ + T)`, T₀ = effective observations behind the prior.
- **Valuation-based value added**: choose positive valuation multiples (exponential in benchmark return, per continuous-time option theory) so observed benchmark and risk-free return sets each price at $1.00; the portfolio's return set prices at 1 + VA (e.g., 1.027 → VA = 2.7%).

## Key Concepts
- **Survivorship bias**: excluding defunct funds inflates averages and fakes persistence; worsens with horizon.
- **One man, one vote**: peer tables weight a $150M and a $15B portfolio equally.
- **Realized alpha vs. average active return**: differ when active beta ≠ 0 (MMI vs S&P: α = +3 bp/mo, active return −4 bp/mo, active β = −0.08).
- **t statistic vs. information ratio**: t measures statistical significance (skill); IR measures realized value added whether significant or not.
- **Selection return**: return unexplained by style exposures.
- **Specific asset selection**: residual return beyond all attribution factors.
- **Best/worst policies**: attributed series with highest/lowest average return.
- **Incidental risk / incremental decision making**: the two pitfalls ex post analysis catches.

## Mental Models
- Use t statistics to judge skill, IRs to judge value; demand both before concluding.
- Think of returns-based analysis as top-down (credit only for the unexplained remainder) and portfolio-based as bottom-up (credit along many dimensions).
- Use a priori (forecast) betas to avoid spurious realized-beta effects: MMI realized β 0.92 vs forecast 0.98–1.03 flips α from +3 to −4 bp/month.
- If best/worst policies simply mirror the largest active exposures, there's no evidence of skill—just bets.

## Anti-patterns
- **Ranking by cumulative return**: ignores risk; Manager B (+20%) looks better than A (+16%) until you see A's smoother path (IR 1.0 vs 0.7).
- **Treating positive returns as proof**: roulette players win on half of spins.
- **Expecting a short track record to prove skill**: 16 years for top quartile.
- **Over-reading style weights**: large standard errors from correlated style indices.
- **Compounding period active returns to get cumulative active return**.
- **Performance analysis without an ex ante agreement** on goals and methods—no way to test promised dimensions.

## Reference Tables
**Table 17.1 — PSN institutional equity, annualized 1988–1992**
| 5th | 25th | Median | 75th | 95th | S&P 500 |
|---|---|---|---|---|---|
| 23.57% | 18.97% | 16.31% | 14.50% | 10.92% | 15.80% |

**Table 17.3 — MMI vs S&P 500 attribution, January 1988**: active systematic +0.06%; active residual −4.88% (common factor −0.75%, specific −4.13%); active total −4.82%.

**Table 17.4 — MMI vs S&P 500, annualized 1988–1992** (rows reconstructed from garbled extraction; additive checks hold)
| Component | Return | Risk | IR | t |
|---|---|---|---|---|
| Active beta surprise | 0.02% | 0.16% | 0.23 | 0.51 |
| Active benchmark timing | 0.03% | 0.19% | 0.13 | 0.28 |
| Total systematic | 0.06% | 0.25% | 0.24 | 0.54 |
| Industry factors | 0.27% | 1.88% | 0.12 | 0.26 |
| Risk index factors | −0.97% | 2.25% | −0.36 | −0.81 |
| Specific | 0.12% | 3.23% | 0.01 | 0.02 |
| Total residual | −0.58% | 4.21% | −0.14 | −0.30 |
| Total active | −0.52% | 4.22% | −0.12 | −0.27 |
No |t| near 2 → no demonstrated skill.

**Table 17.5 — best/worst policies**: best: foreign income 0.44%, international oil 0.36%, drugs/medicine 0.33%, tobacco 0.22%, health care (nondrug) 0.18%; worst: size −1.24%, photographic/optical −0.48%, business machines −0.40%, paper −0.37%, telephone −0.32%.

## Worked Example
- **MMI vs S&P 500 (Jan 1988–Dec 1992, monthly)**: α = 0.03%/mo, β = 0.92, t_α = 0.36 → not significant. Timing regression: β = 0.95, γ = −0.05, t_γ = −0.41 → no timing skill.
- **Track-record arithmetic**: IR 0.5 → t = 0.5√Y = 2 ⇒ Y = 16. IR 1.0 over 1 year → t = 1 (≈68% confidence); need 4 years for t = 2. Problems 1–2: Joe (IR 1, t 1, 1 year) has high value but unproven skill; Jane (IR 0.4, t 2, 25 years) has proven but modest skill.
- **Blended variance (appendix exercise)**: forecast (17%)² = 289 with SE (5.1%)² = 26.0 ⇒ √(2/T₀) = 0.09 ⇒ T₀ ≈ 247; realized (12%)² = 144 over T = 60 ⇒ σ̂² = (247·289 + 60·144)/307 ≈ 261 ⇒ σ̂ ≈ 16.1%.

## Key Takeaways
1. Goal: separate skill from luck, then locate the skill. Cross-sections can't; time series can; holdings-based attribution can say where.
2. Expect noise: proving top-quartile skill takes ~16 years; 13% of IR-0.5 managers lose over 5 years.
3. Use realized alpha t-stats (≥2) for significance and IRs for value; account for beta via forecast betas.
4. Decompose active systematic return into expected beta return, beta surprise, and timing; only the last two are active management.
5. Compare best/worst policies to the manager's stated strategy—mismatch signals style drift or a broken process.
6. Agree ex ante on goals, benchmark, expected benchmark return and target beta.
7. Cumulate attributed returns carefully (top-down) and use blended forecast/realized risk.

## Connects To
- **Ch 12**: t statistics, IC, IR relationships and information analysis.
- **Ch 4 / Ch 19**: benchmark timing value added and its performance measurement.
- **Ch 3**: multiple-factor models underlying attribution.
- **Ch 8**: valuation multiples behind the value-added approach.
- **Ch 20**: empirical record on average performance and persistence.
- **External**: Jensen (1968), Sharpe ratio, Henriksson-Merton timing test, Fama-French/Carhart factor models, Brinson attribution.
