# Chapter 20: The Historical Record for Active Management

## Core Idea
The empirical record shows the average active manager underperforms (especially after survivorship correction) and performance persistence is weak (winners repeat ~50–60% of the time), yet this neither disproves skilled active management nor identifies skilled managers from track records; what *does* predict performance is breadth (more holdings, more turnover), as the Fundamental Law implies.

## Frameworks Introduced
- **The three empirical questions** (start with Jensen 1968, Sharpe 1966, Treynor 1965):
  1. Does the average active manager outperform? (A "no" does not prove active management impossible.)
  2. Are top performers skillful or lucky?
  3. Does performance persist?
- **Successful active management = superior information + efficient implementation.** The book's machinery handles implementation; the historical record tests whether anyone has both.
- **Simple model of the manager population (persistent winners vs. coin tossers)**: a zero-sum population with a fraction δ of persistent winners (always right) and coin tossers right slightly less than half the time so the overall hit rate is 50%. A 2×2 contingency table (win/lose in period 1 × period 2) yields the repeat-winner probability p.
  - When to use: translating a published "winners repeat X%" into an implied fraction of skilled managers.
  - How: `δ = 1 − 1/(2p)`.
  - Failure mode: real skill is noisy (even good managers lose some periods), so the model *underestimates* the skilled fraction; and noise means history never precisely identifies skill. Filtering via more consecutive periods helps in principle (fewer lucky coin tossers survive 3–4 straight wins) but careers and skills don't last that long.
- **Predictors of performance (Fundamental Law test)**: proxies for breadth—number of holdings and turnover—should predict higher IRs (turnover partly offset by costs).
  - Evidence: Stewart (1998), 1,527 US institutional equity products 1978–1996, sorted by *consistency* (fraction of months with positive active return, monotone in IR): more consistent quintiles hold more names and turn over more, and consistent performers had higher next-period active returns. Kahn & Rudd (1997c), 367 institutional portfolios 1993–1996, style-controlled: prior IR not significant, but **number of assets** significantly predicts future IR.
- **Why believe in active management looking forward**: (1) behavioral finance—humans are irrational in consistent, specific ways, leaving persistent inefficiencies; (2) Ambachtsheer's argument—only active management makes markets efficient; at low active share even crude research pays, at high active share only efficient implementation wins, at the expense of inefficient processes and inferior information.

## Key Formulas
- **Fraction of skilled managers from repeat-win probability** (20.1, derived from the model):
  coin-tosser hit rate `q = (½ − δ)/(1 − δ)`; repeat-winner probability `p = 2[δ + (1 − δ)q²] = 1/[2(1 − δ)]` ⇒ `δ = 1 − 1/(2p)`.
  - p = 50% → δ = 0; p = 60% → δ ≈ 17%; p = 100% → δ = 50% (the maximum in a zero-sum game).
- **Probability a manager with information ratio IR outperforms over H years** (normal returns): `Φ(IR · √H)`; two consecutive periods: `Φ(IR·√H)²`.
- **Consistency vs. IR** (Stewart footnote, reconstructed): fraction of positive monthly active returns `= Φ(IR/√12)`.

## Key Concepts
- **Survivorship bias**: databases dropping dead funds overstate average performance and *manufacture* apparent persistence (Brown, Goetzmann, Ibbotson & Ross 1992).
- **Persistence of performance**: winners in one period remaining winners in the next.
- **Persistent winners / coin tossers**: the two populations in the contingency-table model.
- **Consistency of performance**: fraction of periods with positive active return (Stewart).
- **Characteristic-based benchmarks**: Daniel-Grinblatt-Titman-Wermers 125 size/book-to-price/momentum quintile groupings.
- **Behavioral finance**: study of systematic, predictable investor irrationality.

## Mental Models
- Read a 55–60% repeat rate as "at most ~1 in 6 managers is persistently skilled"—not as proof that skill is absent.
- Think of manager selection by track record alone as nearly a coin flip; look at process and breadth instead.
- Use the Fundamental Law as a screening lens: favor managers with many independent bets.
- Think of market efficiency as an equilibrium maintained by active managers—and paid for by the inefficient ones.

## Anti-patterns
- **Picking last year's winners**: barely better than 50% to beat the median again.
- **Concluding "active management is impossible" from average underperformance**: why would the average manager outperform in a zero-sum game?
- **Trusting persistence studies on survivor-biased data**: bias creates false persistence.
- **Expecting historical returns to precisely identify skill**: noise is too large over career-length horizons.
- **Ignoring fees/costs in "outperformance" findings**: DGTW's 1–1.5% for growth funds used buy-and-hold quarterly portfolios, gross of fees and trading—likely never delivered to investors.

## Reference Tables
**Average-performance evidence**
| Study | Finding |
|---|---|
| Early mutual-fund studies | Underperform risk-adjusted, in proportion to expenses |
| Ippolito (1993) summary | Net of expenses, risk-adjusted ≈ index (at best matches) |
| Malkiel (1995), 1982–1991 | Survivors −0.43%/yr vs S&P 500; all funds −1.83%/yr |
| Lakonishok-Shleifer-Vishny (1992), 341 managers 1983–89 (Ch 16) | −1.3% equal-wtd, −2.6% cap-wtd vs S&P 500 |
| Ferson-Schadt/Warther (1996), 67 funds 1968–90 | Conditioning on public info lifts average from below market to matching it |
| Jones (1998) | Median institutional performance explained by market, small-vs-large, value-vs-growth (cash drag, small and growth bias) |
| DGTW (1997), 2,500+ funds 1975–94 | +1–1.5% only for growth/aggressive growth, mostly 1975–79, gross of fees/costs |
| Marcus (1990) | Very top funds (Magellan) significant beyond max-of-sample luck |

**Persistence evidence**
| No persistence | Persistence |
|---|---|
| Jensen 1968 (115 funds, 1945–64) | Grinblatt-Titman 1988 (157 funds, 1975–84) |
| Kritzman 1983 (32 AT&T bond managers) | Lehmann-Modest 1987 (130 funds, 1968–82) |
| Dunn-Theisen 1983 (201 institutional, 1973–82) | Brown-Draper 1992 (550 UK pension managers) |
| Elton-Gruber-Rentzler 1990 (51 commodity funds) | Hendricks-Patel-Zeckhauser 1993 (165 funds, 1974–88) |
| Kahn-Rudd 1995 (300 equity funds, style/fee-adjusted) | Goetzmann-Ibbotson 1994 (728 funds; 62% repeat) |
| Malkiel 1995 (1980s; present in 1970s) | Bauman-Miller 1994 (only over full market cycles) |
| Carhart 1997 (1,892 funds; only worst losers persist) | Gruber 1996 (270 funds, 1985–94, strong) |
| | Kahn-Rudd 1995 bond funds (persist, but not enough to beat average underperformance) |
Range of the debate: winners repeat ~50% (pessimists) to ~62% (optimists); "perhaps 55%."

**IR 0.5 manager, chance of outperforming**: 1 year 69%; two consecutive years 48%; 5 years 87%; two consecutive 5-year periods 75%.

## Worked Example
- **Skilled fraction**: highest credible repeat probability p = 0.60 ⇒ δ = 1 − 1/1.2 ≈ 0.17 → about 17% of managers persistently skilled (an underestimate given noise).
- **Detecting an IR 0.5 manager**: Φ(0.5) = 0.69 per year; two years running 0.69² ≈ 0.48. Over 5 years Φ(0.5·√5) = Φ(1.12) ≈ 0.87; two consecutive 5-year windows ≈ 0.75—so even with 10 years of tenure, a quarter of genuinely skilled managers fail a repeat-winner test.

## Key Takeaways
1. The average active manager does not beat the benchmark; after survivorship correction US equity funds lag by ~1.8%/yr.
2. Top managers can show statistically significant skill (Marcus), so successful active management is possible.
3. Persistence is weak (~55% repeat), implying ≤ ~17% persistently skilled managers under a simple model.
4. Track records alone cannot reliably identify skill; noise dominates over realistic horizons.
5. Breadth proxies (number of holdings, turnover) predict future performance better than past IR—direct support for the Fundamental Law.
6. Behavioral biases and the logic of market efficiency both argue that superior information plus efficient implementation will keep paying.

## Connects To
- **Ch 6**: Fundamental Law—breadth as predictor of IR.
- **Ch 16**: transactions costs explain average underperformance.
- **Ch 17**: methods (Jensen, style analysis, Fama-French, Ferson-Schadt) used in these studies; IR standard errors.
- **Ch 21**: behavioral finance's predictive power is an open question.
- **External**: efficient markets hypothesis (weak/semistrong/strong); Carhart four-factor model; Ellis's "Loser's Game"; Kahneman-Tversky heuristics and biases.
