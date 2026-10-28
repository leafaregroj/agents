# Chapter 13: Multiple Testing

## Core Idea
Test m null hypotheses at level α each and you should expect about α·m false positives among the true nulls. Rare events become inevitable at scale. Fix this by controlling a **family-wise** error measure: the **FWER** (Bonferroni, Holm, Tukey, Scheffé) when any false positive is costly, or the **FDR** (Benjamini–Hochberg) when you want many discoveries with a tolerable fraction of false ones. When no trustworthy theoretical null exists, get p-values and FDRs by **re-sampling/permutation**.

## Frameworks Introduced
- **Hypothesis testing in four steps**: (1) define H₀ and H_a; (2) build a test statistic T (e.g. the two-sample t-statistic); (3) compute the p-value, the probability under H₀ of a statistic at least as extreme as observed; (4) reject H₀ if p < α. "Fail to reject" ≠ accept.
- **Family-Wise Error Rate (FWER) control**: guarantee Pr(V ≥ 1) ≤ α, i.e. at least one false rejection happens with probability at most α.
  - **Bonferroni**: reject H₀ⱼ if pⱼ < α/m. Valid for any dependence among the p-values (union bound). Very conservative for large m, so power is low.
  - **Holm's step-down procedure** (Holm–Bonferroni, Algorithm 13.1): order p₍₁₎ ≤ … ≤ p₍ₘ₎; let L = min{j : p₍ⱼ₎ > α/(m + 1 − j)}; reject all H₀ⱼ with pⱼ < p₍L₎. Same assumption-free guarantee, uniformly **more powerful than Bonferroni**, so always prefer it.
  - **Tukey's method**: for all m = G(G−1)/2 **pairwise comparisons of G means**. It exploits the dependence among those tests to use a threshold above α/m.
  - **Scheffé's method**: a threshold S valid for **arbitrary linear contrasts** of group means, including contrasts chosen after looking at the data.
  - When to use FWER: a small number of tests (a handful to dozens), or when a single false positive is very costly (confirmatory clinical claims).
- **False Discovery Rate (FDR) control**: guarantee E(V/R) ≤ q, the expected fraction of rejections that are false.
  - **Benjamini–Hochberg (BH) procedure** (Algorithm 13.2): specify q; order the p-values; let L = max{j : p₍ⱼ₎ < q·j/m}; reject all H₀ⱼ with pⱼ ≤ p₍L₎. Guarantees FDR ≤ q (strictly, ≤ q·m₀/m) when the p-values are independent or only mildly dependent.
  - When to use: exploratory, large-m settings (genomics with 20,000 genes, screening 2,000 fund managers) where follow-up studies will weed out false leads. There is no conventional q. Pick it from context (e.g. 10–20%).
  - Why it works / failure mode: the threshold adapts to the data. With many small p-values BH becomes lenient, unlike Bonferroni's fixed α/m. Like Holm, you can't know in advance whether a given p-value will be rejected, since that depends on the other p-values.
- **Re-sampling (permutation) p-value** (Algorithm 13.3): compute T on the data; for b = 1,…,B (e.g. B = 10,000) permute the n_X + n_Y pooled observations, split into groups of sizes n_X and n_Y, and recompute T*ᵇ; p = (1/B)Σ_b 1(|T*ᵇ| ≥ |T|). Requires the stronger null that X and Y have the **same distribution**, which makes the labels exchangeable.
- **Re-sampling FDR (plug-in estimate, Algorithm 13.4)**: for threshold c, set R = Σⱼ 1(|Tⱼ| ≥ c). Estimate V̂ by permuting each hypothesis's data B times and counting |T*ⱼᵇ| ≥ c, giving V̂ = (1/B)ΣᵦΣⱼ 1(|T*ⱼᵇ| ≥ c). Then FDR̂(c) = V̂/R. Vary c to trade discoveries against FDR.

## Key Formulas
- Two-sample t-statistic: T = (μ̂_X − μ̂_Y) / (s·√(1/n_X + 1/n_Y)), with pooled s = √[((n_X−1)s_X² + (n_Y−1)s_Y²)/(n_X + n_Y − 2)]. Under H₀ with normal data, T ~ t_{n_X+n_Y−2}. For large n, T ≈ N(0,1).
- Independent tests: FWER = 1 − (1 − α)^m. With α = 0.05 and m = 100, FWER ≈ 0.994.
- Bonferroni bound: FWER ≤ Σⱼ Pr(reject true H₀ⱼ) ≤ m·(α/m) = α.
- Holm: L = min{j : p₍ⱼ₎ > α/(m + 1 − j)}. Reject p-values below p₍L₎.
- FDR = E(FDP) = E(V/R), with V/R := 0 when R = 0.
- BH: L = max{j : p₍ⱼ₎ < qj/m}. Reject pⱼ ≤ p₍L₎.
- **q-value** (BH-adjusted p-value): the smallest FDR level at which H₀ⱼ would be rejected. Reject all hypotheses with q-value ≤ q.
- Adjusted p-values: Bonferroni = min(m·pⱼ, 1). Reject when adjusted p ≤ α.
- Permutation p-value: p = (1/B) Σ_{b=1}^{B} 1(|T*ᵇ| ≥ |T|).
- Pooled re-sampling p-value for many tests: pⱼ = Σ_{j′=1}^{m} Σ_{b=1}^{B} 1(|T*_{j′}ᵇ| ≥ |Tⱼ|) / (Bm). Running BH on these is equivalent to Algorithm 13.4's plug-in approach.

## Key Concepts
- **Type I error**: rejecting a true H₀ (false positive). Its rate is controlled at α.
- **Type II error**: failing to reject a false H₀. **Power** = 1 − Pr(Type II) = Pr(reject H₀ | H_a).
- **Outcome table (Table 13.2)**: V = false positives (Type I), S = true positives, U = true negatives, W = Type II errors. R = V + S rejections, m₀ true nulls.
- **FWER**: Pr(V ≥ 1).
- **FDP / FDR**: V/R and its expectation.
- **Theoretical null distribution**: a known reference distribution for T under H₀ (t, χ², N(0,1)).
- **Re-sampling null distribution**: the empirical distribution of T over permuted data. Prefer it when samples are small or the data are skewed or outlier-ridden. If it disagrees with the theoretical null, trust the re-sampling result.
- **Data snooping**: choosing which hypothesis to test after looking at the data. It implicitly performs many tests, so it needs Tukey/Scheffé-style correction.

## Mental Models
- **The stockbroker/coin trap**: flip 1,024 fair coins 10 times each and about one comes up all tails (p < 0.002), yet it's fair. The tiny p-value is simply the expected best of 1,024 draws. Any "best performer out of many" looks significant by chance.
- **FWER vs power is a hard trade-off**: as m grows, FWER control forces thresholds so low you detect almost nothing. Use FDR when false leads are cheap and missed discoveries are costly.
- Think of BH as a **sloped line q·j/m** on the sorted p-value plot. Reject everything up to the last point under the line. Bonferroni is the flat line α/m, and Holm's staircase α/(m+1−j) lies between them.
- Use Holm instead of Bonferroni by default. Same guarantee, never fewer rejections.

## Anti-patterns
- **Testing many hypotheses at α = 0.05 with no correction**: you expect α·m₀ false positives.
- **Testing the most extreme pair after inspecting the data**: a post hoc two-sample test of the best vs worst manager is really one of G(G−1)/2 tests. Use Tukey (pairwise) or Scheffé (arbitrary contrasts).
- **Bonferroni in large-m exploratory work**: severe loss of power. On the Fund data it rejects nothing.
- **Reading FDR as the probability that a particular discovery is false**: it is an expectation over the whole rejected set.
- **Theoretical nulls with small n or heavy skew/outliers**: p-values can be badly off. Permute instead.
- **Permutation tests when H₀ only equates means but the distributions differ** (e.g. variances): the exchangeability assumption fails.
- **Bonferroni's p-threshold applied to FDR, or vice versa**: they control different quantities. Pick the error rate first.

## Code Examples
```python
from scipy.stats import ttest_1samp, ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests as mult_test
from statsmodels.stats.multicomp import pairwise_tukeyhsd

fund_mini_pvals = np.array([ttest_1samp(fund_mini.iloc[:, i], 0).pvalue
                            for i in range(5)])
reject, bonf = mult_test(fund_mini_pvals, method="bonferroni")[:2]   # rejects Manager 1 only
mult_test(fund_mini_pvals, method="holm", alpha=0.05)[:2]            # rejects Managers 1 and 3

fund_pvalues = np.array([ttest_1samp(Fund.iloc[:, i], 0).pvalue for i in range(2000)])
fund_qvalues = mult_test(fund_pvalues, method="fdr_bh")[1]
(fund_qvalues <= 0.1).sum()              # 146 discoveries at FDR 10%
(fund_pvalues <= 0.1 / 2000).sum()       # 0 under Bonferroni at FWER 0.1

returns = np.hstack([fund_mini.iloc[:, i] for i in range(5)])
managers = np.hstack([[i + 1] * 50 for i in range(5)])
print(pairwise_tukeyhsd(returns, managers).summary())
```
- **What it demonstrates**: FWER corrections (Bonferroni, Holm), BH q-values, and Tukey's HSD from one API (`multipletests` methods `"bonferroni"`, `"holm"`, `"fdr_bh"`).

```python
# Permutation p-value for one gene (Khan data), Algorithm 13.3
observedT = ttest_ind(D2[gene], D4[gene], equal_var=True).statistic
B = 10000; Tnull = np.empty(B); D_ = np.hstack([D2[gene], D4[gene]]); n_ = D2[gene].shape[0]
for b in range(B):
    rng.shuffle(D_)
    Tnull[b] = ttest_ind(D_[:n_], D_[n_:], equal_var=True).statistic
(np.abs(Tnull) > np.abs(observedT)).mean()
```
- **What it demonstrates**: building the null by shuffling the pooled observations. Variable names are lightly simplified from the lab.

## Reference Tables

| Method | Controls | Assumptions | Power | Use when |
|---|---|---|---|---|
| No correction | Per-test Type I | — | Highest | Single pre-specified test |
| Bonferroni | FWER | None | Lowest | Few tests, simplicity |
| Holm | FWER | None | ≥ Bonferroni, always | Default FWER method |
| Tukey | FWER | All pairwise mean comparisons | > Bonferroni there | Post hoc pairwise comparisons |
| Scheffé | FWER | Linear contrasts of means | Suited to arbitrary post hoc contrasts | Contrasts chosen after looking |
| Benjamini–Hochberg | FDR ≤ q | Independent / mildly dependent p's | Much higher at large m | Exploratory screening |
| Re-sampling p / FDR | Same as the procedure it feeds | Exchangeability under H₀ | — | No reliable theoretical null |

Fund example, first 5 managers (n = 50 months, H₀: mean excess return = 0):

| Manager | x̄ | s | t | p |
|---|---|---|---|---|
| One | 3.0 | 7.4 | 2.86 | 0.006 |
| Two | −0.1 | 6.9 | −0.10 | 0.918 |
| Three | 2.8 | 7.5 | 2.62 | 0.012 |
| Four | 0.5 | 6.7 | 0.53 | 0.601 |
| Five | 0.3 | 6.8 | 0.31 | 0.756 |

## Worked Example
**Fund data, FWER = 0.05 over m = 5 managers.**
- Bonferroni threshold 0.05/5 = 0.01: only Manager One (0.006) is rejected. Manager Three (0.012) just misses.
- Holm, with ordered p: 0.006, 0.012, 0.601, 0.756, 0.918. Compare with 0.05/5 = 0.01 ✓, 0.05/4 = 0.0125 ✓, 0.05/3 = 0.0167 ✗, so L = 3. **Reject Managers One and Three.** Holm beats Bonferroni.
- BH at q = 0.05: 0.006 < 0.01 ✓, 0.012 < 0.02 ✓, 0.601 > 0.03 ✗. Reject One and Three.
- Post hoc: a test of Manager One vs Two gives p ≈ 0.035 (two-sample) or 0.038 (paired, lab). But it was chosen as best vs worst among 10 implicit pairs, so Bonferroni with m = 10 needs p < 0.005 and the result is not significant. Tukey's HSD adjusts properly and also does not reject.
- Scheffé: H₀: (μ₁ + μ₃)/2 = (μ₂ + μ₄ + μ₅)/3 has a naive p = 0.004, but the Scheffé threshold at α = 0.05 is S = 0.002, so don't reject. The same S works for any other contrast.

**All 2,000 managers**: BH at q = 0.1 gives **146 rejections**, about 15 of them expected false. At q = 0.3 it gives **279 rejections** (≤ ~84 false). Bonferroni at FWER 0.1 rejects **none**, and even at FWER 0.3 only one.

**Khan genes (n_X = 29 rhabdomyosarcoma, n_Y = 25 Burkitt's)**: gene 11 has T = −2.09, theoretical (t₅₂) p = 0.041 vs permutation p = 0.042, nearly identical. Gene 877 has T = −0.57, p = 0.571 vs 0.673. A single extreme outlier makes the theoretical null inaccurate. Across all 2,308 genes, the plug-in re-sampling FDR and BH give almost identical FDR estimates for a given number of rejections.

## Key Takeaways
1. With m tests at level α, you'll get about α·m₀ false positives. The best of many looks significant by chance.
2. FWER (Pr(V ≥ 1) ≤ α) is strict. Use Holm over Bonferroni, and Tukey or Scheffé for pairwise or contrast tests after looking at the data.
3. FDR (E[V/R] ≤ q) via Benjamini–Hochberg is the practical choice for large-m exploratory work. Choose q by context. There is no 0.05 convention.
4. BH and Holm thresholds are data-dependent. Bonferroni's is fixed.
5. Permutation nulls free you from distributional assumptions and matter most for small or skewed samples. Re-sampling FDR estimates plug a permuted V̂ into V/R.
6. Decide the error rate (per-test, FWER, FDR) before looking at results.

## Connects To
- **Ch 3**: t-statistics and p-values for regression coefficients. Scanning many coefficients is multiple testing.
- **Ch 5**: the bootstrap and resampling ideas behind permutation nulls.
- **Ch 11**: log-rank permutation p-values. Survival comparisons across many strata also need correction.
- **Ch 12**: exploratory clustering and screening often produce many implicit tests.
- **Ch 6**: post-selection inference. p-values after model selection are invalid without correction.
- **External**: Storey's q-value / π₀ estimation, Benjamini–Yekutieli (FDR under arbitrary dependence), Westfall–Young permutation FWER, selective inference, p-hacking and reproducibility.
