# Chapter 11: Survival Analysis and Censored Data

## Core Idea
When the response is a time-to-event that is often **censored** (we only know the true time exceeds some value), you can't drop or naively use the censored rows. Survival analysis uses the censoring indicator to estimate survival curves (Kaplan–Meier), compare groups (log-rank test), and regress hazards on predictors (Cox proportional hazards), while still evaluating models out of sample (C-index).

## Frameworks Introduced
- **Censored-data setup**: for each observation, true survival time T and censoring time C; we observe Y = min(T, C) and the status indicator δ = 1 if T ≤ C (event observed), δ = 0 if censored.
  - When to use: any time-to-event outcome (death, churn, equipment failure, time to publication) where follow-up ends before every event occurs.
  - How: record (yᵢ, δᵢ, xᵢ). Check that **independent censoring** is plausible: conditional on features, T is independent of C. Patients who drop out because they are too sick violate it, and the data cannot test it, so you have to argue it from domain knowledge.
  - Failure mode: dropping censored rows, or treating Y as if it were T, biases survival estimates. With informative dropout, average survival is typically overestimated.
- **Kaplan–Meier survival curve**: product-limit estimator of S(t) = Pr(T > t).
  - When to use: describe survival in one group, or overlay curves for strata (e.g. by sex).
  - How: sort the unique death times d₁ < … < d_K; at each, count deaths q_k and the number at risk r_k (alive and uncensored just before d_k). Ŝ(d_k) = Π_{j≤k} (r_j − q_j)/r_j. The curve is a step function, flat between death times.
  - Why it works: it chains conditional probabilities, Pr(T > d_k) = Pr(T > d_k | T > d_{k−1})·Pr(T > d_{k−1}), and each censored subject contributes to the risk sets up to the time it was censored.
- **Log-rank test** (Mantel–Haenszel / Cochran–Mantel–Haenszel test): tests H₀ that two groups have the same survival curve.
  - How: at each death time d_k, build a 2×2 table (group × died/survived). Sum observed group-1 deaths X = Σ q₁ₖ, compare with expectation under H₀, standardize to W, and refer W to N(0,1) (or to a permutation null).
  - Failure mode: it is a univariate comparison that adjusts for no covariates (see the Publication example below).
- **Cox's proportional hazards model**: h(t | xᵢ) = h₀(t) · exp(Σⱼ xᵢⱼβⱼ), with h₀(t) an **unspecified baseline hazard**. There is no intercept, because h₀ absorbs it.
  - When to use: regression with a censored response, when you want covariate effects as hazard ratios without assuming a distribution for T.
  - How: maximize the **partial likelihood** (h₀ cancels out). Inference works as in GLMs: Wald z-tests, likelihood-ratio tests. Estimate h₀ afterwards (lifelines does this) if you need S(t | x) curves for representative covariate profiles.
  - Why it works: the semi-parametric form avoids specifying h₀. The proportional hazards assumption means a one-unit change in xⱼ multiplies the hazard by exp(βⱼ) at every t. Failure mode: crossing hazards violate the assumption. Check by plotting log hazard for each level of a qualitative feature (or for strata of a quantitative one); under PH the curves differ by a constant vertical shift. Cox fits are fairly robust to mild violations.
  - Special case: with a single binary predictor, the Cox **score test equals the log-rank test** exactly.
- **Shrinkage for the Cox model**: minimize −log PL(β) + λP(β), with a ridge (Σβⱼ²) or lasso (Σ|βⱼ|) penalty. Choose λ by CV on the **partial likelihood deviance** (twice the CV negative log partial likelihood).
- **Harrell's concordance index (C-index)**: AUC generalized to censored data, used to evaluate a survival model on test data.
- **Additional tools**: careful **choice of time scale** (time zero = diagnosis, treatment, or birth, i.e. age as the time scale); **time-dependent covariates** xᵢ(t), which drop straight into the partial likelihood; **survival trees / random survival forests** (splits chosen to maximize the difference between daughter-node survival curves).

## Key Formulas
- Survival function: S(t) = Pr(T > t), decreasing in t.
- Hazard function: h(t) = lim_{Δt→0} Pr(t < T ≤ t + Δt | T > t)/Δt = f(t)/S(t), the instantaneous death rate given survival to t.
- Likelihood with censoring: L = Πᵢ f(yᵢ)^{δᵢ} S(yᵢ)^{1−δᵢ}. Events contribute the density; censored rows contribute "survived past yᵢ".
- Kaplan–Meier: Ŝ(d_k) = Π_{j=1}^{k} (r_j − q_j)/r_j.
- Log-rank statistic: W = (X − E[X]) / √Var(X), X = Σₖ q₁ₖ, where
  E[q₁ₖ] = r₁ₖ·q_k/r_k and Var(q₁ₖ) = q_k (r₁ₖ/r_k)(1 − r₁ₖ/r_k)(r_k − q_k)/(r_k − 1). Under H₀, W ≈ N(0,1) for large samples. Var(X) is approximated by summing the Var(q₁ₖ), treating the death times as roughly independent.
- Cox partial likelihood: PL(β) = Π_{i: δᵢ=1} exp(Σⱼ xᵢⱼβⱼ) / Σ_{i′: y_{i′} ≥ yᵢ} exp(Σⱼ x_{i′j}βⱼ). Each observed death is compared with its **risk set**. Assumes no tied failure times.
- Penalized Cox: minimize −log PL(β) + λ·P(β).
- C-index: C = Σ_{i,i′: yᵢ > y_{i′}} I(η̂ᵢ′ > η̂ᵢ)·δ_{i′} / Σ_{i,i′: yᵢ > y_{i′}} δ_{i′}, with risk score η̂ᵢ = Σⱼ β̂ⱼxᵢⱼ. A pair counts only if the earlier time is an observed event. The model is concordant when the shorter-lived observation has the higher risk score.

## Key Concepts
- **Censoring time (C)**: when follow-up stops for a subject: study end, dropout, or loss to follow-up.
- **Right censoring**: we know T ≥ Y. This is the chapter's focus. **Left censoring** means T ≤ Y; **interval censoring** means T lies in a known interval.
- **Independent censoring**: T ⫫ C given the features. Required by every method here.
- **Risk set**: subjects still alive and uncensored just before a given failure time.
- **Hazard ratio**: exp(βⱼ), the multiplicative change in hazard per unit of xⱼ, constant over time under PH.
- **Baseline hazard h₀(t)**: hazard when all x = 0. It is left unspecified, which makes Cox semi-parametric.
- **Partial likelihood deviance**: CV loss for penalized Cox models. It replaces test MSE, because CV of the Cox objective is awkward: the objective is not a sum over observations.
- **Time-dependent covariate**: a predictor xᵢ(t) whose value changes during follow-up. Use it to avoid immortal-time bias, e.g. transplant status in the Stanford Heart Transplant data.

## Mental Models
- Think of KM as a running product of "fraction of the risk set that survives each death time". Censored subjects quietly leave the denominator.
- Think of Cox as **logistic-regression-like comparisons at each death time**: "given that someone in the risk set died now, how likely was it to be this person?" That is why h₀ cancels.
- Use the log-rank test for a quick unadjusted two-group comparison, and Cox when you must adjust for confounders. They can disagree sharply.
- You can't compute a "test MSE" for survival: true times are censored and Cox predicts a whole curve S(t | x), not a number. Evaluate with the C-index, or by stratifying test points into risk tertiles using training-set coefficients and checking that the KM curves separate in the right order.

## Anti-patterns
- **Dropping censored observations, or regressing on Y with least squares**: biased, because the censored subjects are systematically the long survivors.
- **Assuming censoring is non-informative without justification**: dropout of very sick patients inflates survival estimates.
- **Coding "ever received a transplant" as a fixed covariate**: patients had to live long enough to receive one. Use a time-dependent covariate.
- **Trusting an unadjusted log-rank p-value as the final word**: confounding can mask a real effect.
- **Including an intercept in the Cox design matrix**: there isn't one (use `intercept=False` in ISLP's `ModelSpec`).
- **Parametric survival models with time-varying predictors**: messy, whereas the partial likelihood handles them naturally.

## Code Examples
```python
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from ISLP.models import ModelSpec as MS
from ISLP import load_data

BrainCancer = load_data('BrainCancer')
km = KaplanMeierFitter()
km_brain = km.fit(BrainCancer['time'], BrainCancer['status'])
km_brain.plot(label='Kaplan Meier estimate')

by_sex = {s: df for s, df in BrainCancer.groupby('sex')}
logrank_test(by_sex['Male']['time'], by_sex['Female']['time'],
             by_sex['Male']['status'], by_sex['Female']['status'])  # stat 1.44, p = 0.23
```
- **What it demonstrates**: a KM curve and a two-group log-rank test from (time, status) columns.

```python
coxph = CoxPHFitter
cleaned = BrainCancer.dropna()
all_MS = MS(cleaned.columns, intercept=False)     # Cox has no intercept
all_df = all_MS.fit_transform(cleaned)
fit_all = coxph().fit(all_df, 'time', 'status')   # duration col, event col
fit_all.summary[['coef', 'se(coef)', 'p']]
fit_all.log_likelihood_ratio_test()
pred = fit_all.predict_survival_function(all_MS.transform(modal_df))  # S(t|x) per profile
```
- **What it demonstrates**: a multivariable Cox fit, an LR test against the null model, and predicted survival curves for representative covariate profiles. For simulation, the lab uses `ISLP.survival.sim_time(linpred, cum_hazard, rng)` and `multivariate_logrank_test` for comparing more than two groups.

## Reference Tables

| Tool | Purpose | Adjusts for covariates? | Output |
|---|---|---|---|
| Kaplan–Meier | Estimate S(t) | No (stratify only) | Step curve |
| Log-rank test | Compare ≥2 survival curves | No | W ~ N(0,1), p-value |
| Cox PH | Regression on hazard | Yes | Hazard ratios exp(βⱼ), tests |
| Lasso/ridge Cox | High-dimensional or sparse Cox | Yes | Shrunken β, CV on PL deviance |
| C-index | Test-set evaluation | n/a | Pairwise concordance in [0,1] |
| Survival trees / RSF | Nonlinear, interactions | Yes | Node survival curves |

BrainCancer Cox fit (Table 11.2, n = 88, 53 still alive at study end):

| Covariate | β̂ | p |
|---|---|---|
| sex[Male] | 0.18 (HR 1.2) | 0.61 |
| diagnosis[LG Glioma] | 0.92 | 0.15 |
| diagnosis[HG Glioma] | 2.15 | <0.01 |
| diagnosis[Other] | 0.89 | 0.18 |
| ki (Karnofsky) | −0.05 (HR 0.95/unit) | <0.01 (0.0027) |
| gtv | 0.03 | 0.12 |

## Worked Example
**Publication data** (244 NHLBI-funded clinical trials, 156 published during follow-up; response = months until publication; the "event" is publication, so a higher hazard means faster publication).
1. KM curves stratified by `posres` (positive result) differ only slightly. The log-rank p-value is **0.36**.
2. A Cox model with all covariates (clinend, multi, mech, sampsize, budget, impact, posres) gives posres β̂ = 0.55, i.e. **e^0.55 ≈ 1.74×** the instantaneous chance of publication, with a very small p-value. After adjustment, a positive result strongly predicts faster publication. The log-rank test missed it because it ignored covariates. Adjusted survival curves were built from estimated h₀(t), covariates at their means, and the modal mech level (R01).
3. Lasso-penalized Cox on a 50/50 train/test split: CV on partial likelihood deviance is U-shaped, and the minimum keeps just **budget** and **impact**. Test points stratified into low/medium/high risk tertiles by budget·β̂_budget + impact·β̂_impact give cleanly separated, correctly ordered KM curves.
4. Cox fit on the training set, evaluated on the test set: **C = 0.733**. For a random pair of papers, the model picks the one published first about 73% of the time.

Also: for BrainCancer, the log-rank test of sex gives W = 1.2, p ≈ 0.2 (theoretical) or 0.25 (1,000 permutations). The lab's Cox model with sex alone gives β̂ = 0.41, p = 0.23, the same statistic as the lab's log-rank test (1.44), illustrating score test = log-rank.

## Key Takeaways
1. Model the observed pair (Y, δ). Censored rows carry real information ("survived at least this long").
2. Everything rests on independent censoring, which you have to argue from context, not test.
3. KM for description, log-rank for an unadjusted comparison, Cox for adjusted effects reported as hazard ratios.
4. Cox's partial likelihood removes h₀(t). Coefficients are log hazard ratios, there is no intercept, and the effect is constant over time under PH.
5. Ridge and lasso carry over directly (penalize −log PL). Tune λ by CV on partial likelihood deviance.
6. Evaluate with the C-index or risk-stratified KM curves, not MSE.
7. Time-dependent covariates and survival trees/forests extend the toolkit. Choose time zero deliberately.

## Connects To
- **Ch 4**: logistic regression, ROC/AUC. The C-index generalizes AUC, and the Cox partial likelihood resembles conditional logistic regression.
- **Ch 5**: cross-validation, here on partial likelihood deviance. The permutation null for the log-rank test previews Ch 13.
- **Ch 6**: ridge and lasso penalties reused for the Cox model.
- **Ch 8**: trees, random forests → survival trees and random survival forests.
- **Ch 13**: permutation/re-sampling p-values.
- **External**: Nelson–Aalen estimator, accelerated failure time models, competing risks, Schoenfeld residuals for PH checks, `lifelines` and `scikit-survival` libraries.
