# Chapter 4: Classification

## Core Idea
Classifiers estimate `Pr(Y=k | X=x)` and assign the most probable class, approximating the Bayes classifier. They do it either directly (**logistic regression**, KNN) or generatively via Bayes' theorem (**LDA**, **QDA**, **naive Bayes**). Which wins depends on how linear the true boundary is and on n relative to p. **GLMs** (logistic, Poisson) unify regression for non-Gaussian responses.

## Frameworks Introduced
- **Why not linear regression for a qualitative Y**: dummy coding of more than 2 unordered classes imposes a false ordering and spacing. For a binary Y, least squares estimates Pr(Y=1|X) but can produce values outside [0,1].
- **Logistic regression**: model `p(X) = Pr(Y=1|X) = e^{β₀+β₁X}/(1+e^{β₀+β₁X})`, which is linear in the log-odds.
  - When to use: 2 classes, when you want interpretable effects with inference (z-statistics), and when there are no strong distributional assumptions about X.
  - How: fit by **maximum likelihood**. Interpret βⱼ as a change in log-odds per unit of Xⱼ (odds multiply by e^{βⱼ}). Classify with threshold 0.5 by default, and lower it if missing positives is costly.
  - Failure mode: unstable estimates when classes are perfectly or near-perfectly separated, and the confounding risk shown in the Default student example.
- **Multinomial logistic regression**: K classes with baseline class K: `log[Pr(Y=k|x)/Pr(Y=K|x)] = βₖ₀ + βₖᵀx`. The alternative is **softmax coding** (symmetric, one coefficient vector per class, used in ML). Fitted probabilities and log-odds contrasts are the same under either coding.
- **Generative models via Bayes' theorem**: `pₖ(x) = πₖ fₖ(x) / Σₗ πₗ fₗ(x)`. Estimate the priors πₖ (class fractions) and the class densities fₖ(x), then plug in.
  - Use instead of logistic regression when: classes are well separated (logistic is unstable), predictors are approximately normal within each class and n is small, or there are more than 2 classes.
- **Linear discriminant analysis (LDA)**: `fₖ = N(μₖ, Σ)` with a **shared covariance** Σ. This gives linear discriminant functions δₖ(x). Assign x to argmaxₖ δₖ(x).
  - How: μ̂ₖ = class means, Σ̂ = pooled within-class covariance, π̂ₖ = nₖ/n.
  - Why it works / failure mode: few parameters means low variance, so it is good with small n or many classes. It is biased if the class covariances truly differ.
- **Quadratic discriminant analysis (QDA)**: `fₖ = N(μₖ, Σₖ)` with **class-specific covariance**. This gives quadratic boundaries. It estimates K·p(p+1)/2 covariance parameters vs. p(p+1)/2 for LDA.
  - When to use: large n, or clearly unequal covariances. Avoid with small n and large p (variance).
- **Naive Bayes**: assume the predictors are **independent within each class**, `fₖ(x) = fₖ₁(x₁)·…·fₖₚ(xₚ)`. Model each fₖⱼ as a 1-D Gaussian, a kernel density estimate (KDE), or category proportions for qualitative Xⱼ.
  - Why it works: the assumption is almost never true, but it trades bias for a large reduction in variance. This works well when p is large relative to n, where joint densities are hopeless to estimate.
- **KNN classifier**: non-parametric, no boundary assumptions. Choose K by CV. Standardize predictors first. It gives no table of coefficients, and it needs large n relative to p.
- **Thresholds, confusion matrix and ROC**: vary the threshold to trade sensitivity against specificity. The ROC curve plots TPR against FPR across all thresholds. **AUC** summarizes it (0.5 = chance).
- **Generalized linear models (GLMs)**: (1) Y given X belongs to the exponential family (Gaussian, Bernoulli, Poisson, Gamma, negative binomial); (2) the mean is modeled through a **link function** `η(E[Y|X]) = β₀ + Σβⱼ Xⱼ`. Links: identity (linear), logit (logistic), log (Poisson).
- **Poisson regression**: for counts, `log λ(X) = β₀ + Σβⱼ Xⱼ`, `Pr(Y=k) = e^{−λ}λᵏ/k!`. A one-unit increase in Xⱼ multiplies E[Y] by e^{βⱼ}. It assumes mean = variance, so it handles mean-variance growth that linear regression ignores, and it gives no negative predictions.

## Key Formulas
- Odds: `p/(1−p) = e^{β₀+β₁X}`. Log-odds (logit): `log[p/(1−p)] = β₀ + β₁X`.
- Likelihood: `ℓ(β₀,β₁) = Πᵢ:yᵢ=1 p(xᵢ) · Πᵢ':yᵢ'=0 (1 − p(xᵢ'))`. β̂ maximizes it.
- Softmax: `Pr(Y=k|x) = e^{βₖ₀+βₖᵀx} / Σₗ e^{βₗ₀+βₗᵀx}`.
- LDA, p = 1: `δₖ(x) = x·μₖ/σ² − μₖ²/(2σ²) + log πₖ`. With 2 classes and equal priors, the boundary is x = (μ₁+μ₂)/2.
  Estimates: `σ̂² = (1/(n−K)) Σₖ Σ_{i:yᵢ=k} (xᵢ − μ̂ₖ)²`.
- LDA, p > 1: `δₖ(x) = xᵀΣ⁻¹μₖ − ½ μₖᵀΣ⁻¹μₖ + log πₖ`, which is linear in x.
- QDA: `δₖ(x) = −½(x−μₖ)ᵀΣₖ⁻¹(x−μₖ) − ½ log|Σₖ| + log πₖ`, which is quadratic in x.
- Log-odds against baseline class K, `log[Pr(Y=k|x)/Pr(Y=K|x)]`:
  - LDA: `aₖ + Σⱼ bₖⱼ xⱼ` (linear, same form as logistic regression).
  - QDA: `aₖ + Σⱼ bₖⱼ xⱼ + Σⱼ Σₗ cₖⱼₗ xⱼ xₗ` (quadratic).
  - Naive Bayes: `aₖ + Σⱼ gₖⱼ(xⱼ)` (additive, i.e. a GAM).
  - Multinomial logistic: `βₖ₀ + Σⱼ βₖⱼ xⱼ` (linear, fit by maximum likelihood rather than by Gaussian assumptions).
- Measures: sensitivity = TPR = TP/P (recall, power). Specificity = TN/N. FPR = FP/N = 1 − specificity (Type I error). Precision = PPV = TP/P̂ = 1 − false discovery proportion. NPV = TN/N̂.
- Poisson: `E[Y] = Var[Y] = λ`. `λ(X) = e^{β₀+Σβⱼ Xⱼ}`.

## Key Concepts
- **Maximum likelihood**: choose parameters that make the observed labels most probable. It is the general fitting principle for logistic regression and GLMs.
- **Confounding**: a predictor's marginal effect differs from its partial effect because it is correlated with another predictor (student vs. balance).
- **Bayes decision boundary / Bayes classifier**: the optimal rule if the true pₖ(x) were known. LDA and QDA estimate it under Gaussian assumptions.
- **Null classifier**: always predict the majority class. Use it as the baseline error (3.33% on Default).
- **Confusion matrix**: a cross-tabulation of predicted and true classes. It reveals class-specific errors that the overall error rate hides.
- **ROC curve / AUC**: threshold-free performance. AUC = 0.95 for LDA on Default.
- **Discriminant function δₖ(x)**: the score whose argmax gives the predicted class.
- **Link function**: transforms the mean of Y so that it is linear in the predictors (logit, log, identity).
- **Overdispersion / mean-variance relationship**: count variance grows with the mean. Poisson models this, and linear regression's constant variance does not.

## Mental Models
- **Use logistic regression or LDA when the true boundary is linear.** Use LDA if the predictors are roughly Gaussian or classes are separated. Use logistic regression if the Gaussian assumption is doubtful (e.g. heavy tails).
- **Use QDA or naive Bayes when the boundary is moderately non-linear.** Use KNN with K tuned by CV for highly non-linear boundaries with ample n. Avoid KNN-1.
- **Think of LDA → naive Bayes → QDA as a bias-variance dial.** LDA shares Σ (low variance). QDA estimates a Σₖ per class (high variance). Naive Bayes zeroes off-diagonal covariances (low variance even for large p). With Gaussian fₖⱼ, naive Bayes is a special case of LDA with diagonal Σ. Neither QDA nor naive Bayes is a special case of the other.
- **Choose the threshold from costs, not by default.** Lowering it from 0.5 to 0.2 catches far more defaulters for a small increase in overall error.
- **Use GLMs for counts and other non-Gaussian responses.** Fitting least squares to counts gives negative predictions and ignores how the variance grows with the mean.

## Anti-patterns
- **Linear regression for multi-class unordered Y**: dummy coding imposes an artificial order and spacing.
- **Judging a classifier by overall error alone with imbalanced classes**: LDA's 2.75% error is barely better than the 3.33% null rate and misses 75.7% of defaulters.
- **Interpreting a marginal coefficient as causal**: student is positive alone (0.4049) but negative given balance (−0.6468). Students carry higher balances.
- **Non-standardized predictors in KNN**: variables with large scales (e.g. salary in dollars) dominate the distance.
- **QDA with small n and many predictors**: too many covariance parameters, so variance blows up.
- **Naive Bayes with strongly correlated predictors in large-n settings**: the independence bias stops paying off (Scenarios 2–4).
- **Evaluating on training data**: Smarket logistic regression shows 52.2% training accuracy and only 48% on 2005.

## Code Examples
```python
import statsmodels.api as sm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS
Smarket = load_data('Smarket')
train = (Smarket.Year < 2005)
model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket); y = Smarket.Direction == 'Up'
X_train, X_test = X.loc[train], X.loc[~train]
results = sm.GLM(y[train], X_train, family=sm.families.Binomial()).fit()
probs = results.predict(exog=X_test)
labels = np.where(probs > 0.5, 'Up', 'Down')
confusion_table(labels, Smarket.Direction[~train])
```
- **What it demonstrates**: logistic regression as a GLM (`sm.GLM` with the `Binomial` family), a temporal train/test split (train ≤ 2004, test 2005), thresholding, and `confusion_table`. `sm.families.Poisson()` fits Poisson regression the same way.

```python
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA,
                                           QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
lda = LDA(store_covariance=True).fit(X_train, L_train)   # drop intercept column first
lda_prob = lda.predict_proba(X_test)                     # threshold e.g. lda_prob[:,0] > 0.9
qda = QDA(store_covariance=True).fit(X_train, L_train)
NB = GaussianNB().fit(X_train, L_train)
X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(feature_df)
knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
```
- **What it demonstrates**: the sklearn estimators for the generative classifiers and KNN. Standardize before KNN.

## Reference Tables
**Method comparison (Section 4.5, six simulation scenarios)**
| Method | Boundary | Key assumption | Best when | Weak when |
|---|---|---|---|---|
| Logistic regression | linear | none on X distribution | linear boundary, non-Gaussian X (Scen. 3, t-dist) | classes separated, small n |
| LDA | linear | Gaussian, shared Σ | linear boundary, Gaussian X (Scen. 1–2) | unequal Σ, heavy tails |
| QDA | quadratic | Gaussian, Σₖ per class | quadratic boundary (Scen. 4), large n | small n, large p |
| Naive Bayes | additive in each xⱼ | independent within class | independent predictors, tiny n (Scen. 1, 6) | correlated predictors (Scen. 2–4) |
| KNN-CV | arbitrary | none | complex non-linear boundary (Scen. 5) | small n, large p |
| KNN-1 | arbitrary, very rough | none | — | almost always too much variance |

**Default data (n = 10,000, 333 defaulters), training-set confusion results**
| Classifier, threshold | Overall error | Defaulters caught / 333 | Missed rate |
|---|---|---|---|
| Null (always No) | 3.33% | 0 | 100% |
| LDA, 0.5 | 2.75% | 81 | 75.7% |
| LDA, 0.2 | 3.73% | 195 | 41.4% |
| Naive Bayes, 0.5 | 2.90% | 89 | 73.3% |
| Naive Bayes, 0.2 | 4.58% | 203 | 39.0% |

**Smarket lab (test = 2005, 252 days)**: logistic with all lags is 48% correct (worse than chance). Logistic or LDA with Lag1+Lag2 is 56% (the same as always predicting "Up"), and 58% on days predicted Up. QDA is about 60%. Naive Bayes is about 59%. KNN K=1 is 50%. **Caravan** (6% buyers): KNN on standardized data hits 14.5% (K=1) among predicted buyers. Logistic at threshold 0.25 hits about 31% (9/29).

## Worked Example
**Default logistic regression.** `default ~ balance`: β̂₀ = −10.6513, β̂₁ = 0.0055 (each $1 of balance adds 0.0055 to the log-odds). p̂(1000) = e^{−10.6513+5.5}/(1+…) = **0.00576**. p̂(2000) = **0.586**. `default ~ student`: β̂ = 0.4049, so Pr = **0.0431** for students vs **0.0292** for non-students. Multiple model `balance + income($1000s) + student`: β̂ = −10.869, 0.00574, 0.0030 (p = 0.71), **−0.6468**. The student coefficient flips sign. At balance $1,500 and income $40k, a student has p̂ = **0.058** vs **0.105** for a non-student. At a *given* balance, students are less risky. Students tend to carry higher balances, so marginally they look riskier: confounding.

**LDA on Default.** The 2.75% error looks good, but the null classifier gets 3.33%, and LDA misses 252/333 = 75.7% of defaulters (23 false positives). Lowering the threshold to Pr(default) > 0.2 cuts the misses to 138 (41.4%) while overall error rises only to 3.73%. AUC = 0.95.

**Bikeshare (counts).** Least squares on `bikers ~ mnth + hr + workingday + temp + weathersit` gives 9.6% negative fitted values, and the variance grows with the mean. Poisson regression (log link) fixes both. Coefficients become multiplicative effects (e.g. clear weather vs. rain).

## Key Takeaways
1. Model `Pr(Y=k|X)` rather than Y directly. Logistic regression does it via the logit, and generative methods via Bayes' theorem.
2. LDA, logistic regression, naive Bayes and QDA differ mainly in the form of their log-odds: linear, linear, additive, quadratic.
3. No method dominates. Match boundary complexity and n/p to method flexibility, and tune KNN's K by CV.
4. With imbalanced classes, report the confusion matrix, sensitivity and specificity, and ROC/AUC, and set the threshold from domain costs.
5. Watch for confounding in multivariable logistic models (the student example).
6. GLMs (linear, logistic, Poisson) share the structure exponential family + link + linear predictor. Use Poisson for counts.

## Connects To
- **Ch 2**: the Bayes classifier, KNN and the bias-variance trade-off underlie every method here.
- **Ch 3**: logistic regression and GLMs extend linear-model inference (coefficients, SEs, dummy variables, interactions).
- **Ch 5**: cross-validation picks K and estimates test error. The bootstrap gives SEs.
- **Ch 6**: regularized (ridge/lasso) logistic regression for large p.
- **Ch 7**: naive Bayes log-odds have GAM form. Logistic GAMs give flexible boundaries.
- **Ch 8–10**: trees, SVMs (whose hinge loss is close to the logistic loss) and neural networks (whose softmax output layer is multinomial logistic).
- **Ch 11**: survival models, another GLM-like setting with censoring.
