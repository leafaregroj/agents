# Chapter 2: Statistical Learning

## Core Idea
Statistical learning means estimating an unknown f in `Y = f(X) + ε`, for **prediction** or **inference**. Choose a method's flexibility to minimize *test* error. That choice is governed by the **bias-variance trade-off**, not by training fit.

## Frameworks Introduced
- **Y = f(X) + ε model**: f is the fixed, unknown systematic part. ε is a random error, independent of X, with mean zero.
  - When to use: as the framing for every supervised problem in the book.
  - How: estimate f̂ from training data `{(x₁,y₁),…,(xₙ,yₙ)}`, then predict with `Ŷ = f̂(X)`.
- **Prediction vs. inference**: for prediction, f̂ can be a black box and only accuracy matters. For inference, f̂'s form matters: which predictors matter, the sign and size of each effect, and whether the relationship is linear.
  - When to use: decide the goal *before* picking a method. Inference pushes you toward restrictive, interpretable models. Pure prediction allows flexible ones.
- **Parametric vs. non-parametric methods**:
  - Parametric (two steps): (1) assume a functional form, e.g. linear `f(X)=β₀+β₁X₁+…+βₚXₚ`; (2) fit or train it, e.g. by least squares. The problem reduces to estimating p+1 parameters.
  - Non-parametric: no explicit form. Fit f̂ close to the data subject to smoothness (e.g. a thin-plate spline with a chosen smoothness level).
  - Why it works / failure mode: a parametric fit is cheap in data but biased if the form is wrong. A non-parametric fit avoids form mis-specification but needs far more observations, and can **overfit** (a rough spline with zero training error).
- **Flexibility–interpretability trade-off** (Fig 2.7): from least flexible and most interpretable to most flexible and least interpretable: subset selection and lasso → least squares → GAMs and trees → bagging, boosting, SVMs with non-linear kernels, deep learning.
- **Bias-variance trade-off**: expected test MSE splits into variance + squared bias + irreducible error. More flexibility lowers bias and raises variance, which gives a U-shaped test MSE.
  - How: pick the flexibility at the bottom of the U. Estimate test error with cross-validation (Ch 5).
  - Why it works: a very flexible model fits the noise (high variance). A rigid model misses the structure (high bias).
- **Bayes classifier**: assign x₀ to the class j that maximizes `Pr(Y=j | X=x₀)`. It is the gold standard and minimizes expected test error rate, but it is unattainable because the conditional distribution is unknown.
- **K-nearest neighbors (KNN) classifier**: estimate class probabilities from the K training points nearest to x₀, then apply Bayes' rule.
  - How: K controls flexibility. K=1 is the most flexible (zero training error). Large K gives a near-linear boundary. Choose K by test or CV error.

## Key Formulas
- Reducible vs. irreducible error (f̂, X fixed):
  `E(Y − Ŷ)² = [f(X) − f̂(X)]² + Var(ε)`. The first term is reducible, the second irreducible. Var(ε) is an upper bound on achievable accuracy.
- Training MSE: `MSE = (1/n) Σᵢ (yᵢ − f̂(xᵢ))²`. Test MSE: `Ave(y₀ − f̂(x₀))²` over unseen (x₀, y₀).
- Bias-variance decomposition (expected test MSE at x₀):
  `E(y₀ − f̂(x₀))² = Var(f̂(x₀)) + [Bias(f̂(x₀))]² + Var(ε)`. The expectation averages over repeated training sets.
- Training error rate (classification): `(1/n) Σᵢ I(yᵢ ≠ ŷᵢ)`. Test error rate: `Ave(I(y₀ ≠ ŷ₀))`.
- Bayes error rate: `1 − E[maxⱼ Pr(Y=j | X)]`. This is the classification analogue of irreducible error.
- KNN: `Pr(Y=j | X=x₀) ≈ (1/K) Σ_{i∈𝒩₀} I(yᵢ = j)`, where 𝒩₀ is the set of K nearest training points to x₀.

## Key Concepts
- **Reducible error**: error from f̂ ≠ f. A better method can shrink it.
- **Irreducible error**: Var(ε), from unmeasured variables or unmeasurable variation. No method removes it.
- **Training data**: the n observations used to fit f̂.
- **Overfitting**: following the noise too closely. It shows up as small training MSE but large test MSE.
- **Flexibility / degrees of freedom**: a summary of how wiggly a fit can be. Linear regression has 2 df in the 1-D examples.
- **Variance (of a method)**: how much f̂ would change with a different training set.
- **Bias (of a method)**: error from approximating a complex real problem with a simpler model.
- **Bayes decision boundary**: the set where the top class probabilities tie (e.g. 50% in two classes).
- **Supervised / unsupervised / semi-supervised**: Y observed for all, none, or only some observations. Cluster analysis is the main unsupervised example.
- **Regression vs. classification problem**: set by whether Y is quantitative or qualitative. Predictor types matter less, because qualitative predictors can be coded.

## Mental Models
- **Training error always falls as flexibility rises. Test error is U-shaped.** Never select a model on training error.
- **Think of bias as "wrong shape" and variance as "chasing noise."** If the truth is near-linear, a linear model wins. If the truth is wiggly, a moderately flexible one wins. If noise dominates (low signal-to-noise ratio), flexible fits lose.
- **Use a restrictive model when inference is the goal.** It can even win at prediction, because flexible models overfit (a counterintuitive point the authors stress).
- **Think of K in KNN as 1/flexibility.** Plot error against 1/K to see the same U-shape as in regression.

## Anti-patterns
- **Choosing the model with the lowest training MSE**: it rewards overfitting. The most flexible spline can have the worst test MSE.
- **Picking K=1 because training error is zero**: a K=1 boundary is overly flexible. Test error of 0.1695 vs 0.1363 at K=10 in the simulated example.
- **Assuming a flexible model is always more accurate**: with a near-linear truth (Fig 2.10), extra flexibility only adds variance.
- **Expecting zero prediction error**: Var(ε) > 0 is a hard floor.

## Code Examples
```python
import pandas as pd, numpy as np
Auto = pd.read_csv('Auto.data', na_values=['?'], delim_whitespace=True)
Auto = Auto.dropna()                    # drop rows with missing horsepower
Auto_re = Auto.set_index('name')
Auto_re.loc[lambda df: (df['year'] > 80) & (df['mpg'] > 30),
            ['weight', 'origin']]
```
- **What it demonstrates**: loading data with `na_values` (the `?` entries made horsepower non-numeric), dropping missing rows, and Boolean and `lambda` selection with `.loc[]`. `.iloc[]` is the positional equivalent.

```python
rng = np.random.default_rng(1303)        # reproducible random numbers
X = rng.standard_normal((10, 3))
X.mean(axis=0)                           # column means
from matplotlib.pyplot import subplots
fig, ax = subplots(figsize=(8, 8)); ax.scatter(Auto['horsepower'], Auto['mpg'], marker='o')
```
- **What it demonstrates**: the lab's NumPy, random-generator and matplotlib idioms (`subplots`, `scatter`, `savefig`).

## Reference Tables
| | Parametric | Non-parametric |
|---|---|---|
| Assumption on f | explicit form (e.g. linear) | none beyond smoothness |
| Data needed | modest | large |
| Main risk | bias from a wrong form | overfitting / variance |
| Interpretability | high | low |
| Example | least squares | thin-plate spline, KNN |

| Simulated scenario (Figs 2.9–2.11) | Best flexibility |
|---|---|
| Truth moderately non-linear | intermediate (smoothing spline) |
| Truth close to linear | low (linear regression nearly optimal) |
| Truth highly non-linear, low noise | high |

## Worked Example
**KNN on simulated 2-class data** (Figs 2.13–2.17, 100 observations per class): the Bayes error rate is about 0.13 (the text reports 0.1304 alongside the KNN results and 0.133 earlier). KNN test error rates: **K=10 → 0.1363** (close to Bayes), **K=1 → 0.1695** (overly flexible, training error 0), **K=100 → 0.1925** (too rigid, near-linear boundary). Test error against 1/K is U-shaped, with its minimum near K=10.

**Income data**: a linear fit `income ≈ β₀ + β₁·education + β₂·seniority` captures the main trend but misses curvature. A smooth thin-plate spline recovers the true surface well. A rough spline interpolates all 30 points and overfits.

## Key Takeaways
1. Statistical learning estimates f in `Y = f(X) + ε`. Only the reducible error can be attacked.
2. The goal (prediction vs. inference) drives the choice between flexible and interpretable models.
3. Judge methods on **test** error. Training error always decreases with flexibility.
4. Expected test MSE = Var + Bias² + Var(ε). Good methods balance the first two.
5. The Bayes classifier is the unreachable benchmark. KNN approximates it, with K setting flexibility.
6. There is no free lunch: no method dominates on every data set, so model selection is a core skill.

## Connects To
- **Ch 3**: linear regression is the canonical parametric, low-variance method. Ch 3 also compares it with KNN regression.
- **Ch 4**: logistic regression, LDA, QDA and naive Bayes estimate `Pr(Y|X)` to approximate the Bayes classifier.
- **Ch 5**: cross-validation estimates test error and picks flexibility (K, smoothness).
- **Ch 6–10**: regularization, splines, trees, SVMs and deep learning as points on the flexibility axis.
- **Ch 12**: unsupervised learning (PCA, clustering).
- **No free lunch theorem / double descent (Ch 10)**: limits and refinements of the U-shaped picture.
