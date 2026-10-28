# Chapter 9: Support Vector Machines

## Core Idea
The chapter builds up in three steps. The maximal margin classifier separates two classes with the widest possible hyperplane margin. The support vector classifier allows a budget of margin violations. The SVM swaps inner products for kernels so it can fit non-linear boundaries cheaply. Underneath, the support vector classifier is hinge loss plus a ridge penalty, which makes it a close relative of logistic regression.

## Frameworks Introduced
- **Hyperplane / separating hyperplane**: a hyperplane is β₀ + β₁X₁ + … + β_pX_p = 0. Classify by the sign of f(x*) = β₀ + βᵀx*. The magnitude |f(x*)| measures confidence, since it reflects distance from the hyperplane. A separating hyperplane satisfies yᵢ(β₀ + βᵀxᵢ) > 0 for all i, with y ∈ {−1, +1}.
- **Maximal margin classifier (optimal separating hyperplane)**: among all separating hyperplanes, pick the one farthest from the training data.
  - When to use: almost never as-is. It applies only when the classes are perfectly separable, and even then it can overfit when p is large.
  - Failure mode: it depends only on the few *support vectors*. Adding one observation can swing the hyperplane dramatically (Fig. 9.5), and no solution exists if the classes overlap.
- **Support vector classifier (soft margin classifier)**: maximize M subject to Σβⱼ² = 1, yᵢ(β₀ + βᵀxᵢ) ≥ M(1 − εᵢ), εᵢ ≥ 0, and Σεᵢ ≤ C.
  - When to use: the class boundary is roughly linear, the classes overlap, or you want robustness to individual observations.
  - How: choose the budget C by cross-validation. Slack εᵢ = 0 means the observation is on the correct side of the margin. 0 < εᵢ ≤ 1 means it violates the margin. εᵢ > 1 means it is on the wrong side of the hyperplane. At most C observations can be misclassified.
  - Why it works: only observations on or inside the margin (the support vectors) determine the fit. A large C gives a wide margin, many support vectors, low variance and higher bias. A small C gives the opposite.
- **Support vector machine (SVM) / kernel trick**: the classifier's solution depends on the data only through inner products ⟨xᵢ, xᵢ′⟩. Replace each one with a kernel K(xᵢ, xᵢ′) to get f(x) = β₀ + Σ_{i∈S} αᵢK(x, xᵢ). The coefficients αᵢ are nonzero only for support vectors.
  - When to use: the class boundary is non-linear.
  - How: pick a kernel (linear, polynomial of degree d, or radial with γ). Tune C together with d or γ by CV.
  - Why it works: you compute only the n(n−1)/2 values of K(xᵢ, xᵢ′) and never construct the enlarged feature space, which for the radial kernel is infinite-dimensional and implicit. Explicit feature expansion (adding Xⱼ² and so on) gives the same kind of boundary, but the computation blows up.
- **One-versus-one (all-pairs)**: for K > 2 classes, fit C(K,2) pairwise SVMs and assign the class that wins the most pairwise votes.
- **One-versus-all (one-versus-rest)**: fit K SVMs, each class against the rest. Assign x* to the class with the largest β₀ₖ + β₁ₖx₁* + … + β_pₖx_p*.
- **Support vector regression**: the regression analogue. Only residuals whose absolute value exceeds a positive constant contribute to the loss.

## Key Formulas
- Maximal margin: maximize M over β₀,…,β_p subject to Σⱼβⱼ² = 1 and yᵢ(β₀ + β₁xᵢ₁ + … + β_pxᵢₚ) ≥ M for all i. With the norm constraint, yᵢf(xᵢ) is the perpendicular distance to the hyperplane.
- Support vector classifier: the same, but with constraints ≥ M(1 − εᵢ), εᵢ ≥ 0, and Σᵢεᵢ ≤ C. Here C is the **budget** for violations; C = 0 gives the maximal margin classifier.
- Inner-product representation: f(x) = β₀ + Σᵢ αᵢ⟨x, xᵢ⟩ = β₀ + Σ_{i∈S} αᵢ⟨x, xᵢ⟩, with ⟨xᵢ, xᵢ′⟩ = Σⱼ xᵢⱼxᵢ′ⱼ.
- Linear kernel: K(xᵢ, xᵢ′) = Σⱼ xᵢⱼxᵢ′ⱼ.
- Polynomial kernel: K(xᵢ, xᵢ′) = (1 + Σⱼ xᵢⱼxᵢ′ⱼ)ᵈ. With d = 1 it is the support vector classifier.
- Radial kernel: K(xᵢ, xᵢ′) = exp(−γ Σⱼ (xᵢⱼ − xᵢ′ⱼ)²), γ > 0. It is very *local*: distant training points contribute almost nothing.
- SVM decision function: f(x) = β₀ + Σ_{i∈S} αᵢ K(x, xᵢ). Classify by sign(f(x)).
- Loss + penalty form: minimize over β of Σᵢ max[0, 1 − yᵢf(xᵢ)] + λΣⱼβⱼ². This is **hinge loss** plus a **ridge penalty**. A large λ corresponds to a large budget C (low variance, high bias).

## Key Concepts
- **Margin**: the smallest perpendicular distance from the training observations to the hyperplane.
- **Support vectors**: the observations that lie exactly on the margin, inside it, or on the wrong side. They alone determine the classifier.
- **Slack variables (εᵢ)**: they record where each observation sits relative to the margin and the hyperplane.
- **Budget C (book) vs. cost C (sklearn)**: in the book, C is the total tolerated violation, so a larger C means a wider margin. In sklearn `SVC(C=…)`, C is the *cost* of a violation, so a larger C means a *narrower* margin. The two are roughly inverses.
- **Kernel**: a function that quantifies the similarity between two observations and stands in for the inner product.
- **Hinge loss**: max[0, 1 − yf(x)]. It is exactly zero for observations beyond the margin on the correct side.
- **ROC curve via decision scores**: varying the threshold t on f̂(X) from 0 traces out the true-positive and false-positive rates.

## Mental Models
- **Think of the SVC as "logistic regression with hinge loss."** The two losses look alike (Fig. 9.12). When the classes are well separated, SVMs tend to do better. When the classes overlap substantially, logistic regression is often preferred and also gives probabilities.
- **Think of the radial kernel as a smooth, weighted nearest-neighbour vote.** γ sets the neighbourhood size: a larger γ means a more local, more flexible, higher-variance fit.
- **C, d and γ are bias-variance knobs, not nuisance parameters.** People once set C = 1 by default, but the loss + penalty view shows it controls underfitting versus overfitting.
- **Kernels are not unique to SVMs.** Logistic regression and other methods can be kernelized too; it is just conventional with SVMs. Explicit basis expansion (Ch 7) is the same idea done by hand.

## Anti-patterns
- **Using the maximal margin classifier with noisy or overlapping data**: it is either infeasible or overfit to a handful of points.
- **Picking γ or C from training ROC curves**: on Heart, γ = 10⁻¹ gave an almost perfect *training* ROC curve and the *worst* *test* ROC curve.
- **Confusing the book's C with sklearn's C**: the direction of the effect is reversed. A large sklearn C gives a narrow margin and few support vectors.
- **Expanding the features explicitly when a kernel would do**: the number of features explodes and the computation becomes unmanageable.
- **Treating the SVM as fundamentally different from classical methods**: it is a loss + penalty method. Compare it against logistic regression and LDA.

## Code Examples
```python
from sklearn.svm import SVC
import sklearn.model_selection as skm
svm_linear = SVC(C=10, kernel='linear').fit(X, y)
svm_linear.coef_                          # β for a linear kernel
kfold = skm.KFold(5, random_state=0, shuffle=True)
grid = skm.GridSearchCV(svm_linear, {'C':[0.001,0.01,0.1,1,5,10,100]},
                        refit=True, cv=kfold, scoring='accuracy')
grid.fit(X, y); grid.best_params_          # {'C': 1}
```
- **What it demonstrates**: fitting a support vector classifier and tuning its sklearn cost C by 5-fold CV. With C = 0.1 you get more support vectors and a wider margin than with C = 10.

```python
svm_rbf = SVC(kernel="rbf", gamma=1, C=1)
grid = skm.GridSearchCV(svm_rbf, {'C':[0.1,1,10,100,1000], 'gamma':[0.5,1,2,3,4]},
                        refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(grid.best_estimator_, X_test, y_test, name='Test')
```
- **What it demonstrates**: a joint CV grid over C and γ for the radial kernel, and an ROC curve built from `decision_function` scores. Use `kernel="poly"` with `degree=d` for polynomial kernels. For multiclass problems, `SVC` uses `decision_function_shape='ovo'` or `'ovr'`. The tuned radial SVM misclassified about 12% of the lab's test set. The printed best parameters and the book's prose disagree slightly (C=100, γ=1 vs. C=1, γ=0.5), because several settings tie.

## Reference Tables
| Classifier | Boundary | Handles overlap? | Tuning | Depends on |
|---|---|---|---|---|
| Maximal margin | linear | No (requires separability) | none | support vectors on the margin |
| Support vector classifier | linear | Yes (slack, budget C) | C | points on or inside the margin |
| SVM (polynomial) | degree-d polynomial | Yes | C, d | support vectors |
| SVM (radial) | local, highly non-linear | Yes | C, γ | support vectors (local influence) |

| | Support vector classifier | Logistic regression |
|---|---|---|
| Loss | hinge, max[0, 1 − yf] (exactly 0 beyond the margin) | log-loss (never exactly 0) |
| Penalty | ridge λΣβⱼ² | none by default (can add ridge) |
| Better when | classes well separated | classes overlap; probabilities needed |
| Outputs | class / score f(x) | probabilities |

| Book budget C | Margin | # support vectors | Bias | Variance |
|---|---|---|---|---|
| Large | wide | many | higher | lower |
| Small | narrow | few | lower | higher |

## Worked Example
**Heart data** (13 predictors). After removing 6 rows with missing values there were 297 subjects, split into 207 training and 90 test observations. LDA and the support vector classifier (equivalent to an SVM with a degree-1 polynomial kernel) both produce linear scores f̂(X). ROC curves come from varying the cutoff t. On the training set the SVC looks slightly better than LDA. For radial SVMs with γ = 10⁻³, 10⁻², 10⁻¹, the training ROC improves as γ grows, and γ = 10⁻¹ looks almost perfect. On the 90 **test** observations, the SVC keeps a small advantage over LDA, but the difference is not significant. γ = 10⁻¹ does the **worst**, while γ = 10⁻² and 10⁻³ match the SVC. The lesson: more flexibility lowers training error but does not necessarily lower test error.

**Lab (simulated, barely separable data)**: sklearn C = 1e5 gives zero training errors with only **3 support vectors**, a fragile fit. C = 0.1 also gives zero training errors but uses a wider margin with **12 support vectors**. It is more stable and is likely the better model on test data.

## Key Takeaways
1. The SVC is the maximal margin classifier with a violation budget C. Only support vectors matter, which makes it robust to points far from the boundary.
2. Choose C, together with d or γ, by cross-validation. Each one is a bias-variance knob.
3. The kernel trick gives non-linear boundaries using only pairwise K(xᵢ, xᵢ′) values. The radial kernel is local; the polynomial kernel is global.
4. The SVC equals hinge loss plus a ridge penalty, so it behaves very much like logistic regression. Prefer the SVM when the classes are well separated.
5. Extend to K classes with one-versus-one (C(K,2) votes) or one-versus-all (largest score).
6. Judge flexibility on test or CV error, never on training ROC curves.

## Connects To
- **Ch 4**: logistic regression, LDA and ROC curves are the comparison baselines.
- **Ch 6**: the ridge penalty and the loss + penalty framework.
- **Ch 7**: basis expansion and non-linear features, which kernels compute implicitly.
- **Ch 2**: KNN. The radial kernel's local behaviour resembles a nearest-neighbour vote.
- **Ch 10**: neural networks also handle non-linear boundaries. On the Khan gene data, linear SVMs work well when p ≫ n.
- **External**: Mercer kernels and reproducing kernel Hilbert spaces, kernel ridge and logistic regression, the SVM dual and the SMO solver.
