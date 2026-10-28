---
name: james-statistical-learning
description: "Knowledge base from \"An Introduction to Statistical Learning, with Applications in Python\" (ISLP) by James, Witten, Hastie, Tibshirani & Taylor. Use when applying ISLP's methods for the bias-variance trade-off, cross-validation, regression/classification, regularization (ridge/lasso), splines/GAMs, tree ensembles, SVMs, deep learning, survival analysis, clustering/PCA, or multiple testing, studying the book, or referencing its concepts and Python labs."
---

<!-- argument-hint: [topic, method name, or chapter number] -->

# An Introduction to Statistical Learning, with Applications in Python (ISLP)
**Authors**: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor | **Pages**: ~613 | **Chapters**: 13 | **Generated**: 2026-10-28

## How to Use This Skill

- **Without arguments**: load the core frameworks below
- **With a topic**: e.g. `lasso`, `random forest`, `Cox model`, `FDR`. I find and read the relevant chapter
- **With a chapter**: e.g. `ch08`. I load that chapter, including its Python lab snippets
- **Browse**: ask "what chapters do you have?"

For a method not covered in Core Frameworks, read the relevant `chapters/` file before answering. For a quick choice or default, use [cheatsheet.md](cheatsheet.md).

---

## Core Frameworks & Mental Models

### 1. The statistical learning setup
Y = f(X) + ε. Estimate f for **prediction** (accuracy matters, f̂ may be a black box) or **inference** (understand how Y depends on each X; prefer restrictive, interpretable models). Error splits into **reducible** error (f̂ ≠ f) and **irreducible** error Var(ε), which no method can beat. Know whether the task is supervised or unsupervised, and regression (quantitative Y) or classification (qualitative Y), before choosing a method. (Ch 1, 2)

### 2. The bias–variance trade-off
E[(y₀ − f̂(x₀))²] = Var(f̂(x₀)) + [Bias(f̂(x₀))]² + Var(ε). As flexibility rises, bias falls and variance rises, so **test error is U-shaped** while training error keeps falling. Every method in the book is a dial on this curve: K in KNN, λ in ridge/lasso, df in splines, tree size and α, C and γ in SVMs, number of epochs in networks. **Never choose flexibility by training error.** (Ch 2)

### 3. Parametric vs. non-parametric
Parametric methods (linear, logistic) assume a form for f. They need little data and are interpretable, but are biased if the form is wrong. Non-parametric methods (KNN, splines, trees) are flexible but need many observations and suffer the **curse of dimensionality**: linear regression beats KNN once p ≳ 4 or f is near-linear. (Ch 2, 3)

### 4. Resampling estimates test error
- **Validation set approach**: simple, but noisy and overestimates error.
- **LOOCV**: nearly unbiased but high variance; exact one-fit shortcut for least squares.
- **k-fold CV (k = 5 or 10)** is the default. Use the same folds for every model and pick the minimum, or apply the **one-standard-error rule** for a simpler model, then refit on all data.
- **Bootstrap**: SEs for any statistic, B ≈ 1,000. Not for prediction error. (Ch 5)

### 5. Linear models, done carefully
- **Fit and assess**: least squares; assess with RSE, R², t-tests and the overall **F-test**, checking F first when p is large.
- **Predictors**: dummy variables for qualitative predictors; **interactions** follow the **hierarchical principle**.
- **Six potential problems**: non-linearity, correlated errors, heteroscedasticity, outliers (|studentized residual| > 3), high leverage, collinearity (VIF > 5–10).
- **Intervals**: a prediction interval is wider than a confidence interval. (Ch 3)

### 6. Classification
- **Logistic regression** models log-odds linearly and is fit by maximum likelihood.
- **Generative models** via Bayes' theorem:
  - **LDA** (shared covariance, linear boundary).
  - **QDA** (class covariances, quadratic boundary; needs large n).
  - **Naive Bayes** (independent features; good when p is large relative to n).
- **KNN**: standardize first, and tune K by CV.
- **Evaluation**: confusion matrix, sensitivity/specificity, ROC/AUC. **Move the threshold to match error costs** and compare to the null classifier when classes are imbalanced.
- **Counts**: use Poisson regression, which is a GLM. (Ch 4)

### 7. Model selection and regularization
- **Subset selection**: best subset or stepwise, judged by Cp/AIC/BIC/adjusted R² or CV.
- **Ridge** (ℓ₂) shrinks all coefficients. **Lasso** (ℓ₁) shrinks and selects, giving sparse models. Use the lasso when the truth is sparse and ridge when it is dense.
- **Always standardize, and choose λ by CV.**
- **PCR/PLS** regress on a few directions.
- **High dimensions (p > n)**: least squares interpolates, training R² is meaningless, and the selected model is one of many plausible ones. (Ch 6)

### 8. Moving beyond linearity
Use basis expansions:
- **Polynomials** (degree ≤ 4) and **step functions**.
- **Regression splines** with knots at quantiles; prefer **natural splines** for stable tails.
- **Smoothing splines** (penalty λ∫g''², effective df by LOOCV).
- **Local regression** (span s).
- **GAMs** combine these additively and keep interpretability, but interactions must be added by hand. (Ch 7)

### 9. Trees and ensembles
- **Single tree**: grow a large tree, then **cost-complexity prune** with α chosen by CV. Interpretable but high-variance.
- **Bagging**: averages bootstrap trees, with **OOB error** for free.
- **Random forests** decorrelate trees with m ≈ √p.
- **Boosting**: fits small trees (d = 1–4) to residuals with shrinkage λ = 0.01–0.001. Choose B by CV, because boosting can overfit while bagging and random forests can't.
- **BART**: perturbs and averages trees.
- **Interpretability**: use variable importance. (Ch 8)

### 10. Support vector machines
- **Maximal margin classifier**: needs separable classes.
- **Support vector classifier**: soft margin with budget C. Only the **support vectors** matter.
- **SVM**: uses **kernels** (polynomial d, radial γ). Tune C and γ/d together by CV.
- **Loss view**: hinge loss + ridge penalty, which makes SVMs close cousins of logistic regression. Prefer SVMs for well-separated classes and logistic regression when classes overlap.
- **sklearn caution**: `C` is a cost, the inverse of the book's budget. (Ch 9)

### 11. Deep learning
- **Layers**: ReLU hidden layers; softmax output with cross-entropy.
- **Training**: SGD with backpropagation.
- **Regularization**: dropout, weight penalties, early stopping, augmentation.
- **CNNs** for images (convolution + pooling, transfer learning). **RNNs/LSTMs** with embeddings for sequences and text.
- **When to use**: networks shine with huge n and structured inputs. On small or tabular data, **Occam's razor** applies: linear models or the lasso often tie.
- **Double descent** does not refute the bias–variance trade-off. (Ch 10)

### 12. Survival analysis
- **Censored data**: observe min(T, C). **Never drop censored observations**, and assume independent censoring.
- **Kaplan–Meier** for S(t). **Log-rank test** to compare groups.
- **Cox proportional hazards** (h₀(t)·e^{xᵀβ}, partial likelihood) to adjust for covariates. Check the proportional-hazards assumption.
- **High dimensions**: use a penalized Cox model.
- **Evaluation**: use the **C-index**. (Ch 11)

### 13. Unsupervised learning
- **PCA**: max-variance directions that are also the best low-rank approximation. **Standardize** first, choose M from the scree-plot elbow, and visualize with a biplot. **Matrix completion** imputes missing data.
- **K-means**: run with many random starts.
- **Hierarchical clustering**: dendrogram with complete or average linkage.
- **Caution**: no ground truth, so results depend on scaling, distance, linkage and K. Report only robust clusters. (Ch 12)

### 14. Multiple testing
Testing m hypotheses at level α gives FWER = 1 − (1−α)^m.
- **FWER control**: Bonferroni (α/m), **Holm** (always at least as powerful), **Tukey** (pairwise), Scheffé (contrasts).
- **FDR control**: **Benjamini–Hochberg** at q, for large exploratory m.
- **Re-sampling (permutation) nulls** when theory is doubtful.
- **Data snooping**: post hoc best-vs-worst comparisons need correction. (Ch 13)

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introduction.md) | Introduction | Supervised vs unsupervised; regression vs classification; notation |
| [ch02](chapters/ch02-statistical-learning.md) | Statistical Learning | Y = f(X)+ε; bias–variance trade-off; Bayes classifier; KNN |
| [ch03](chapters/ch03-linear-regression.md) | Linear Regression | Least squares inference; interactions; diagnostics; linear vs KNN |
| [ch04](chapters/ch04-classification.md) | Classification | Logistic; LDA/QDA/naive Bayes; KNN; ROC; Poisson GLM |
| [ch05](chapters/ch05-resampling-methods.md) | Resampling Methods | Validation set; LOOCV; k-fold CV; bootstrap |
| [ch06](chapters/ch06-model-selection-regularization.md) | Linear Model Selection and Regularization | Subset selection; Cp/BIC; ridge; lasso; PCR/PLS; p > n |
| [ch07](chapters/ch07-moving-beyond-linearity.md) | Moving Beyond Linearity | Polynomials; splines; smoothing splines; local regression; GAMs |
| [ch08](chapters/ch08-tree-based-methods.md) | Tree-Based Methods | Pruning; bagging; random forests; boosting; BART |
| [ch09](chapters/ch09-support-vector-machines.md) | Support Vector Machines | Maximal margin; soft margin; kernels; hinge loss |
| [ch10](chapters/ch10-deep-learning.md) | Deep Learning | MLPs; CNNs; RNN/LSTM; SGD & regularization; double descent |
| [ch11](chapters/ch11-survival-analysis.md) | Survival Analysis and Censored Data | Kaplan–Meier; log-rank; Cox PH; C-index |
| [ch12](chapters/ch12-unsupervised-learning.md) | Unsupervised Learning | PCA; matrix completion; K-means; hierarchical clustering |
| [ch13](chapters/ch13-multiple-testing.md) | Multiple Testing | FWER (Bonferroni, Holm, Tukey); FDR (BH); permutation nulls |

## Topic Index

- **AIC / BIC / Cp / adjusted R²** → ch06
- **Bagging / random forests / boosting / BART** → ch08
- **Bayes classifier / KNN** → ch02, ch04
- **Bias–variance trade-off / overfitting** → ch02, ch05, ch10
- **Bootstrap** → ch05, ch08
- **Censoring / Kaplan–Meier / Cox / C-index** → ch11
- **CNN / RNN / LSTM / transfer learning** → ch10
- **Collinearity / VIF / leverage / outliers** → ch03
- **Confusion matrix / ROC / AUC / threshold** → ch04, ch09
- **Cross-validation (LOOCV, k-fold) / one-SE rule** → ch05, ch06
- **Curse of dimensionality / high-dimensional data** → ch03, ch06
- **Decision trees / pruning** → ch08
- **Double descent** → ch10
- **FDR / FWER / Benjamini–Hochberg / Holm / Tukey** → ch13
- **GAMs / splines / local regression** → ch07
- **Interactions / dummy variables** → ch03
- **K-means / hierarchical clustering** → ch12
- **LDA / QDA / naive Bayes / logistic regression** → ch04
- **Matrix completion / missing data** → ch12
- **PCA / PCR / PLS** → ch06, ch12
- **Permutation tests** → ch13
- **Poisson regression / GLMs** → ch04
- **Ridge / lasso / elastic net** → ch06
- **SVM / kernels / hinge loss** → ch09
- **Subset / stepwise selection** → ch06

## Supporting Files

- [glossary.md](glossary.md): all key terms with definitions
- [patterns.md](patterns.md): method recipes, tuning, trade-offs
- [cheatsheet.md](cheatsheet.md): method choice, defaults, decision rules, warning signs

---

## Scope & Limits

This skill covers the book's content (ISLP, 1st ed., 2023). Lab snippets use the `ISLP`, statsmodels, scikit-learn, pygam, lifelines and PyTorch APIs as printed in the book; library versions may have changed since. Formulas were rebuilt from the PDF's text extraction, which mangled some math symbols.
