# Chapter 1: Introduction

## Core Idea
Statistical learning is a toolkit for understanding data, split into **supervised** learning (predict or estimate an output from inputs) and **unsupervised** learning (find structure among inputs with no output). The book aims to make these tools usable by non-specialists: concepts and practical use over optimization theory.

## Frameworks Introduced
- **Supervised vs. unsupervised learning**: supervised problems pair inputs X with an output Y. Unsupervised problems have only X.
  - When to use: if you have a labeled outcome you care about, use supervised learning. If you only want groups or low-dimensional structure, use unsupervised learning.
  - How: supervised learning splits further by output type (below). Unsupervised tasks include clustering and dimension reduction (e.g. principal components).
- **Regression vs. classification**: a **quantitative** (continuous) output is a *regression problem*. A **qualitative** (categorical) output is a *classification problem*.
  - When to use: the output's type decides it, not the inputs'.
- **Book premises** (the authors' philosophy):
  1. Many statistical learning methods are relevant in a broad range of settings, not just in statistics.
  2. Statistical learning should not be treated as a series of black boxes. Know the model, its intuition, and its assumptions.
  3. It is important to know the details of each method, but not necessarily how to code the optimization by hand.
  4. Readers care about real-world problems, so each chapter ends with a Python lab.

## Key Formulas
Notation (shared with ESL):
- `n` = number of observations, `p` = number of variables (predictors).
- `xᵢⱼ` = value of variable j for observation i. **X** is the n × p data matrix whose rows are `xᵢ = (xᵢ₁, …, xᵢₚ)ᵀ` and whose columns are `𝐱ⱼ = (x₁ⱼ, …, xₙⱼ)ᵀ`.
- `yᵢ` = the i-th response. Observed data = `{(x₁,y₁), …, (xₙ,yₙ)}`.
- Vectors of length n are bold lowercase. Vectors of length p and scalars are normal font. Matrices are bold capitals. `a ∈ ℝ`, `a ∈ ℝᵏ`, and `A ∈ ℝ^(r×s)` denote a scalar, a vector, and a matrix.
- Matrix product: `(AB)ᵢⱼ = Σₖ aᵢₖ bₖⱼ`. Requires that the number of columns of A equals the number of rows of B.

## Key Concepts
- **Statistical learning**: a vast set of tools for modeling and understanding complex data sets. It overlaps with statistics and machine learning.
- **Supervised learning**: building a model that predicts or estimates an output from one or more inputs.
- **Unsupervised learning**: learning relationships and structure from inputs alone, with no supervising output.
- **Regression problem**: predicting a continuous or quantitative output (e.g. wage).
- **Classification problem**: predicting a categorical or qualitative output (e.g. market Up/Down).
- **Clustering**: grouping observations by similarity of their measured features.
- **Principal components (Z₁, Z₂)**: low-dimensional summaries of high-dimensional data, used here to visualize 6,830 genes in 2D.
- **Generalized linear model**: the 1970s umbrella that contains linear and logistic regression as special cases.

## Mental Models
- **Classify the problem by its output first.** Is there a Y? Is it numeric or categorical? That choice routes you to Ch 3/7/8 (regression), Ch 4/8/9 (classification), or Ch 12 (unsupervised).
- **Weak signal is still signal.** In the Smarket data, lagged returns barely differ between Up and Down days, yet QDA reaches about 60% accuracy on 2005. Expect small effects in efficient-market-type data.
- **Validate clusters externally when you can.** NCI60 clusters found without labels roughly match the 14 cancer types, which gives independent support.
- **The history is a flexibility timeline.** Methods went from linear (least squares in the early 1800s, LDA in 1936, logistic regression in the 1940s, GLMs in the 1970s) to non-linear once computing allowed it (CART and GAMs in the mid-1980s, neural networks in the 1980s, SVMs in the 1990s).

## Anti-patterns
- **Treating methods as black boxes**: you cannot judge when a method will fail without knowing its assumptions and trade-offs.
- **Predicting from one predictor when several matter**: age alone predicts wage poorly. Combining age, education and year does much better.

## Code Examples
No lab in this chapter. All lab data sets load through the `ISLP` package:
```python
from ISLP import load_data
Wage = load_data('Wage')
```
- **What it demonstrates**: the book's standard data-access pattern. USArrests is the exception and comes from base R (see Section 12.5.1).

## Reference Tables
| Data set | Type | Task | Chapter |
|---|---|---|---|
| Wage | Quantitative Y | Regression: wage vs age (non-linear, peaks near 60), year (about +$10k from 2003 to 2009), education | 3, 7 |
| Smarket | Qualitative Y | Classification: S&P daily Up/Down from 5 lagged returns, 2001–2005 | 4 |
| NCI60 | No Y | Clustering and PCA: 6,830 genes × 64 cell lines | 12 |
| Advertising, Auto, Boston, Carseats, Default, Caravan, Credit, Hitters, Khan, College, OJ, Portfolio, Weekly, Bikeshare, BrainCancer, Publication, Fund, NYSE… | various | labs and exercises | 2–13 |

## Worked Example
**Smarket**: of 1,250 days, the market rose on 648 and fell on 602. Box plots of the previous 1, 2 and 3 days' returns look almost identical for Up and Down days. Fitting QDA on 2001–2004 and predicting 2005, the mean predicted probability of a decrease is slightly higher on days that actually fell (all values sit in about 0.46–0.52). The model is right about 60% of the time.

## Key Takeaways
1. Decide supervised vs. unsupervised, then regression vs. classification, from the output alone.
2. n = observations, p = predictors, and X is n × p. This notation is used throughout the book.
3. ISL is the accessible companion to ESL. It emphasizes intuition, assumptions and applied Python (`ISLP` package).
4. Flexible non-linear methods became practical only with modern computing. Linear methods are still the interpretable baseline.

## Connects To
- **Ch 2**: formalizes Y = f(X) + ε, prediction vs. inference, and the bias-variance trade-off.
- **Ch 3 / Ch 7**: linear and non-linear regression for Wage.
- **Ch 4**: classification for Smarket (logistic regression, LDA, QDA, KNN, naive Bayes).
- **Ch 12**: PCA and clustering for NCI60.
- **ESL (Hastie, Tibshirani, Friedman)**: the deeper, more technical reference.
