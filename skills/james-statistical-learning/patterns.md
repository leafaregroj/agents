# Patterns & Techniques — ISLP

## Resampling for Model Assessment & Selection
**When to use**: Any time you need test error or must pick a tuning parameter (Ch 5).
**How**: Use k-fold CV with k = 5 or 10 and the same `KFold` splits for every candidate model. Pick the minimum, or the simplest model within one SE of it (the one-standard-error rule), then refit on all the data. Use the bootstrap (B ≈ 1,000) for standard errors of arbitrary statistics.
**Trade-offs**: LOOCV is nearly unbiased but high-variance. The validation set approach is noisy and overestimates error. The bootstrap is wrong for prediction error, because samples overlap the original data.

## Subset Selection
**When to use**: Small to moderate p and a need for an interpretable model (Ch 6).
**How**: Best subset (2ᵖ fits) or forward/backward stepwise (1 + p(p+1)/2 fits). Choose the size by CV, or by Cp/AIC/BIC/adjusted R².
**Trade-offs**: Backward needs n > p. Stepwise isn't guaranteed to find the best model. BIC picks smaller models than Cp.

## Shrinkage: Ridge & Lasso
**When to use**: Many predictors, collinearity, or p ≈ n or p > n (Ch 6).
**How**: Standardize, then minimize RSS + λΣβⱼ² (ridge) or RSS + λΣ|βⱼ| (lasso). Choose λ by CV. sklearn: `ElasticNetCV(l1_ratio=0 or 1)` in a Pipeline with `StandardScaler`.
**Trade-offs**: The lasso gives sparse, interpretable models and wins when the truth is sparse. Ridge wins when many small effects exist. Neither is scale-equivariant.

## Dimension Reduction Regression (PCR / PLS)
**When to use**: Highly correlated predictors (Ch 6).
**How**: Regress Y on the first M principal components (PCR) or on PLS directions. Choose M by CV.
**Trade-offs**: PCR is unsupervised, so its directions may ignore Y. It isn't feature selection. PLS rarely beats ridge or PCR in practice.

## Basis Expansions: Polynomials, Splines, GAMs
**When to use**: A non-linear relationship that you still want to keep interpretable (Ch 7).
**How**: Polynomials of degree ≤ 4. Regression splines with knots at quantiles (prefer natural splines). Smoothing splines with λ by LOOCV. Local regression with span s. Combine terms additively in a GAM (pygam `LinearGAM`/`LogisticGAM`). Test nested models with ANOVA F-tests.
**Trade-offs**: Additivity misses interactions unless you add them. Local regression degrades in more than 3–4 dimensions.

## Tree Ensembles
**When to use**: Complex interactions, mixed predictor types, strong predictive accuracy (Ch 8).
**How**:
- Single tree: grow large, then cost-complexity prune with α chosen by CV.
- Bagging: B bootstrap trees, with OOB error for free.
- Random forest: m ≈ √p (classification) or p/3 (regression).
- Boosting: λ = 0.01–0.001, depth d = 1–4, B chosen by CV.
- BART: K = 200 trees, 1,000 iterations, 100 burn-in.
**Trade-offs**: Single trees are interpretable but high-variance. Ensembles lose interpretability, so use variable importance or partial dependence. Boosting can overfit with B; bagging and random forests can't.

## Support Vector Machines
**When to use**: Binary classification with well-separated classes or non-linear boundaries (Ch 9).
**How**: Fit a support vector classifier with budget C, or an SVM with polynomial (d) or radial (γ) kernel. Tune C and d/γ jointly by CV. For K > 2 classes, use one-vs-one or one-vs-all.
**Trade-offs**: SVMs don't give probabilities. With overlapping classes, logistic regression is similar and gives probabilities. **sklearn's `C` is a cost, the inverse of the book's budget.**

## Classification Model Choice
**When to use**: Choosing a classifier (Ch 4).
**How**:
- Logistic regression or LDA for linear boundaries.
- QDA or naive Bayes for moderately non-linear ones.
- KNN (standardized, K by CV) for highly non-linear ones.
- Pick the threshold from the costs of each error type, and evaluate with the confusion matrix and ROC/AUC.
**Trade-offs**: LDA is unstable when classes are well separated or n is small. QDA needs large n. Naive Bayes works when p is large relative to n.

## Neural Networks
**When to use**: Very large n, and image, text or sequence data (Ch 10).
**How**:
- Architecture: ReLU hidden layers; softmax output with cross-entropy loss.
- Training: minibatch SGD.
- Regularization: dropout, weight penalties, early stopping on a validation set, data augmentation.
- Images: CNNs, or pretrained networks (transfer learning).
- Sequences: RNNs or LSTMs, with embeddings for text.
**Trade-offs**: On small or tabular data, simpler models often match or beat networks (Occam's razor). Networks are hard to interpret.

## Survival Modeling
**When to use**: Time-to-event outcomes with censoring (Ch 11).
**How**: Kaplan–Meier curves, the log-rank test for groups, and Cox PH for covariates (lifelines `CoxPHFitter`). For high dimensions, use a penalized Cox model tuned by CV. Evaluate with the C-index.
**Trade-offs**: Cox assumes proportional hazards, so check it. It requires independent censoring.

## Unsupervised Structure
**When to use**: There is no response, and you want to explore, visualize or impute (Ch 12).
**How**:
- Scale the variables first.
- PCA: choose M from the scree-plot elbow, and visualize with a biplot.
- Matrix completion for missing data.
- K-means with n_init 20–50.
- Hierarchical clustering with complete or average linkage, then cut the dendrogram.
**Trade-offs**: There is no ground truth. Results depend on scaling, the dissimilarity measure, linkage and K, so report only clusters that are robust.

## Multiple Testing Control
**When to use**: Testing m hypotheses at once (Ch 13).
**How**: For FWER, use Holm, or Tukey for pairwise comparisons. For FDR, use Benjamini–Hochberg at q. Use permutation nulls (B ≈ 10,000) when the theoretical null is doubtful. statsmodels: `multipletests(p, method='holm'|'fdr_bh')`.
**Trade-offs**: FWER is very conservative for large m, while FDR finds more discoveries at the cost of some false ones.
