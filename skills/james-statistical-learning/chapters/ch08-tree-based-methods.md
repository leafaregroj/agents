# Chapter 8: Tree-Based Methods

## Core Idea
Decision trees split the predictor space into simple rectangular regions and predict the region's mean (regression) or majority class (classification). One tree is easy to interpret but high-variance and seldom very accurate. Ensembles of trees (bagging, random forests, boosting, BART) give up some interpretability and gain a lot of accuracy.

## Frameworks Introduced
- **Regression tree via recursive binary splitting**: a top-down, greedy partition into boxes R₁…R_J that minimizes RSS.
  - When to use: you need an interpretable model, the response depends on interactions or thresholds, or predictors are a mix of quantitative and qualitative.
  - How: at each step, try every predictor Xⱼ and every cutpoint s. Pick the split {X | Xⱼ < s} / {X | Xⱼ ≥ s} that reduces RSS the most. Repeat inside one of the current regions. Stop when, for example, no node has more than about 5 observations.
  - Failure mode: the greedy splits do not look ahead, and a fully grown tree overfits. A small change in the data can change the tree a lot (non-robust).
- **Cost complexity pruning (weakest link pruning)**, Algorithm 8.1:
  - How: (1) grow a large tree T₀. (2) For each α ≥ 0, find the subtree T ⊂ T₀ that minimizes RSS + α|T|. Subtrees are pruned in a nested, predictable order, so you get the whole sequence cheaply. (3) Choose α by K-fold CV, repeating steps 1–2 within each fold. (4) Return the subtree for that α, fit on the full data.
  - Why it works: it is a lasso-like penalty on the number of leaves. Growing large and then pruning avoids the short-sightedness of stopping early, where a weak split might have been followed by a strong one.
- **Classification tree**: predict the most common class in each region and report the class proportions p̂ₘₖ. Grow the tree using the Gini index or entropy. When the final goal is prediction accuracy, prune using the classification error rate.
- **Bagging (bootstrap aggregation)**:
  - How: draw B bootstrap samples and grow a deep, unpruned tree on each. Average the predictions (regression) or take a majority vote (classification).
  - Why: each deep tree has low bias but high variance. Averaging B estimates reduces variance, just as the variance of a mean is σ²/n.
  - Tuning: B is not critical and does not cause overfitting. Use a B large enough that the error has settled (about 100 was enough for Heart).
- **Out-of-bag (OOB) error estimation**: each bagged tree uses about 2/3 of the observations. Predict observation i using only the roughly B/3 trees for which it was out-of-bag. The OOB MSE or error rate is a valid test-error estimate and, for large B, is essentially LOOCV, with no refitting needed.
- **Random forests**: bagging, except that each split may use only m randomly chosen predictors, with a fresh sample at every split.
  - How: the usual choice is m ≈ √p for classification (4 of 13 for Heart). sklearn's regressor defaults to m = p, which is bagging. m = p reduces to bagging.
  - Why it works: when one predictor is strong, bagged trees all split on it first, so they are highly correlated and averaging them reduces variance little. Restricting m decorrelates the trees. A small m helps most when there are many correlated predictors.
- **Boosting (Algorithm 8.2)**: grow trees sequentially, each fit to the current residuals, and add a shrunken copy of each tree.
  - How: set f̂ = 0 and rᵢ = yᵢ. For b = 1…B, fit a tree f̂ᵇ with d splits to (X, r). Then update f̂ ← f̂ + λf̂ᵇ and rᵢ ← rᵢ − λf̂ᵇ(xᵢ). The output is f̂(x) = Σ λf̂ᵇ(x).
  - Tuning: **B**, chosen by CV, because boosting *can* overfit if B is too large, though it does so slowly. **λ** is the shrinkage, typically 0.01 or 0.001; a smaller λ needs a larger B. **d** is the *interaction depth*: d = 1 gives stumps and an additive model, and d splits involve at most d variables.
  - Why: it is a "learn slowly" method. Because each tree builds on the ones before it, small trees are enough.
- **Bayesian additive regression trees (BART)**, Algorithm 8.3: K trees, B iterations, and L burn-in iterations.
  - How: start every tree as a single root node with value ȳ/K. For each tree k, compute its partial residual (y minus the other K−1 trees). Then *randomly perturb* that tree's previous version: grow or prune branches, or change leaf values, favoring perturbations that fit better. Average the summed fits over iterations b = L+1…B. Percentiles of the draws give uncertainty bands.
  - Why: perturbing the previous tree instead of fitting a fresh one limits how hard each iteration fits, so BART resists overfitting. It is MCMC sampling from a posterior over tree ensembles.
  - Defaults: K = 200, B = 1,000, L = 100. It performs very well out of the box with little tuning.

## Key Formulas
- Tree RSS: Σⱼ₌₁ᴶ Σ_{i∈Rⱼ} (yᵢ − ŷ_{Rⱼ})², where ŷ_{Rⱼ} is the mean of the training responses in box j.
- Split search: minimize over (j, s) the quantity Σ_{i: xᵢ∈R₁(j,s)} (yᵢ − ŷ_{R₁})² + Σ_{i: xᵢ∈R₂(j,s)} (yᵢ − ŷ_{R₂})², with R₁ = {X | Xⱼ < s} and R₂ = {X | Xⱼ ≥ s}.
- Cost complexity: Σₘ₌₁^{|T|} Σ_{i: xᵢ∈Rₘ} (yᵢ − ŷ_{Rₘ})² + α|T|, where |T| is the number of terminal nodes. α = 0 gives T₀.
- Classification error: E = 1 − maxₖ p̂ₘₖ, where p̂ₘₖ is the proportion of class k in region m.
- Gini index: G = Σₖ p̂ₘₖ(1 − p̂ₘₖ). It is small when the node is pure.
- Entropy: D = −Σₖ p̂ₘₖ log p̂ₘₖ. It is numerically similar to Gini.
- Bagging: f̂_bag(x) = (1/B) Σ_b f̂^{*b}(x).
- Boosting: f̂(x) = Σ_b λ f̂ᵇ(x).
- BART partial residual: rᵢ = yᵢ − Σ_{k′<k} f̂ᵇ_{k′}(xᵢ) − Σ_{k′>k} f̂ᵇ⁻¹_{k′}(xᵢ). The estimate is f̂(x) = (1/(B−L)) Σ_{b=L+1}^{B} f̂ᵇ(x).

## Key Concepts
- **Terminal node / leaf; internal node; branch**: a leaf is a final region Rⱼ. An internal node is a split point. Branches are the segments that connect nodes. Trees are drawn upside down.
- **Node purity**: how strongly a single class dominates a node. Gini and entropy measure it and are more sensitive to it than the error rate.
- **Ensemble / weak learner**: an ensemble combines many simple, mediocre models (weak learners) into one strong predictor.
- **Out-of-bag observations**: the roughly 1/3 of the data that a given bootstrap tree did not see.
- **Variable importance**: the total decrease in RSS (regression) or Gini (classification) from splits on a predictor, averaged over all trees and usually shown relative to the maximum.
- **Decorrelating trees**: random forests limit the predictors available at each split so the trees differ and averaging them reduces more variance.
- **Interaction depth (d)**: the number of splits per boosted tree. It sets the order of interactions the model can capture.
- **Burn-in**: the early BART iterations (L) that are discarded because their fits are poor.
- **Qualitative predictors in trees**: a split can assign some levels to one branch and the rest to the other, with no dummy variables needed. sklearn does not do this; it treats one-hot columns as separate variables.

## Mental Models
- **Think of a single deep tree as a high-variance, low-bias estimator.** Bagging and random forests attack the variance by averaging. Boosting attacks the bias by slowly fitting residuals with small trees.
- **Choose a tree vs. a linear model based on the true decision boundary.** If the true boundary is roughly linear, a linear model will beat the axis-parallel steps of a tree. If it is highly non-linear or box-like, a tree wins (Fig. 8.7).
- **Pruning is the lasso for trees.** α|T| trades off fit against the number of leaves. Pick α by CV, just as you pick λ.
- **The number of trees B is a stopping rule, not a complexity knob, except in boosting.** Bagging and random forests cannot overfit in B; boosting can.

## Anti-patterns
- **Using early stopping with an RSS-reduction threshold**: it is short-sighted, because a useless split may come before a valuable one. Grow large and prune instead.
- **Growing splits with the classification error rate**: it is not sensitive enough to node purity. Use Gini or entropy to grow the tree.
- **Bagging when one predictor dominates**: the bagged trees come out nearly identical and correlated, so the variance reduction is small. Use a random forest with m < p.
- **Running boosting to a huge B with a large λ without CV**: boosting overfits. On Heart, its test error rises after a few hundred iterations while BART's stays flat.
- **Reporting training accuracy for a tree**: unpruned trees fit the training data almost perfectly. Use a test set, CV, or OOB error.
- **Expecting interpretability from an ensemble**: a forest has no single diagram. Use variable importance plots instead.

## Code Examples
```python
from sklearn.tree import DecisionTreeClassifier as DTC
import sklearn.model_selection as skm
clf = DTC(criterion='entropy', random_state=0)
clf.fit(X_train, High_train)
ccp_path = clf.cost_complexity_pruning_path(X_train, High_train)
kfold = skm.KFold(10, random_state=1, shuffle=True)
grid = skm.GridSearchCV(clf, {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, High_train)
best_ = grid.best_estimator_      # pruned tree (30 leaves on Carseats)
```
- **What it demonstrates**: cost complexity pruning in sklearn. The pruning path supplies the α grid (`ccp_alpha`), and CV chooses among those values.

```python
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GBR
bag_boston = RF(max_features=X_train.shape[1], random_state=0).fit(X_train, y_train)  # bagging: m = p
RF_boston  = RF(max_features=6, random_state=0).fit(X_train, y_train)                 # random forest
boost_boston = GBR(n_estimators=5000, learning_rate=0.001, max_depth=3, random_state=0)
boost_boston.fit(X_train, y_train)
for idx, y_ in enumerate(boost_boston.staged_predict(X_test)):   # test error vs. B
    test_error[idx] = np.mean((y_test - y_)**2)
```
- **What it demonstrates**: bagging is `RandomForestRegressor` with `max_features = p`. In `GBR`, λ is `learning_rate` and the tree depth is `max_depth`. `staged_predict` traces test error as B grows. BART is `ISLP.bart.BART(random_state=0, burnin=5, ndraw=15)`.

## Reference Tables
| Method | Samples data? | Trees grown | Key tuning | Overfits as B grows? |
|---|---|---|---|---|
| Single tree | — | 1, pruned | α (CV) | — |
| Bagging | bootstrap | independent, deep, unpruned | B | No |
| Random forest | bootstrap + random m predictors per split | independent, deep | m (≈√p), B | No |
| Boosting | original data | sequential, on residuals, small | B (CV), λ (0.01/0.001), d (often 1) | Yes, slowly |
| BART | original data | sequential, perturb previous tree | K = 200, B = 1000, L = 100 | Largely no |

| Trees: pros | Trees: cons |
|---|---|
| Very easy to explain, arguably closer to human decision-making | Lower predictive accuracy than other methods |
| Graphical display | Non-robust: a small change in the data gives a very different tree |
| Handle qualitative predictors without dummies | Axis-parallel splits approximate linear boundaries poorly |

Boston lab test MSE (30% test split): pruned tree 28.07; bagging 14.63; random forest (m = 6) 20.04; boosting (λ = 0.001, depth 3, B = 5000) 14.48; boosting with λ = 0.2 gave 14.50; BART 20.92. The top random-forest importances were lstat and rm.

## Worked Example
**Hitters salary tree** (log Salary on Years and Hits). The top split is Years < 4.5, which predicts a mean log salary of 5.107, or $165,174. For players with Years ≥ 4.5, the next split is Hits < 117.5, giving 5.999 ($402,834); Hits ≥ 117.5 gives 6.740 ($845,346). So Years matters most, and Hits matters only for veterans. In the pruning study, the 263 players were split into 132 training and 131 test observations. A large tree was grown on 9 features, and 6-fold CV (132 is divisible by 6) chose the **3-leaf tree**. Test error also dips at 3 leaves, though its minimum is at 10.
**Gene-expression random forest** (15 classes, the 500 highest-variance genes): a single tree had 45.7% error against a 75.4% null rate. Random forests with m = √p improved slightly on bagging (m = p). In the boosting comparison (cancer vs. normal), depth-1 stumps with λ = 0.01 beat depth-2 trees and the random forest, though the standard errors were about 0.02. A single tree's error was 24%.

## Key Takeaways
1. Grow a large tree, prune it with cost complexity α|T|, and choose α by K-fold CV.
2. Grow classification trees with Gini or entropy. If accuracy is the goal, prune with the error rate.
3. Bagging reduces variance by averaging deep bootstrap trees. Its OOB error is a free, approximately LOOCV estimate of test error.
4. Random forests decorrelate the trees with m ≈ √p predictors per split. Neither bagging nor random forests overfit as B grows.
5. Boosting learns slowly from residuals and has three knobs (B, λ, d). Choose B by CV, because it can overfit.
6. BART perturbs the trees of an MCMC ensemble. It does well with defaults and gives uncertainty for free.
7. Ensembles trade interpretability for accuracy. Variable importance gives back some of the interpretability.

## Connects To
- **Ch 5**: the bootstrap underlies bagging, and OOB error relates to LOOCV.
- **Ch 6**: the cost complexity penalty is analogous to the lasso penalty. Prediction accuracy is compared with regularized linear models.
- **Ch 7**: boosting with stumps produces an additive model, like GAMs. BART's backfitting-style partial residuals resemble the GAM backfitting algorithm.
- **Ch 2**: the bias-variance trade-off is the lens for comparing single trees with ensembles.
- **External**: gradient boosting (XGBoost/LightGBM), AdaBoost, MCMC.
