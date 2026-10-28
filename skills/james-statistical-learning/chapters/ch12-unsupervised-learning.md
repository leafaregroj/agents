# Chapter 12: Unsupervised Learning

## Core Idea
With only features X₁,…,X_p and no response Y, the goal is discovery, not prediction. You either find a low-dimensional representation that captures most of the variation (**principal components analysis**, which also powers **matrix completion**), or find homogeneous subgroups (**K-means** and **hierarchical clustering**). With no Y there is no CV, so results are subjective and need robustness checks.

## Frameworks Introduced
- **Principal Components Analysis (PCA)**: find orthogonal directions of maximal variance. The first PC is Z₁ = φ₁₁X₁ + … + φ_p₁X_p with Σⱼφⱼ₁² = 1 and the largest sample variance. Each later PC maximizes variance subject to being uncorrelated with (orthogonal to) the earlier ones.
  - When to use: visualization (plot the first 2–3 score vectors, biplots), data exploration, preprocessing for supervised learning (PCR, Ch 6), imputation.
  - How: center the columns (mean 0), usually **scale to SD 1** unless all variables share units. Compute via eigendecomposition or SVD (`X = U D Vᵀ`, where V holds the loadings and U·D the scores). Choose M by eyeballing the **scree plot** for an **elbow**, or, if PCs feed a supervised model, by CV.
  - Why it works: the first M PCs give the **best rank-M approximation** to X in squared error, and maximizing variance explained equals minimizing reconstruction error. Failure mode: unscaled variables with big variances (e.g. Assault in USArrests) dominate PC1. PCs capture only linear structure.
- **Matrix completion (Algorithm 12.1, "Hard-Impute")**: impute missing entries by iterating a low-rank PCA fit.
  - When to use: values are **missing at random** (not informative), and you need imputation or prediction. This is the basis of recommender systems (the Netflix customer × movie matrix).
  - How: (1) fill missing x̃ᵢⱼ with the observed column mean x̄ⱼ; (2) repeat: (a) compute the top-M PCs of X̃; (b) set each missing x̃ᵢⱼ ← Σₘ âᵢₘb̂ⱼₘ; (c) compute the objective on the observed entries, Σ_{(i,j)∈O}(xᵢⱼ − Σₘ âᵢₘb̂ⱼₘ)², until it stops decreasing; (3) return the imputed entries. Choose M by holding out extra observed entries as a validation set.
  - Failure mode: converges to a local optimum, not guaranteed global. Works only if the variables are correlated (low-rank structure).
- **K-means clustering (Algorithm 12.2)**: partition into exactly K non-overlapping clusters minimizing total within-cluster variation.
  - How: (1) randomly assign each observation to one of K clusters; (2) iterate until assignments stop changing: (a) compute each cluster **centroid** (vector of feature means); (b) reassign each observation to the nearest centroid (Euclidean).
  - Why it works: each step can only decrease the objective, but this finds a **local** optimum. Run from many random starts (`n_init` = 20 or 50) and keep the lowest objective. Set `random_state` for reproducibility. You must pre-specify K.
- **Hierarchical (agglomerative, bottom-up) clustering (Algorithm 12.3)**: build a **dendrogram** without pre-specifying K.
  - How: treat each of n observations as its own cluster, with all C(n,2) pairwise dissimilarities. For i = n, n−1, …, 2, fuse the two least dissimilar clusters. The fusion height records their dissimilarity. Recompute inter-cluster dissimilarities using a **linkage**. Cut the dendrogram horizontally at some height to get clusters.
  - Choices: **dissimilarity measure** (Euclidean, or **correlation-based distance**, which groups observations by profile shape rather than magnitude) and **linkage** (complete, average, single, centroid).
  - Failure mode: assumes a **hierarchical** structure. If the true groups are not nested (e.g. best split by sex into 2 groups vs. by nationality into 3), it can be less accurate than K-means for a given K.

## Key Formulas
- PC1 loadings: maximize over φ₁₁,…,φ_p₁ of (1/n) Σᵢ (Σⱼ φⱼ₁xᵢⱼ)² subject to Σⱼ φⱼ₁² = 1. The scores are zᵢ₁ = Σⱼ φⱼ₁xᵢⱼ.
- Best rank-M approximation: minimize over A (n×M), B (p×M) of Σⱼ Σᵢ (xᵢⱼ − Σₘ aᵢₘbⱼₘ)². The solution is âᵢₘ = zᵢₘ (scores), b̂ⱼₘ = φⱼₘ (loadings), so xᵢⱼ ≈ Σₘ zᵢₘφⱼₘ, exact when M = min(n−1, p).
- Matrix completion objective: the same sum restricted to observed pairs (i,j) ∈ O.
- PVE of PC m (centered data): PVE_m = Σᵢ zᵢₘ² / Σⱼ Σᵢ xᵢⱼ². There are min(n−1, p) PCs, and their PVEs sum to 1.
- Variance decomposition: (1/n)ΣⱼΣᵢ xᵢⱼ² = (1/n)ΣₘΣᵢ zᵢₘ² + (1/n)ΣⱼΣᵢ (xᵢⱼ − Σₘ zᵢₘφⱼₘ)², i.e. data variance = variance of the first M PCs + MSE of the M-dimensional approximation. So PVE = 1 − RSS/TSS, an R² for the approximation.
- K-means objective: minimize over C₁,…,C_K of Σₖ W(C_k), with W(C_k) = (1/|C_k|) Σ_{i,i′∈C_k} Σⱼ (xᵢⱼ − x_{i′j})² = 2 Σ_{i∈C_k} Σⱼ (xᵢⱼ − x̄ₖⱼ)². The identity explains why centroid updates reduce the objective.

## Key Concepts
- **Loading vector φₘ**: unit-norm weights defining the mth PC direction. Unique up to a **sign flip**.
- **Score vector zₘ**: the projection of each observation onto φₘ.
- **Biplot**: scores (points) and loadings (arrows) in one plot.
- **Scree plot**: PVE (and cumulative PVE) against component number. Look for an elbow. Choosing M is ad hoc in unsupervised use.
- **Dendrogram**: tree of fusions. Only the **vertical** fusion height measures similarity. Horizontal proximity of leaves is meaningless.
- **Linkage**: complete (max pairwise dissimilarity), single (min), average (mean), centroid (distance between centroids, which can produce **inversions**, fusions below either child's height).
- **Correlation-based distance**: treats observations as similar when their features are highly correlated, even if far apart in Euclidean terms (e.g. shoppers with similar preferences but different volumes).
- **Missing at random**: the missingness pattern is not informative. Required for matrix completion (not satisfied if, say, patients with high weight decline to report it).

## Mental Models
- Think of PCA as **the hyperplane closest to the data**. M = 1 is the line minimizing squared perpendicular distance, and the PVE is its R².
- Use **complete or average linkage** for balanced, interpretable dendrograms. **Single linkage** produces "trailing" clusters where leaves fuse one at a time.
- Use **K-means** when you can commit to K and clusters are roughly spherical. Use **hierarchical** to explore all K at once and when nested structure is plausible.
- **Small decisions, big consequences**: standardize or not, which dissimilarity, which linkage, where to cut, which K. Try several and report only the patterns that persist. Cluster subsets of the data to check robustness, since clustering is not robust to perturbations.

## Anti-patterns
- **Running PCA on unscaled variables measured in different units**: the variable with the largest variance dominates. In USArrests, unscaled PC1 is almost entirely Assault.
- **Reading leaf adjacency in a dendrogram as similarity**: only the fusion height matters.
- **Running K-means once**: you get one local optimum. Use many random starts.
- **Treating clusters as truth**: every observation is forced into a cluster, so outliers distort the result. Consider soft/mixture models, and treat clusters as hypotheses to validate on independent data.
- **Picking M for an unsupervised PCA by CV**: there's no response to validate against. CV applies only when PCs feed a supervised model (PCR).
- **Matrix completion when missingness is informative**: imputations will be biased.

## Code Examples
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler(with_std=True, with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)
pcaUS = PCA().fit(USArrests_scaled)
scores = pcaUS.transform(USArrests_scaled)      # z_im
pcaUS.components_                               # loadings (rows = PCs)
pcaUS.explained_variance_ratio_                 # PVE: ~[0.62, 0.247, 0.089, 0.043]

def low_rank(X, M=1):                           # Step 2(a) of Algorithm 12.1
    U, D, V = np.linalg.svd(X)
    L = U[:, :M] * D[None, :M]
    return L.dot(V[:M])
```
- **What it demonstrates**: standardized PCA, the PVE, and SVD as the engine behind PCA and Hard-Impute. In the lab's loop, `Xhat[ismiss] = low_rank(Xhat, M=1)[ismiss]` is iterated until the relative error falls below 1e-7.

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage
kmeans = KMeans(n_clusters=3, random_state=3, n_init=20).fit(X)
kmeans.labels_, kmeans.inertia_                  # inertia_ = within-cluster SS
hc_comp = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
                                  linkage='complete').fit(X)
linkage_comp = compute_linkage(hc_comp)
dendrogram(linkage_comp, color_threshold=4)
cut_tree(linkage_comp, n_clusters=4).T           # cluster labels at K=4
```
- **What it demonstrates**: K-means with multiple starts (in the lab, inertia is 78.06 with n_init=1 vs 75.04 with n_init=20), and the full dendrogram cut to K clusters. For correlation distance, pass `metric="precomputed"` with a distance matrix.

## Reference Tables

| Linkage | Inter-cluster dissimilarity | Typical behavior |
|---|---|---|
| Complete | Largest pairwise | Balanced, compact. Preferred. |
| Average | Mean pairwise | Balanced. Preferred. |
| Single | Smallest pairwise | Long trailing chains |
| Centroid | Distance between centroids | Can cause inversions (used in genomics) |

| | K-means | Hierarchical |
|---|---|---|
| K specified? | Yes, up front | No. Cut the dendrogram afterwards. |
| Output | Flat partition | Nested tree for all K |
| Randomness | Random starts, local optima | Deterministic |
| Assumption | Compact clusters near centroids | Nested hierarchical structure |

## Worked Example
**USArrests** (n = 50 states, p = 4: Murder, Assault, Rape per 100,000, and UrbanPop, all standardized).
- PC1 loadings ≈ (Murder 0.536, Assault 0.583, UrbanPop 0.278, Rape 0.543): roughly equal weight on the three crimes, so it is an **overall crime rate**. PC2 loads mainly on UrbanPop (≈0.873), so it measures **urbanization**. Sign conventions vary across software. The crime variables correlate with each other, and UrbanPop is less correlated with them.
- PVE: PC1 **62.0%**, PC2 **24.7%**, so the first two PCs explain ≈87%. The scree plot has an elbow after PC2. States with high PC1 scores (e.g. California, Nevada, Florida) have high crime.
- Matrix completion: 20 random states each had one of the 4 variables deleted (10% of entries missing), then Algorithm 12.1 was run with M = 1. Over 100 runs, the correlation between true and imputed values was **0.63 ± 0.11**, against an unattainable complete-data PCA benchmark of 0.79 ± 0.08.

**NCI60** (64 cancer cell lines × 6,830 genes): PCA score plots show lines of the same cancer type have similar scores. The first 7 PCs explain about 40% of the variance, with an elbow near PC7. Hierarchical clustering with complete linkage, cut at 4 clusters, puts the leukemia lines together. It differs from K-means with K = 4, though some clusters coincide.

## Key Takeaways
1. Scale variables before PCA or clustering unless they share meaningful units.
2. PCA gives both maximal-variance directions and the best low-rank approximation. PVE is the R² of that approximation.
3. Low-rank PCA fits impute missing-at-random data (Hard-Impute), the core of recommender systems.
4. K-means needs K and multiple random starts. Hierarchical clustering needs a dissimilarity, a linkage and a cut height.
5. Prefer complete or average linkage. Consider correlation-based distance when profile shape matters.
6. Unsupervised conclusions are fragile: vary the choices, cluster subsamples, and treat clusters as hypotheses.

## Connects To
- **Ch 6**: principal components regression uses PCs as supervised predictors. There, choose M by CV.
- **Ch 10**: autoencoders and embeddings as nonlinear analogues of PCA.
- **Ch 13**: clustering p-values and multiple testing issues in exploratory analysis.
- **Ch 2**: supervised vs unsupervised learning, curse of dimensionality.
- **External**: SVD, Soft-Impute / nuclear-norm matrix completion, Gaussian mixture models (soft clustering), gap statistic and silhouette for choosing K, t-SNE and UMAP.
