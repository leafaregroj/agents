# Chapter 6: The Universal Workflow of Machine Learning

## Core Idea
In the real world "you don't start from a dataset, you start from a problem." **The universal workflow of machine learning** is a step-by-step blueprint for taking any ML problem from a vague customer request to a maintained production system. It has three parts: **define the task**, **develop a model**, **deploy the model**. Model development, the part most tutorials and research cover, is only one step and not the hardest; framing problems and collecting, annotating, and cleaning data are. Chollet runs seven hypothetical consulting projects through every stage: photo search engine, chat-app spam/offensive-content filter, music recommender, credit card fraud detection, ad click-through-rate prediction, anomalous-cookie detection on a conveyor belt, and archeological-site discovery from satellite images.

## Frameworks Introduced
- **The universal workflow of machine learning**:
  1. *Define the task*: frame the problem, collect a dataset, understand your data, choose a measure of success.
  2. *Develop a model*: prepare the data, choose an evaluation protocol, beat a baseline, scale up until the model overfits, regularize and tune.
  3. *Deploy the model*: explain your work and set expectations, ship an inference model, monitor it in the wild, maintain it.
  - When to use: any new ML project, before writing model code.
  - How: walk the steps in order; treat the end of step 3 as the start of the next generation's step 1.
- **The two hypotheses**: (1) your targets can be predicted given your inputs; (2) the available data is sufficiently informative to learn that relationship.
  - When to use: at framing time and whenever you cannot beat a baseline.
  - How: treat both as unvalidated until a model achieves statistical power. If several reasonable architectures fail to beat a simple baseline, suspect the hypotheses, not the model (e.g. stock price history barely predicts stock movements).
- **Develop-a-model ladder** (from chapter 5, now formalized): achieve **statistical power** (a small model beats a baseline), then **scale up** (add layers, make layers bigger, train longer until validation degrades), then **regularize and tune**.
  - When to use: the model-development phase of every project.
  - How: at the statistical-power stage focus on three things: feature engineering, correct architecture priors, and a good-enough training configuration (loss, batch size, learning rate).
- **Deployment target selection**: REST API vs. on-device vs. in-browser, chosen by connectivity, latency, data sensitivity, and model size.

## Key Concepts
- **Framing questions**: What is the input and what are you predicting (data availability is usually the limiting factor)? What type of task is it (binary, multiclass, multilabel, scalar/vector regression, segmentation, ranking, clustering, generation, RL), or is ML even the right tool? What do existing solutions look like? What constraints apply (encryption, latency, on-device)?
- **Data collection**: the most arduous, time-consuming, costly part of most projects. If you have an extra 50 hours, collecting more data usually beats chasing modeling gains ("The Unreasonable Effectiveness of Data").
- **Data annotation infrastructure**: choose between self-labeling, crowdsourcing (cheap, scalable, noisy), or specialist labeling companies (save time, lose control). Ask whether labelers need subject-matter expertise, whether you can train them, and whether you understand how experts annotate (if not, you cannot do manual feature engineering). Invest early in good annotation software.
- **Non-representative data**: training data must resemble production data; the "cardinal sin". Collect from the deployment environment when possible; otherwise understand and actively correct for the differences.
- **Concept drift**: production data properties change over time, decaying accuracy. Acute with user-generated data and adversarial settings (fraud patterns change daily). Requires continuous collection, annotation, and retraining. Using past data to predict the future assumes the future behaves like the past.
- **Sampling bias**: data collection interacts with what you predict (the 1948 "DEWEY DEFEATS TRUMAN" phone-survey headline).
- **Understand your data**: look at raw samples and labels; histogram numerical features; map location data; count missing values; count instances per class to detect imbalance; check for **target leaking** (features that encode the target and will not exist in production, e.g. "has been diagnosed with cancer" when predicting future cancer treatment).
- **Measure of success**: the metric guides every technical choice and should align with business goals. Balanced classification: accuracy, ROC AUC. Imbalanced, ranking, or multilabel: precision and recall, weighted accuracy or ROC AUC. Custom metrics are common.
- **Data preparation**: vectorization (everything becomes float tensors), value normalization, handling missing values.
- **Proxy losses**: losses must be computable on a mini-batch (ideally a single point) and differentiable, so metrics like ROC AUC are optimized indirectly via crossentropy.
- **Inference**: using a model only to predict, which permits optimizations unavailable during training.
- **Weight pruning**: keep only the most significant weight coefficients; trades a little accuracy for smaller memory/compute.
- **Weight quantization**: float32 to int8 weights for an inference-only model a quarter the size, with near-original accuracy.
- **A/B testing**: route a random subset through the new model and a control subset through the old process to isolate the model's impact.

## Mental Models
- **Problem first, dataset second.** The book's datasets were pre-labeled gifts; real projects start with a business question and no data.
- **Technology is never neutral.** Technical choices are ethical choices. A model trained on biased labels (e.g. rating trustworthiness from faces) launders human prejudice into a black box with a veneer of objectivity.
- **You can only recognize what you've seen before.** ML memorizes patterns in training data; representativeness and drift follow directly.
- **Every model has an expiry date.** Weeks for a music recommender, days for fraud detection, a couple of years at best for image search. Start building the next generation at launch.
- **Validation is a finite resource.** Each tuning decision spends some of its reliability; systematic tuning eventually overfits the validation process.

## Anti-patterns
- Assuming that having inputs X and targets Y means X contains enough information to predict Y.
- Training on curated, professional data for a product that will see messy user data (the food-photo app: over 90% test accuracy, wrong 8 times out of 10 in production).
- Treating the dataset as a black box: skipping visualization, class counts, and leak checks.
- Imputing missing numerical values with an arbitrary constant like 0, which creates a discontinuity in the latent space.
- Training on data with no missing values when production data will have missing categorical features: the network never learns to ignore them.
- Feeding large-valued (multi-digit integers) or heterogeneous-range features straight into a network, triggering large gradient updates that prevent convergence.
- Reaching for deep learning by reflex: the music recommender is better served by matrix factorization; "anomaly detection" techniques are a poor fit for cookie flagging.
- Communicating "98% accuracy" to non-specialists (they round it to 100%) instead of false positive/negative rates in business terms.
- Shipping the exact Python training object to production, or publishing a model trained on sensitive data (training data can be partially recovered from weights).

## Code Examples
```python
import numpy as np

# Value normalization: x is a 2D matrix of shape (samples, features).
# Compute statistics on training data only, then reuse them on test data.
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Missing numerical values: impute with the feature's median, not 0.
col = x_train[:, 0]
x_train[np.isnan(col), 0] = np.nanmedian(col)

# Expecting missing categorical features at test time? Augment training data:
# duplicate some samples and blank out the features likely to be missing,
# encoding "missing" as its own category.
```
- **What it demonstrates**: the domain-agnostic preprocessing basics: per-feature zero mean / unit standard deviation, median imputation, and synthesizing missing-value examples. (The book shows only the two-line in-place normalization; the train-statistics detail follows chapter 4's house-price practice.)

## Reference Tables

**Last-layer activation and loss by problem type** (the book's table lists the three classification rows; the regression row is from chapter 4)

| Problem type | Last-layer activation | Loss |
|---|---|---|
| Binary classification | sigmoid | binary_crossentropy |
| Multiclass, single-label | softmax | categorical_crossentropy |
| Multiclass, multilabel | sigmoid | binary_crossentropy |
| Scalar regression | none | mse |

**Example projects framed**

| Project | Task type | Data source | Deployment |
|---|---|---|---|
| Photo search | Multiclass, multilabel classification (10,000 categories) | Hand-tag hundreds of thousands of user images | REST API |
| Spam/offensive filter | Binary (or three-way) classification | External social posts, hand-labeled (chats are end-to-end encrypted) | On-device / in-browser |
| Music recommender | Matrix factorization, not deep learning | Existing user likes | REST API |
| Credit card fraud | Binary classification | Historical records; fast concept drift | REST API |
| Ad click-through rate | Scalar regression | Years of past CTR logs | (not specified) |
| Anomalous cookies | Object detection stage + binary classification | New cameras; factory staff label images | On-device at the factory (GPU) |
| Archeological sites | Image-similarity ranking | Archeologists compile thousands of known sites | REST API |

**Choosing an evaluation protocol**

| Situation | Protocol |
|---|---|
| Plenty of data (most cases) | Holdout validation set |
| Too few samples for reliable holdout | K-fold cross-validation |
| Little data, need highly accurate estimate | Iterated K-fold with shuffling |

**Deployment options**

| Option | Use when | Tooling |
|---|---|---|
| REST API | Reliable internet; no strict latency (round trip ~500 ms); input data not highly sensitive (use SSL) | Flask, TensorFlow Serving, managed cloud (Cloud AI Platform) |
| On device | Strict latency or low connectivity; model can be made small enough; top accuracy not mission critical; data must not be decryptable on a server | TensorFlow Lite, Model Optimization Toolkit |
| In browser | Offload compute to users; data stays local; avoid ~100 ms network trip; offline after caching; model small and non-confidential | TensorFlow.js |

**Regularize-and-tune checklist**: try different architectures (add/remove layers); add dropout; add L1/L2 if the model is small; vary hyperparameters (units per layer, learning rate); iterate on data curation and feature engineering; consider automated tuning (KerasTuner, chapter 13).

## Worked Example
**Setting expectations for a fraud model.** Instead of "98% accuracy", say: "With these settings, the fraud model has a 5% false negative rate and a 2.5% false positive rate. Each day about 200 valid transactions get flagged for manual review, about 14 fraudulent transactions are missed, and about 266 are caught." (These imply roughly 280 frauds and 8,000 valid transactions daily.) Then decide the flagging probability threshold with stakeholders, because moving it trades false negatives against false positives, a business decision. Also show surprising failure cases so non-specialists do not assume the model "understands" or matches human performance.

**When the final test score disappoints.** After tuning, retrain on training plus validation data and evaluate once on the test set. If test performance is significantly worse than validation, either the validation procedure was unreliable or you overfit to it while tuning; switch to a more reliable protocol such as iterated K-fold validation.

## Key Takeaways
- Start from the problem: understand the business context, constraints, and existing solutions before touching data.
- Data availability and quality, not algorithms, usually decide success; invest in collection and annotation infrastructure.
- Ensure training data represents production data; plan for concept drift and sampling bias.
- Explore your data and check for target leaking before modeling.
- Pick a success metric aligned with business goals; optimize a differentiable proxy when needed.
- Prepare data so values are small and homogeneous; handle missing values deliberately.
- Model development: beat a baseline, overfit, then regularize and tune on validation data, with a single final test.
- Deploy with honest expectations, an optimized inference model on the right target, A/B tests and manual audits, and a plan for the next model generation.

## Connects To
- Chapter 4: the binary, multiclass, and regression examples supply the activation/loss choices and normalization practice.
- Chapter 5: evaluation protocols, baselines, information leaks, the fit-then-regularize sequence, and regularization techniques.
- Chapters 8-11: domain-specific preprocessing and architecture priors for images, timeseries, and text.
- Chapter 13: KerasTuner, mixed precision and other real-world best practices.
- Chapter 14: limits of interpolative models, which inform stakeholder expectations.
