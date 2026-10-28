# Chapter 5: Fundamentals of Machine Learning

## Core Idea
The fundamental issue in machine learning is the **tension between optimization and generalization**. Optimization fits the model to training data (the part you control); generalization is performance on never-before-seen data (the part you actually care about). Push optimization too far and overfitting sets in. Deep learning generalizes not because of magic in the model but because natural data lies on low-dimensional, structured manifolds (the **manifold hypothesis**), and a smooth model fitted gradually by gradient descent passes through a state that approximates that manifold, enabling **interpolation**. Everything else in the chapter, from evaluation protocols to regularization, is machinery for managing that tension: first get a model that can beat a baseline and can overfit, then fight overfitting.

## Frameworks Introduced
- **Underfitting / robust fit / overfitting curve**: early in training, training and validation loss fall together (underfit); validation then stalls and degrades while training loss keeps falling (overfit). The robust fit sits at the turning point.
  - When to use: reading every training run's loss curves.
  - How: always plot training vs. validation loss/metrics per epoch; the epoch where validation peaks is your target.
- **Manifold hypothesis**: all natural data lies on a low-dimensional manifold within the high-dimensional space where it is encoded.
  - When to use: deciding whether deep learning can work on a problem at all.
  - How: ask whether samples can be smoothly morphed into each other (continuous, connected). If the problem is noisy or fundamentally discrete (e.g. list sorting), deep learning will not help.
- **Interpolation as a source of generalization**: a model makes sense of new points by relating them to nearby training points on the learned latent manifold. This yields **local generalization**; humans add **extreme generalization** via abstraction and reasoning, which models lack.
  - When to use: setting expectations for what a model can and cannot do.
  - How: make interpolation easy: dense sampling of the input space, especially near decision boundaries.
- **Evaluation recipes**: simple holdout validation, K-fold validation, iterated K-fold validation with shuffling.
  - When to use: holdout with plenty of data; K-fold when scores vary a lot across splits; iterated K-fold when data is scarce and you need precise estimates (costs P x K trainings).
  - How: always keep a separate test set on top of whatever validation scheme you use.
- **Two-phase model improvement**: Phase 1 "improve model fit" (get a model with some generalization power that can overfit); Phase 2 "improve generalization" (regularize). "To achieve the perfect fit, you must first overfit."
  - When to use: structuring every model-development effort.
  - How: diagnose which of three fitting problems you have (below), fix it, then apply generalization levers.

## Key Concepts
- **Optimization vs. generalization**: fitting training data vs. performing on new data; you only directly control the former.
- **Sources of overfitting**: noisy training data (invalid inputs, mislabeled samples); ambiguous features (inherent uncertainty, e.g. banana ripeness, weather); rare features and spurious correlations (e.g. the word "cherimoya" appearing once in a negative review, or a 54/46 split over 100 samples that is pure fluke).
- **Feature selection**: score each feature's usefulness (e.g. mutual information with labels) and keep those above a threshold; restricting IMDB to the top 10,000 words is a crude version.
- **Memorization capacity**: a large enough model can fit shuffled labels or pure noise, acting like a Python dict. Training loss falling proves nothing about generalization.
- **Latent manifold**: lower-dimensional subspace, locally Euclidean, on which valid samples live. Interpolating on the manifold differs from linear interpolation in input space (the pixel average of two digits is not a digit).
- **Why deep learning works**: models are smooth, continuous (differentiable) high-dimensional curves; gradient descent morphs them gradually, so an intermediate state approximates the data manifold. Architecture priors (convnets, sequence models, hierarchical/modular representations) mirror the shape of the data.
- **Training data is paramount**: generalization depends more on data structure than on model properties. "The only thing you will find in a deep learning model is what you put into it": architecture priors plus training data.
- **Dense sampling**: training data should densely cover the input manifold, especially near decision boundaries.
- **Regularization**: actively impeding the model's ability to fit training data perfectly so it approximates the latent manifold more closely.
- **Hyperparameters vs. parameters**: configuration (layer count/size) vs. learned weights. Tuning hyperparameters is itself learning, which overfits the validation set.
- **Information leaks**: each validation-driven tuning decision leaks bits about the validation set into the model; many iterations make validation scores artificially good.
- **Common-sense baseline**: performance of a random classifier or the simplest non-ML method; the altitude meter for the "invisible rocket".
- **Overparameterization and early stopping**: deep models always have far more capacity than needed; you never train to minimum training loss, you interrupt at the best validation point.
- **Weight regularization**: L1 (cost proportional to |w|) and L2 (cost proportional to w^2, also called **weight decay**). Penalty applies only at training time, so training loss reads higher than test loss.
- **Dropout**: randomly zero a fraction (the dropout rate) of a layer's outputs during training; in practice rescale up at training time and leave outputs untouched at test time. Breaks up happenstance "conspiracies" between units (Hinton's bank-teller analogy).

## Mental Models
- **Deep learning is curve fitting, not magic.** A deep model is a smooth curve fitted to points; it can only interpolate between what it has seen.
- **Uncrumpling a paper ball.** The model disentangles a crumpled latent manifold; regularization keeps the uncrumpling smooth.
- **The invisible rocket.** Training happens in a space you cannot see; validation metrics are your only altitude meter, and the baseline tells you whether you left the ground.
- **Limited memory forces compression.** A model that can memorize only a few, regular patterns must focus on the most prominent ones, which generalize best.
- **Overfitting is guaranteed; underfitting is always fixable.** You can always get training loss down (you can fit random labels) and you can always overfit (add capacity). Failure to generalize at all is the only truly hard problem.

## Anti-patterns
- Evaluating on training data, or tuning on the test set. Any test-set-driven decision invalidates the generalization estimate.
- Splitting class-ordered data without shuffling (training gets only classes 0-7, test only 8-9).
- Shuffling time-ordered data before splitting: creates a **temporal leak** (training on the future). Test data must be posterior to training data.
- Duplicate samples spanning training and validation sets: you end up testing on training data.
- Reporting accuracy without a baseline: 0.9 accuracy is worthless when 90% of samples are class A.
- Assuming more noise features are harmless because they add no information: noise channels lowered MNIST validation accuracy by about one point purely through spurious correlations.
- Using L2 regularization on very large models and expecting much: heavily overparameterized models barely notice; use dropout instead.
- Reading choppy validation curves as model behavior when the validation split is simply too small.

## Code Examples
```python
# Diagnose: deliberately overfittable model on MNIST with a sane learning rate.
# A learning rate of 1.0 stalls at 30-40% accuracy; 1e-2 trains fine.
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(96, activation="relu"),
    layers.Dense(96, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.RMSprop(1e-2),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(train_images, train_labels,
                    epochs=20, batch_size=128, validation_split=0.2)
```
- **What it demonstrates**: fixing a stalled run by tuning the learning rate, and giving the model enough capacity (two 96-unit layers vs. a logistic regression that plateaus near 0.26 val loss) so it overfits after about 8 epochs.

```python
import numpy as np

k = 3
num_val = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_val * fold: num_val * (fold + 1)]
    training_data = np.concatenate([data[:num_val * fold],
                                    data[num_val * (fold + 1):]])
    model = get_model()                 # fresh, untrained model each fold
    model.fit(training_data, ...)
    validation_scores.append(model.evaluate(validation_data, ...))
validation_score = np.average(validation_scores)

model = get_model()                     # final model on all non-test data
model.fit(data, ...)
test_score = model.evaluate(test_data, ...)
```
- **What it demonstrates**: K-fold validation (labels omitted for brevity), averaging fold scores, then retraining on all non-test data before a single test evaluation.

```python
from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])
# Alternatives: regularizers.l1(0.001), regularizers.l1_l2(l1=0.001, l2=0.001)
```
- **What it demonstrates**: the two standard regularizers. `l2(0.002)` adds `0.002 * w**2` per weight coefficient to the loss; `Dropout(0.5)` drops half of the preceding layer's outputs during training. (The book applies them separately on IMDB; combined here for compactness. Dropout alone gave the best minimum validation loss.)

## Reference Tables

**Diagnosing fitting problems (Phase 1)**

| Symptom | Cause | Fix |
|---|---|---|
| Training loss does not go down / stalls early | Gradient descent configuration | Tune learning rate (lower if overshooting, higher if crawling) and increase batch size; keep other settings fixed |
| Trains but cannot beat the common-sense baseline | Inputs lack the information, or wrong architecture | Check whether the problem is solvable; use better architecture priors (e.g. recurrent vs. dense for timeseries); read prior art |
| Train and val loss both fall, but no overfitting even after many epochs | Insufficient capacity | Add layers, make layers bigger, use more appropriate layer types |

**Improving generalization (Phase 2), in rough priority**

| Lever | Notes |
|---|---|
| More or better training data | Number one lever; densify coverage, fix labels |
| Better features (feature engineering, feature selection) | Fewer resources, far less data needed |
| Early stopping | Stop at the best validation epoch (`EarlyStopping` callback, chapter 7) |
| Reduce capacity | Start small, grow until validation loss shows diminishing returns |
| Weight regularization (L1/L2) | Mostly for smaller models |
| Dropout (rate 0.2-0.5) | Most effective general-purpose regularizer for neural nets |

**Common-sense baselines from earlier examples**

| Task | Baseline accuracy |
|---|---|
| MNIST (10 balanced classes) | 0.1 |
| IMDB (binary, balanced) | 0.5 |
| Reuters (46 imbalanced classes) | about 0.18-0.19 |
| Binary, 90/10 imbalance | 0.9 (always predict majority) |

**Capacity effects on IMDB (16-unit reference model)**

| Model | Starts overfitting | Behavior |
|---|---|---|
| 4-unit layers | after ~6 epochs | Degrades more slowly |
| 16-unit layers (reference) | after ~4 epochs | Baseline |
| 512-unit layers | after ~1 epoch | Severe, noisy val loss, near-zero train loss |

## Worked Example
**Feature engineering: reading a clock.** Raw pixels require a convnet and substantial compute. Extracting (x, y) coordinates of each hand's tip (a short script following dark pixels) lets a simple model learn the mapping. Converting to polar angles (theta of each hand) makes the problem so easy that rounding plus a dictionary lookup replaces machine learning entirely. The lesson: feature engineering makes the latent manifold smoother and simpler, and requires understanding the problem. Deep learning removes most of the need for it, but good features still (1) solve problems more elegantly with fewer resources and (2) let you succeed with far less data.

## Key Takeaways
- The goal is generalization; overfitting happens in every ML problem.
- Deep models generalize by interpolating on a learned approximation of the data's latent manifold; this is local generalization only.
- Data quality and density matter more than model cleverness. Spending on data collection almost always beats spending on modeling.
- Use three splits: train, validation, test. Validation tuning leaks information, so the test set must stay untouched until the end.
- Choose holdout, K-fold, or iterated K-fold based on data volume and score variance; watch representativeness, the arrow of time, and redundancy.
- Always set a common-sense baseline before training.
- First reach a model that beats the baseline and can overfit; only then regularize.
- Regularization toolbox: more/better data, better features, smaller capacity, L1/L2 (small models), dropout, early stopping.

## Connects To
- Chapter 2: the crumpled-paper-ball metaphor and gradient descent mechanics underpin the manifold view.
- Chapter 4: the IMDB, Reuters, and house-price examples supply the baselines and models regularized here.
- Chapter 6: the universal workflow operationalizes this chapter's two-phase process into a full project blueprint.
- Chapter 7: callbacks, including `EarlyStopping` and `ModelCheckpoint`.
- Chapters 8-10: architecture priors for images and timeseries (dense nets fail to beat baselines on timeseries).
- Chapter 13: automated hyperparameter tuning with KerasTuner.
- Chapter 14: extreme generalization, abstraction, and the limits of interpolation.
