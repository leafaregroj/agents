# Chapter 4: Getting Started with Neural Networks: Classification and Regression

## Core Idea
The three most common tasks on vector data are binary classification, single-label multiclass classification, and scalar regression. All three can be solved with stacks of `Dense` layers with `relu` activations. What changes between tasks is the **last layer, its activation, and the loss**. Each example follows the same end-to-end workflow. Vectorize the data, build a model, hold out validation data, watch training and validation curves to find where overfitting begins, retrain from scratch for that many epochs, evaluate on the test set, and compare against a baseline.

## Frameworks Introduced
- **Task-to-output-to-loss mapping**: the last-layer configuration is determined by the task.
  - When to use: every time you set up a new supervised problem.
  - How: binary → `Dense(1, "sigmoid")` + `binary_crossentropy`; single-label N-class → `Dense(N, "softmax")` + `categorical_crossentropy` (one-hot labels) or `sparse_categorical_crossentropy` (integer labels); scalar regression → `Dense(1)` with no activation + `mse` loss, `mae` metric.
- **Train → spot overfitting → retrain from scratch**: train for many epochs with `validation_data`, plot the `History` curves, find the epoch where validation metrics peak, then train a fresh model on all training data for that many epochs and evaluate once on the test set.
  - When to use: as the basic tuning loop before learning more formal methods (chapter 5).
  - How: IMDB peaked at about epoch 4, Reuters at about 9, and Boston housing at about 120–140 epochs.
- **K-fold cross-validation**: split the data into K partitions (typically K = 4 or 5). Train K identical fresh models, each on K-1 partitions, validate on the held-out partition, and average the K scores.
  - When to use: when data is so scarce that a single validation split would be tiny and its score high-variance with respect to the split.
  - How: use a `build_model()` factory so each fold gets a fresh compiled model. Average per-epoch validation histories across folds to choose the epoch count.
- **Information bottleneck**: each `Dense` layer sees only the previous layer's output, so information dropped by a too-narrow layer is lost for good.
  - When to use: sizing intermediate layers when the output space is large.
  - How: avoid intermediate layers much smaller than the number of classes. With 46 classes, 64-unit layers work, while a 4-unit layer drops validation accuracy from about 80% to about 71%.
- **Random baseline check**: estimate chance accuracy by shuffling test labels and measuring agreement. For Reuters this gives about 19%, which makes the model's ~80% meaningful.

## Key Concepts
- **Glossary**: *sample/input* (one data point fed to the model); *prediction/output*; *target* (the truth); *prediction error/loss value*; *classes* (the possible labels); *label* (a specific class annotation for one sample); *ground-truth/annotations* (all of a dataset's targets, usually human-collected); *binary*, *multiclass*, and *multilabel* classification (in multilabel, a sample can carry several labels); *scalar regression* (continuous scalar target); *vector regression* (continuous vector target, e.g., a bounding box); *mini-batch* (typically 8–128 samples, often a power of 2 for GPU memory allocation).
- **Multi-hot encoding**: turn a variable-length list of word indices into a fixed 10,000-dim 0/1 vector so a `Dense` layer can consume it. The alternative is padding to an integer tensor and using an `Embedding` layer (covered later).
- **`num_words=10000`**: keep only the top 10,000 most frequent words. IMDB has 88,585 unique words, many appearing only once and therefore useless for classification. Word indices are offset by 3 because 0, 1, 2 are reserved for padding, start of sequence, and unknown.
- **One-hot / categorical encoding**: an all-zero vector with a 1 at the label index (`to_categorical`). Integer labels plus `sparse_categorical_crossentropy` is mathematically the same loss with a different interface.
- **Units = dimensionality of the representation space**: "how much freedom you allow the model" when learning internal representations. More units allow more complex representations but cost more compute and risk learning patterns that do not generalize.
- **Activation functions (non-linearities)**: without them, `Dense` computes only an affine transform, and stacking linear layers is still linear, so depth adds nothing to the hypothesis space. `relu` zeroes out negatives; `sigmoid` squashes to [0, 1]; `softmax` outputs a probability distribution summing to 1.
- **Crossentropy**: an information-theory measure of the distance between probability distributions, here between the ground truth and the predictions. It is usually the best loss for probability outputs.
- **Feature-wise normalization**: for each feature, subtract the mean and divide by the standard deviation, so it is centered at 0 with unit std. Compute these statistics on the training data only.
- **MSE vs. MAE**: MSE (squared error) is the regression loss. MAE (absolute error) is the readable metric; on Boston housing an MAE of 0.5 means $500 off on average.
- **Regression is not logistic regression**: logistic regression is a classification algorithm.

## Mental Models
- **The last layer is the task contract**: its width and activation determine what `predict()` returns. A sigmoid gives a probability per sample, a softmax gives a distribution over classes (take `argmax`), and a linear unit gives an unbounded value.
- **Linear output = freedom of range**: any activation on a regression output constrains its range (a sigmoid can only predict values in 0–1).
- **Training curves diverge**: training loss falls every epoch by construction. Validation loss bottoms out and then rises, and that turning point is where overfitting begins.
- **Less data, smaller model**: the less training data you have, the worse overfitting gets. Use one or two small intermediate layers.

## Anti-patterns
- Feeding raw variable-length integer lists into a network. Vectorize to tensors first.
- Evaluating on training data, or tuning against the test set.
- Training to a fixed large epoch count without checking where validation performance peaks.
- Computing normalization statistics on test data. Never use any quantity computed on test data, "even for something as simple as data normalization".
- Feeding features with wildly different ranges unscaled. Learning becomes harder.
- Using a single tiny validation split on a small dataset. Scores are dominated by split variance; use K-fold.
- Making intermediate layers narrower than the output space (information bottleneck).
- Putting an activation on a scalar regression output layer.
- Mismatching labels and loss: one-hot labels need `categorical_crossentropy`, integer labels need `sparse_categorical_crossentropy`.
- Judging accuracy without a baseline. "80%" means little until compared to the ~19% random baseline.

## Code Examples
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.          # multi-hot encoding
    return results

x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype("float32")

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="rmsprop", loss="binary_crossentropy",
              metrics=["accuracy"])
history = model.fit(x_train[10000:], y_train[10000:], epochs=20,
                    batch_size=512,
                    validation_data=(x_train[:10000], y_train[:10000]))
```
- **What it demonstrates**: the binary classification recipe for IMDB, with vectorization, a sigmoid output, binary crossentropy, and a held-out validation set whose curves reveal overfitting after about 4 epochs.

```python
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),                      # linear output for regression
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

k = 4
num_val_samples = len(train_data) // k
all_mae_histories = []
for i in range(k):
    val_slice = slice(i * num_val_samples, (i + 1) * num_val_samples)
    val_data, val_targets = train_data[val_slice], train_targets[val_slice]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]])
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]])
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=500, batch_size=16, verbose=0)
    all_mae_histories.append(history.history["val_mae"])
average_mae_history = np.mean(all_mae_histories, axis=0)
```
- **What it demonstrates**: K-fold validation on a small dataset with a fresh model per fold. Per-epoch validation MAE is averaged across folds to pick the epoch count. The book writes the slicing inline and averages with a list comprehension; this is a condensed equivalent.

```python
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std     # training statistics only
```
- **What it demonstrates**: feature-wise normalization without test-set leakage.

## Reference Tables
| Task | Last layer | Loss | Metric | Label format |
|---|---|---|---|---|
| Binary classification | `Dense(1, activation="sigmoid")` | `binary_crossentropy` | accuracy | 0/1 floats |
| Single-label multiclass (N) | `Dense(N, activation="softmax")` | `categorical_crossentropy` | accuracy | one-hot |
| Same, integer labels | `Dense(N, activation="softmax")` | `sparse_categorical_crossentropy` | accuracy | integers |
| Scalar regression | `Dense(1)` (no activation) | `mse` | `mae` | floats |

| Dataset | Size | Model | Val split | Best epochs | Test result |
|---|---|---|---|---|---|
| IMDB (sentiment) | 25k train / 25k test, 50/50 balanced | 16-16-1 | first 10,000 | ~4 | ~88% acc (state of the art ~95%) |
| Reuters (46 topics) | 8,982 train / 2,246 test | 64-64-46 | first 1,000 | ~9 | ~80% acc (random ~19%) |
| Boston housing | 404 train / 102 test, 13 features | 64-64-1 | 4-fold CV | ~120–140 (used 130) | MAE ~2.46 (about $2,500 off) |

Defaults used throughout: optimizer `rmsprop` ("usually a good default choice for virtually any problem"), `batch_size=512` for the text tasks, `batch_size=16` for Boston housing.

## Worked Example
Boston housing, a small dataset with heterogeneous feature scales. (1) Normalize the 13 features using training mean and std. (2) Define `build_model()` with 64-64-1, `mse` loss, `mae` metric. (3) Run 4-fold CV for 100 epochs: fold MAEs of 2.1–3.1 (mean about 2.6) show how unreliable any single split would be. (4) Rerun for 500 epochs, saving `val_mae` per epoch per fold. Average across folds and plot, dropping the first 10 epochs because they are on a different scale. Validation MAE stops improving around 120–140 epochs. (5) Train a fresh model on all 404 samples for 130 epochs and evaluate on the test set: MAE ~2.46, i.e., about $2,460 error on prices of $10k–$50k. (6) `predict()` returns prices in thousands of dollars (the first test house is about $10,000).

## Key Takeaways
- Raw data usually needs substantial preprocessing into tensors. Word sequences can be multi-hot encoded.
- Stacks of `Dense` + `relu` layers solve a wide range of vector-data problems.
- Binary classification: 1 sigmoid unit + binary crossentropy. N-class single-label: N softmax units + categorical crossentropy (or the sparse variant for integer labels). Scalar regression: 1 linear unit + MSE, monitored with MAE.
- `rmsprop` is generally a good-enough optimizer.
- Networks eventually overfit. Always monitor performance on data outside the training set, and retrain to the epoch where validation peaks.
- Scale features with different ranges independently, using training-set statistics only.
- With little data, use K-fold validation and a small model (one or two intermediate layers).
- With many classes, avoid information bottlenecks from too-small intermediate layers.

## Connects To
- Chapter 3: `compile()`, `fit()`, `History`, `evaluate()`, `predict()`, and validation data.
- Chapter 5: formal principles for choosing layers and units, overfitting vs. generalization, and mitigation techniques.
- Chapter 6: the universal workflow of machine learning, which generalizes this chapter's ad hoc loop, including baselines and evaluation protocols.
- Chapter 11: processing raw text from scratch, the `Embedding` layer, and padding.
