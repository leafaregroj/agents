# Chapter 13: Best Practices for the Real World

## Core Idea
Earlier chapters get models to "works okay" at small scale. Going to "works great and wins machine learning competitions" takes two families of techniques. The first improves model quality systematically: automated **hyperparameter optimization** and **model ensembling**. The second makes training faster: **mixed-precision training**, **multi-GPU data parallelism**, and **TPU training**. Speed matters because faster training means more turns of the **loop of progress** (idea, experiment, results), and more refinement cycles produce better solutions. Don't spend your own time tuning knobs by hand. Hand that search to a machine and design a good search space for it.

## Frameworks Introduced
- **Automated hyperparameter optimization loop**: choose hyperparameters (automatically), build the model, fit on training data and measure on validation data, choose the next set (automatically), repeat, and only at the end measure on test data.
  - When to use: whenever you want close to the best achievable performance. Chollet calls it "an absolute requirement" for state-of-the-art results or competition wins.
  - How: use KerasTuner. Write a model-building function that takes `hp`, pick a tuner (`RandomSearch`, `BayesianOptimization`, `Hyperband`), call `tuner.search(...)` with validation data, then retrain the top configurations on train+val data.
- **The art of crafting the right search space**: hyperparameter tuning is "automation, not magic." Search spaces grow combinatorially, so you choose which choices are worth searching.
  - When to use: every time you set up a tuner.
  - How: move your own decisions up from micro-decisions (units per layer) to higher-level architecture decisions (residual connections everywhere or not). Those generalize across tasks. Start from premade search spaces such as `kt.applications.HyperXception` and `kt.applications.HyperResNet`.
- **Ensembling for diversity**: pool the predictions of several models that are each "as good as possible while being as different as possible."
  - When to use: when you need the last few points of accuracy, as in competitions. Kaggle winners use large ensembles.
  - How: take a weighted average of predictions, with weights learned on validation data by random search or Nelder-Mead. Mix model families, for example tree-based models plus deep nets.
- **Scaling recipe (mixed precision → MirroredStrategy → TPUStrategy)**: three ways to train faster, from free on one GPU to specialized hardware.
  - When to use: once coding speed stops being your bottleneck and training speed takes its place.
  - How: `keras.mixed_precision.set_global_policy("mixed_float16")`; `tf.distribute.MirroredStrategy()` and build/compile inside `strategy.scope()`; `TPUStrategy(tpu)` plus `steps_per_execution`.

## Key Concepts
- **Hyperparameters vs. parameters**: hyperparameters are architecture-level and training-level choices (depth, units, activation, dropout, use of BatchNormalization). Parameters are weights learned by backpropagation.
- **Why hyperparameter search is hard**: (1) the space is discrete and not differentiable, so it needs gradient-free optimization, which is far less efficient; (2) every feedback signal means training a new model from scratch; (3) the signal is noisy, so a 0.2% gain may just be lucky initialization.
- **Search space**: the set of hyperparameter ranges a tuner explores, sampled through `hp.Int`, `hp.Float`, `hp.Boolean`, and `hp.Choice`.
- **HyperModel**: a subclass of `kt.HyperModel` with a `build(self, hp)` method. Lets you pass model constants (such as `num_classes`) as constructor arguments.
- **Trial / execution**: a trial is one hyperparameter configuration. `executions_per_trial` trains each configuration several times and averages the results to reduce variance.
- **Validation-set overfitting**: hyperparameters are tuned on the validation signal, so they effectively "train" on validation data and overfit to it. Keep the test set completely separate.
- **AutoML**: automated machine learning, the next step beyond hyperparameter search: generating architectures or whole pipelines through reinforcement learning or genetic algorithms (for example AutoKeras). Engineers will move "up the value-creation chain" to data curation, loss functions that reflect business goals, and the model's effect on its ecosystem.
- **Floating-point precision**: float16 (half), float32 (single), float64 (double). The resolution is about 1e-3, 1e-7, and 1e-16 respectively. Representable values are not evenly spaced: larger magnitudes have coarser precision.
- **Mixed precision**: run most of the forward pass in float16 and keep weights stored and updated in float32. Each layer has a `compute_dtype` (float16 under the policy) and a `variable_dtype` (stays float32).
- **Data parallelism vs. model parallelism**: data parallelism replicates one model on each device, and each copy processes different batches. Model parallelism splits one model across devices and is used only when the model does not fit on a single device.
- **Single-host, multi-device synchronous training**: what `MirroredStrategy` does. A *global batch* is split into *local batches*, each replica computes weight deltas, and the deltas are merged and applied to every replica, so replicas stay identical.
- **TPU / ASIC**: Google's Tensor Processing Unit, a chip built specifically for deep learning. Colab's TPU runtime uses two VMs, so the TPU cannot read the notebook VM's local disk.
- **Step fusing**: `compile(..., steps_per_execution=N)` runs N training steps per TPU execution, so the device stays busy without huge batches.

## Mental Models
- **Blind men and the elephant**: each model captures part of the truth of the data manifold through its own architecture and random initialization. Pooling the views gives a more accurate picture. If everyone touches only the trunk, the ensemble keeps the same bias.
- **The loop of progress**: idea → experiment (Keras) → results (TensorBoard) → idea, with GPUs and TPUs driving it. A turnaround of 10–15 minutes allows dozens of iterations per day.
- **Precision is to numbers what resolution is to images**: float16 cannot represent gradient updates around 1e-5 to 1e-6 (typical learning rate 1e-3, weight updates around 1e-6). That is why weights stay in float32 while most compute moves to float16.
- **Tuning moves you up the decision ladder**: once a machine handles micro-decisions, your judgment goes into architecture-level choices that carry over to other problems.

## Anti-patterns
- **Using the test set as validation data during search**: you will quickly overfit the test set and can no longer trust its metrics.
- **Turning everything into a hyperparameter**: the space explodes combinatorially. Tuning does not replace knowing architecture best practices.
- **Hand-fiddling hyperparameters all day**: that work belongs to a machine.
- **Ensembling the same network trained from different seeds**: low diversity gives only a tiny gain.
- **Plain averaging when one model is much worse**: the ensemble can end up worse than the best single model. Use validation-learned weights.
- **Implicit float64 from NumPy**: `tf.convert_to_tensor(np.zeros((2, 2)))` produces float64. Pass `dtype="float32"`. (`fit()` converts NumPy input for you.)
- **Leaving softmax/crossentropy in float16**: these are numerically unstable. Opt out per layer with `dtype="float32"`.
- **Small batches on multi-GPU or TPU**: local batches too small to keep each device busy waste the hardware.
- **Huge batches with an unchanged learning rate**: with fewer but more accurate updates, raise the learning rate.
- **Reading TPU data from GCS repeatedly**: creates an I/O bottleneck. Use `dataset.cache()` when the data fits in memory, and TFRecord files when it doesn't.

## Code Examples
```python
import keras_tuner as kt  # the book imports it as `kerastuner`
from tensorflow import keras
from tensorflow.keras import layers

def build_model(hp):
    units = hp.Int(name="units", min_value=16, max_value=64, step=16)
    model = keras.Sequential([
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",       # always a validation metric
    max_trials=100,
    executions_per_trial=2,         # average out noise
    directory="mnist_kt_test",
    overwrite=True)                 # False to resume a crashed search

tuner.search(x_train, y_train, batch_size=128, epochs=100,
             validation_data=(x_val, y_val),
             callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)])
best_hps = tuner.get_best_hyperparameters(4)
```
- **What it demonstrates**: replacing hard-coded values with sampled ranges, choosing a validation objective, and using a large epoch budget with EarlyStopping because you don't know in advance how many epochs each model needs. For a custom metric, pass `objective=kt.Objective(name=..., direction="max"|"min")`.

```python
def get_best_trained_model(hp):
    best_epoch = get_best_epoch(hp)   # refit with patience=10, argmin of val_loss + 1
    model = build_model(hp)
    model.fit(x_train_full, y_train_full,           # train + val combined
              batch_size=128, epochs=int(best_epoch * 1.2))
    return model

# Shortcut if slight underperformance is acceptable:
# best_models = tuner.get_best_models(top_n)
```
- **What it demonstrates**: the post-search retraining protocol. Find the best epoch with a high-patience EarlyStopping run, then retrain on all the data for about 20% more epochs, since there is more data.

```python
import tensorflow as tf
from tensorflow import keras

keras.mixed_precision.set_global_policy("mixed_float16")   # up to 3x on modern NVIDIA GPUs

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")
with strategy.scope():                 # everything that creates variables
    model = get_compiled_model()       # build + compile() go here
model.fit(train_dataset.prefetch(tf.data.AUTOTUNE),
          epochs=100, validation_data=val_dataset, callbacks=callbacks)

# TPU variant:
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
# strategy = tf.distribute.TPUStrategy(tpu)
# with strategy.scope(): model = build_model(...)  # compile(..., steps_per_execution=8)
```
- **What it demonstrates**: the scaling stack. Turn on the mixed-precision policy, open a strategy scope for variable creation, and feed a prefetched `tf.data.Dataset`.

## Reference Tables
| Technique | Typical gain | Key setting / caveat |
|---|---|---|
| Mixed precision (GPU) | up to 3x | `mixed_float16`; keep softmax/crossentropy at `dtype="float32"` |
| Mixed precision (TPU) | up to 60% | same policy idea |
| MirroredStrategy, 2 GPUs | ~2x | large global batch |
| MirroredStrategy, 4 GPUs | ~3.8x | e.g., global batch 512 → 4 × 128 local |
| MirroredStrategy, 8 GPUs | ~7.3x | merging deltas adds overhead |
| TPU v2 vs. NVIDIA P100 | ~15x faster, ~3x more cost-effective | batch ≥ 1024; small models may need >10,000 per batch, or `steps_per_execution` |

| Precision | Bits | Smallest safely processed distance |
|---|---|---|
| float16 | 16 | ~1e-3 |
| float32 | 32 (1 sign, 8 exponent, 23 mantissa) | ~1e-7 |
| float64 | 64 | ~1e-16 |

## Worked Example
In 2014, Chollet and Andrey Kolev took 4th place in Kaggle's Higgs Boson challenge with an ensemble of tree models and deep nets. One member, a regularized greedy forest, scored significantly worse than the others and got a small weight, yet it improved the ensemble by a large factor because it was so different. The lesson from the chapter: an ensemble's strength comes less from its best model than from the diversity of its candidates.

## Key Takeaways
- Use KerasTuner to automate configuration search. Always optimize a validation metric and watch for validation-set overfitting.
- Design the search space deliberately. Tuning makes your decisions higher-level but doesn't remove them.
- After the search, retrain the top configurations on train+val data for about 1.2x the best epoch count.
- A validation-weighted average of diverse models is a very strong ensembling baseline. Diversity is strength.
- Turn on mixed precision on GPU for a nearly free speedup.
- Scale out with `MirroredStrategy` (build and compile inside the scope), use large global batches, and prefetch data.
- On TPUs, use `TPUStrategy`, keep data in memory or GCS/TFRecord, use large batches with a higher learning rate, or use step fusing.

## Connects To
- Ch 5–6: validation protocols, the overfit-then-regularize workflow, and information leaks that tuning makes worse.
- Ch 7: the loop of progress, callbacks (EarlyStopping), and TensorBoard.
- Ch 8–9: residual connections and BatchNormalization as higher-level search-space choices, and HyperXception/HyperResNet.
- Ch 14: Kaggle practice for hyperparameter tuning, validation-set overfitting, and ensembling; the shortcut rule (competition-winning ensembles rarely reach production).
