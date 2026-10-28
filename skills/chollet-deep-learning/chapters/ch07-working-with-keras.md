# Chapter 7: Working with Keras: A Deep Dive

## Core Idea
Keras is not one API but a **spectrum of workflows** governed by the principle of **progressive disclosure of complexity**: simple things are easy, arbitrarily advanced things are possible, and each step up requires only incremental learning. Every workflow is built on the same shared abstractions (`Layer`, `Model`, `Metric`, `Callback`), so components from any point on the spectrum interoperate. The chapter walks the spectrum twice: once for *building* models (Sequential -> Functional -> subclassing) and once for *training* them (`fit()` -> custom metrics/callbacks -> overriding `train_step()` -> fully custom loops).

## Frameworks Introduced
- **Progressive disclosure of complexity**: start with the simplest workflow that works; move one notch toward flexibility only when the task demands it, reusing what you already know.
  - When to use: every time you choose how to build or train a model.
  - How: default to Functional API + `fit()`; escalate to subclassed layers, then a custom `train_step()`, then a from-scratch loop only when a concrete need forces it.
- **Three model-building APIs**:
  - **Sequential model**: a Python list of layers; single input, single output, linear stack. Use for novices and simple stacks.
  - **Functional API**: layers called on symbolic tensors form an explicit directed acyclic graph. Use by default; it handles multi-input, multi-output, and nonlinear topologies and gives access to layer connectivity.
  - **Model subclassing**: define sublayers in `__init__()`, forward pass in `call()`. Use only for models that are not DAGs of layers (loops, recursion) or research needing full control.
- **Spectrum of training workflows**:
  - `compile()/fit()/evaluate()/predict()` for standard supervised learning.
  - Customize with **custom metrics** (subclass `keras.metrics.Metric`) and **callbacks** (built-in or subclass `keras.callbacks.Callback`).
  - **Override `train_step()`** when you need a custom learning algorithm but still want callbacks, distribution, and the progress bar of `fit()`.
  - **Write a training loop from scratch** with `tf.GradientTape` for GANs, self-supervised, RL, or any non-standard setup.
- **The loop of progress**: Idea -> Experiment (Keras) -> Results (GPUs/TPUs) -> Visualization (TensorBoard) -> next Idea. Speed of iteration through this loop determines how refined your ideas become.

## Key Concepts
- **Deferred building**: layers (and a Sequential model) create weights only on first call or `build(input_shape)`, because weight shapes depend on input shape. Accessing `model.weights` before that raises an error. Declaring `keras.Input(shape=...)` up front builds incrementally so `summary()` works after each `add()`.
- **Input shape is per-sample**: `Input(shape=(3,))` excludes the batch dimension; `None` in reported shapes means any batch size.
- **Symbolic tensor**: the object returned by `keras.Input`; carries shape and dtype but no data, standing for future tensors. Calling a layer on it returns a new symbolic tensor with updated shape.
- **Layer connectivity**: a Functional model is an inspectable graph. `model.layers`, `layer.input`, `layer.output` let you plot (`keras.utils.plot_model(..., show_shapes=True)`) and do **feature extraction** by building new models from intermediate nodes.
- **Layer vs Model**: nearly identical classes; a Model is the top-level object with `fit()`, `evaluate()`, `predict()`, and can be saved to disk.
- **Subclassed model limits**: it is bytecode, not a data structure, so `summary()` cannot show connectivity, `plot_model()` fails, and you cannot extract intermediate nodes; the forward pass is a black box and the error surface is larger.
- **Metric state**: metrics hold TensorFlow variables updated manually (not by backprop) through `update_state()`, read via `result()`, cleared with `reset_state()` so one instance serves across epochs and train/eval.
- **Callbacks**: objects called by `fit()` at hook points (`on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`), each receiving a `logs` dict. The progress bar itself is a callback.
- **training argument**: layers like `Dropout` behave differently in training and inference; always call `model(x, training=True)` inside a custom training step and `training=False` for evaluation.
- **Trainable vs non-trainable weights**: trainable weights are updated by backprop; non-trainable ones are updated by their layer during the forward pass (in built-in layers, only `BatchNormalization` has them). Take gradients against `model.trainable_weights`.
- **Eager vs graph execution**: TF runs eagerly (line by line) by default, which is easy to debug but slow; `@tf.function` compiles to an optimizable graph.
- **Compiled attributes**: after `compile()`, a custom `train_step()` can use `self.compiled_loss`, `self.compiled_metrics`, and `self.metrics`.

## Mental Models
- **LEGO bricks**: the Functional API snaps layers together into arbitrary graphs; subclassing is building a new Python object rather than snapping bricks.
- **Paper airplane vs drone**: a bare `fit()` call is a paper airplane with no control after launch; callbacks turn it into a drone that senses its state and steers itself (stops, saves, adjusts learning rate).
- **Functional model with subclassed layers = best of both worlds**: bespoke computation where needed, while retaining plotting, summaries, and feature extraction.

## Anti-patterns
- Reaching for model subclassing when the model is expressible as a DAG of layers; you lose inspection, plotting, and feature extraction for nothing.
- Training for a fixed epoch count, finding the overfit point, then retraining from scratch; use `EarlyStopping` + `ModelCheckpoint(save_best_only=True)` instead.
- Forgetting `training=True` in a custom forward pass (dropout stays off) or differentiating against `model.weights` instead of `model.trainable_weights`.
- Forgetting to reset metric state at the start of each epoch and before evaluation.
- Debugging inside `@tf.function`; develop eagerly, then add the decorator once the code works.
- Monitoring a metric in a callback (e.g., `val_accuracy`) without passing `validation_data` to `fit()` or listing the metric in `compile()`.
- In subclassed models, passing loss/metric/target structures that do not exactly match what `call()` returns.

## Code Examples
```python
# Multi-input, multi-output Functional model (ticket triage)
vocabulary_size, num_tags, num_departments = 10000, 100, 4
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")
features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax",
                          name="department")(features)
model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])
model.compile(optimizer="rmsprop",
              loss={"priority": "mean_squared_error",
                    "department": "categorical_crossentropy"},
              metrics={"priority": ["mean_absolute_error"],
                       "department": ["accuracy"]})
# Feature extraction: add a new head on an intermediate node
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(
    model.layers[4].output)
new_model = keras.Model(inputs=[title, text_body, tags],
                        outputs=[priority, department, difficulty])
```
- **What it demonstrates**: named inputs/outputs allow dict-based `compile()`/`fit()`; graph access lets you extend a trained model without retraining from scratch.

```python
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2),
    keras.callbacks.ModelCheckpoint(filepath="checkpoint_path.keras",
                                    monitor="val_loss", save_best_only=True),
]
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10, callbacks=callbacks_list,
          validation_data=(val_images, val_labels))
model = keras.models.load_model("checkpoint_path.keras")
```
- **What it demonstrates**: the standard early-stopping + best-checkpoint pairing.

```python
class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}

inputs = keras.Input(shape=(28 * 28,))
x = layers.Dense(512, activation="relu")(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = CustomModel(inputs, outputs)
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=3)
```
- **What it demonstrates**: the middle ground: your own learning step, with `fit()` still providing callbacks, distribution, metric resets, and automatic `tf.function` compilation. Works with Functional construction of a subclass.

## Reference Tables
| Need | Use |
|---|---|
| Simple stack, one input/output | Sequential |
| Multi-input/output, shared layers, plotting, feature extraction | Functional API (default) |
| Loops/recursion in forward pass, non-DAG | Model subclassing |
| Unusual per-layer computation inside a graph model | Functional model + subclassed `Layer` |
| Standard supervised training | `fit()` |
| Stop when val metric stalls / keep best model | `EarlyStopping` + `ModelCheckpoint(save_best_only=True)` |
| Adjust LR during training | `LearningRateScheduler`, `ReduceLROnPlateau` |
| Log metrics to file | `CSVLogger` |
| Live curves, histograms, embeddings | `TensorBoard` callback |
| Custom algorithm, keep `fit()` features | override `train_step()` (+ `metrics` property) |
| GANs, RL, self-supervised, novel research | from-scratch `GradientTape` loop + `@tf.function` |

Custom metric skeleton: `__init__` (create state with `add_weight`), `update_state(y_true, y_pred, sample_weight=None)`, `result()`, `reset_state()`.

## Worked Example
From-scratch training loop on MNIST (3 epochs, batch 32): (1) prepare `loss_fn`, `optimizer`, a list of metrics, and a `keras.metrics.Mean()` loss tracker; (2) `train_step` runs the forward pass under a tape with `training=True`, takes gradients w.r.t. `trainable_weights`, applies them, updates each metric and the loss tracker, and returns a logs dict; (3) `reset_metrics()` at the start of each epoch; (4) iterate a `tf.data.Dataset.from_tensor_slices(...).batch(32)`; (5) `test_step` is the same minus tape and optimizer, with `training=False` and `val_` prefixes. Adding `@tf.function` to `test_step` cut evaluation from 1.80 s to 0.8 s on a Colab CPU.

## Key Takeaways
- There is no single "true" way to use Keras; choose the point on the spectrum of workflows that fits the job, and know you can mix them freely.
- Prefer the Functional API whenever the model is a DAG of layers; prefer Functional models containing subclassed layers over fully subclassed models.
- `fit()` covers most needs; extend it with custom metrics and callbacks before abandoning it.
- Always pair early stopping with best-only checkpointing and validation data.
- Custom loops require three details: `training=True`, `trainable_weights`, and metric resets. Debug eagerly, then compile with `@tf.function`.
- Overriding `train_step()` is the sweet spot for custom algorithms that still want the built-in machinery.

## Connects To
- Chapter 3 (Layer subclassing, `GradientTape` basics) and Chapter 2 (the training loop).
- Chapter 6 (universal workflow; early stopping replaces manual epoch search).
- Chapter 8 onward: all models use the Functional API; `ModelCheckpoint` is used throughout.
- Chapter 9 (`BatchNormalization` and its non-trainable weights).
- Chapter 12 (generative models needing custom training steps).
