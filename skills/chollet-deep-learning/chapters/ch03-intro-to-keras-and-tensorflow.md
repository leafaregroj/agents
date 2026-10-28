# Chapter 3: Introduction to Keras and TensorFlow

## Core Idea
Training a neural network maps onto two layers of tooling. **TensorFlow** supplies low-level tensor infrastructure: tensors, variables, tensor operations, and automatic differentiation via `GradientTape`, runnable on CPU, GPU, or TPU and exportable to other runtimes. **Keras** supplies the high-level deep learning concepts: layers combined into models, a loss, an optimizer, metrics, and a training loop (`compile()` / `fit()` / `evaluate()` / `predict()`). Once you know both levels, you can build any gradient-descent model from scratch, or assemble one productively from Keras building blocks.

## Frameworks Introduced
- **TensorFlow vs. Keras division of labor**: TensorFlow = tensor manipulation infrastructure (tensors, variables, autodiff, distribution); Keras = deep learning development (layers, models, optimizers, losses, metrics). Hardware sits beneath both.
  - When to use: deciding which API level to reach for. Use Keras for model building and training; drop to raw TensorFlow when you need control over the math or the update rule.
  - How: TensorFlow is "NumPy plus gradients plus accelerators plus deployment"; Keras is "the Python of deep learning", one API offering many workflows.
- **Spectrum of workflows / progressive disclosure of complexity**: Keras does not impose one "true" way to build and train models. It offers a range from very high level (Scikit-learn-like `fit()`) to very low level (NumPy-like full control), each trading usability against flexibility.
  - When to use: start at the highest level that fits; move down only when you need custom behavior.
  - How: the same skills carry over from student to researcher to engineer, with no framework switch needed.
- **The Layer as the central abstraction**: a `Layer` encapsulates state (weights, created in `build()`) and computation (the forward pass, in `call()`). Everything in Keras is either a Layer or something that interacts closely with one.
  - When to use: whenever you implement a custom building block.
  - How: subclass `keras.layers.Layer`; create weights with `add_weight()` in `build(input_shape)`; put the forward pass in `call(inputs)`.
- **Model topology as hypothesis space**: choosing an architecture constrains the space of possibilities to a specific series of tensor operations. Training then searches for good weight values inside that space.
  - When to use: when reasoning about what a model can possibly learn.
  - How: architecture encodes your prior assumptions. A single `Dense` layer with no activation assumes the classes are linearly separable.
- **The three compile-time choices**: loss (the quantity minimized; must be differentiable), optimizer (how weights are updated; a variant of SGD), metrics (what you monitor; need not be differentiable, not optimized directly).

## Key Concepts
- **Constant tensors**: TensorFlow tensors are immutable. Item assignment fails ("EagerTensor object does not support item assignment").
- **`tf.Variable`**: the class for mutable state. Update it with `assign()`, `assign_add()`, `assign_sub()`, including on slices (`v[0, 0].assign(3.)`).
- **Eager execution**: TensorFlow ops run immediately, and results can be printed like NumPy.
- **GradientTape**: records operations inside a `with` scope; `tape.gradient(result, inputs)` returns gradients. Trainable variables are watched automatically. Constant tensors must be marked with `tape.watch()`, since storing gradient info for everything would be too costly. Nested tapes give second-order gradients.
- **Automatic shape inference**: layers build their weights lazily on first call, inferring input dimensions from the first input seen. The base `__call__()` checks `self.built`, calls `build(inputs.shape)` once, then `call(inputs)`. It also handles eager/graph routing and masking.
- **Layer–data matching**: rank-2 `(samples, features)` vector data → `Dense`; rank-3 `(samples, timesteps, features)` sequences → `LSTM` or `Conv1D`; rank-4 images → `Conv2D`.
- **Model**: a graph of layers. `Sequential` is a simple stack. Other topologies (two-branch, multihead, residual, Transformer) are built with the Functional API or by subclassing `Model`.
- **String shortcuts vs. object instances in `compile()`**: `"rmsprop"` becomes `keras.optimizers.RMSprop()`. Pass instances when you need configuration (e.g., `learning_rate`) or custom losses/metrics.
- **`History` object**: `fit()` returns it; `history.history` maps `"loss"`, metric names, and `"val_*"` keys to per-epoch lists.
- **Validation data**: a held-out subset used only to compute loss and metrics, never to update weights. The "validation loss" is distinct from the "training loss".
- **Inference**: using a trained model. `model(x)` processes everything at once; `model.predict(x, batch_size=...)` iterates in batches, returns NumPy, and accepts `Dataset` objects.

## Mental Models
- **Layers as LEGO bricks**: models are made by clipping together compatible layers into data-transformation pipelines. Compatibility means matching shapes, which Keras handles through shape inference.
- **The ruthless optimizer**: a network will take any shortcut to lower its loss. If the objective does not fully correlate with success, you get unintended behavior. Chollet's illustration: an AI maximizing "average well-being of all humans alive" might eliminate most humans. Choose the objective wisely.
- **Linear classifier = a line (hyperplane)**: the decision boundary `w1*x + w2*y + b = 0.5` is a line in 2D. Linear classification is finding the parameters of a separating hyperplane.
- **Batch vs. mini-batch trade-off**: full-batch steps are slower but each update is more informative. So you need fewer steps and can use a larger learning rate (0.1 in the toy example) than with mini-batches.

## Anti-patterns
- Trying to assign into a TensorFlow tensor. Use a `tf.Variable` instead.
- Asking for gradients with respect to a constant tensor without `tape.watch()`. You get `None`.
- Passing input sizes manually to every layer (the `NaiveDense(input_size=..., output_size=...)` style). Let `build()` infer shapes.
- Putting the forward pass in `__call__()` instead of `call()`. This breaks lazy building and the base class's other services.
- Evaluating only on training data, or letting validation samples leak into training. Validation metrics then become meaningless.
- Splitting validation data from unshuffled, class-sorted arrays. You can end up with samples from only one class.
- Calling `model(huge_array)` for inference on large data. It may exceed GPU memory; use `predict()`.
- Running deep learning on native Windows. The book recommends Unix, WSL, or an Ubuntu dual boot.

## Code Examples
```python
import tensorflow as tf

# Linear classifier in pure TensorFlow (inputs: (2000, 2) float32, targets: (2000, 1))
input_dim, output_dim = 2, 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

def model(inputs):
    return tf.matmul(inputs, W) + b

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

for step in range(40):  # full-batch training; loss settles near 0.025
    loss = training_step(inputs, targets)
```
- **What it demonstrates**: tensors, variables, a forward pass, a loss, gradients, and the weight update are enough to build any gradient-descent model. This is a classic interview-filter exercise.

```python
from tensorflow import keras

class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):           # called once, on first __call__
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    def call(self, inputs):                 # the forward pass
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

model = keras.Sequential([SimpleDense(32, activation="relu"),
                          SimpleDense(10, activation="softmax")])
```
- **What it demonstrates**: the state-plus-computation anatomy of a Keras layer, and lazy weight creation that enables automatic shape inference.

```python
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)   # 9.8
```
- **What it demonstrates**: nested tapes compute second-order gradients.

## Reference Tables
| Concept | TensorFlow / Keras API |
|---|---|
| Tensor creation | `tf.ones`, `tf.zeros`, `tf.random.normal`, `tf.random.uniform`, `tf.constant` |
| Mutable state | `tf.Variable`, `.assign`, `.assign_add`, `.assign_sub` |
| Math | `tf.square`, `tf.sqrt`, `tf.matmul`, `+`, `*`, `tf.reduce_mean` |
| Gradients | `tf.GradientTape`, `tape.gradient`, `tape.watch` |
| Configure training | `model.compile(optimizer, loss, metrics)` |
| Train | `model.fit(x, y, epochs, batch_size, validation_data)` → `History` |
| Evaluate | `model.evaluate(x, y, batch_size)` → loss, or `[loss, *metrics]` |
| Predict | `model.predict(x, batch_size)` → NumPy array |

| Built-ins | Examples |
|---|---|
| Optimizers | SGD (with or without momentum), RMSprop, Adam, Adagrad |
| Losses | CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError, KLDivergence, CosineSimilarity |
| Metrics | CategoricalAccuracy, SparseCategoricalAccuracy, BinaryAccuracy, AUC, Precision, Recall |

| Workspace option | Use when |
|---|---|
| Colaboratory (free GPU/TPU) | getting started, small workloads (the book's recommendation) |
| Cloud GPU (Google Cloud, AWS) | scaling up without buying hardware (about $2.48/hr for a V100, mid-2021) |
| Local NVIDIA GPU workstation | heavy, sustained use ($1,500–$2,500 consumer GPU) |

## Worked Example
Linear classifier on synthetic data. Draw 1,000 points per class from `np.random.multivariate_normal` with the same covariance `[[1, 0.5], [0.5, 1]]` and means `[0, 3]` and `[3, 0]`. Stack them into `(2000, 2)` float32 inputs with `(2000, 1)` targets of 0s and 1s. Train `prediction = W·input + b` with MSE, full-batch, `learning_rate=0.1`, for 40 steps. Loss stabilizes near 0.025. Classify by thresholding at 0.5. The decision boundary is the line `y = -w1/w2 * x + (0.5 - b)/w2`.

The same problem in Keras uses `keras.Sequential([keras.layers.Dense(1)])`, compiled with `RMSprop(learning_rate=0.1)`, `MeanSquaredError()`, and `BinaryAccuracy()`. Shuffle with `np.random.permutation`, reserve 30% for validation, then call `fit(..., epochs=5, batch_size=16, validation_data=(val_inputs, val_targets))`.

## Key Takeaways
- TensorFlow = NumPy-like tensors plus autodiff, hardware acceleration, distribution, and deployment. Keras is its official high-level API (since 2018, central in TF 2.0).
- Use `tf.Variable` for state, `GradientTape` for gradients, and `assign_sub` for updates.
- Implement custom layers by creating weights in `build()` and putting the forward pass in `call()`.
- The architecture is your hypothesis space: it encodes prior assumptions about the problem.
- Before training, pick a loss, an optimizer, and metrics via `compile()`. The loss must correlate with real success.
- `fit()` runs mini-batch gradient descent. Always monitor validation data kept strictly separate. Use `evaluate()` after training and `predict()` for batched inference.
- GPUs typically give a 5–10x speedup over CPU, and much more for convnets.

## Connects To
- Chapter 2: `NaiveDense`/`NaiveSequential` and the from-scratch training loop that this chapter refines.
- Chapter 4: first real classification and regression tasks with `compile()` and `fit()`.
- Chapter 5: generalization, overfitting, and why validation matters.
- Chapter 7: the full spectrum of workflows, Functional API, `Model` subclassing, custom training loops, custom losses and metrics, and eager vs. graph execution.
- Chapter 11: masking and the Transformer architecture. Chapter 13: TPU setup.
