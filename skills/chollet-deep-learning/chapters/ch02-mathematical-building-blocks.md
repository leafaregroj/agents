# Chapter 2: The Mathematical Building Blocks of Neural Networks

## Core Idea
A neural network is a chain of simple, differentiable **tensor operations** whose **weights** are tensors. Those weights are where the model's "knowledge" lives. Training adjusts the weights with **mini-batch stochastic gradient descent**: run a forward pass, measure the loss, compute the gradient of the loss with respect to every weight using **backpropagation** (the chain rule applied to a computation graph), and step against the gradient. Chollet uses executable code rather than math notation, because "the most precise, unambiguous description of a mathematical operation is its executable code."

## Frameworks Introduced
- **The four-part training setup (layers, loss, optimizer, metrics)**: a `Sequential` stack of layers, compiled with an optimizer, a loss function and metrics, then trained with `fit()`.
  - When to use: for any supervised Keras model.
  - How: the loss is the quantity you minimize, so it must stand for success on the task. The optimizer defines how the gradient of the loss updates the parameters (RMSProp, SGD with momentum, and so on). The metrics are what you monitor, such as accuracy.
- **The training loop**: (1) draw a batch `x`, `y_true`; (2) forward pass → `y_pred`; (3) compute the loss; (4) backward pass, computing the gradient of the loss with respect to the parameters; (5) update `W -= learning_rate * gradient`.
  - When to use: this is what `fit()` does internally. Write it yourself (chapter 7) when you need custom training.
- **Backpropagation = chain rule + computation graph**: `grad(y, x) == grad(y, x1) * grad(x1, x)`. Along a path in the backward graph, multiply the edge derivatives. Across several paths, add them.
  - When to use: to reason about how gradients flow, for example why they vanish in deep stacks or why skip connections help.
  - How: in practice you never write it by hand. Use automatic differentiation through `tf.GradientTape`.
- **Geometric interpretation of deep learning**: every tensor operation is a geometric transformation. Addition is translation. A dot product is a linear transform (rotation, scaling). Dot plus bias is an affine transform. Affine plus relu is a non-linear fold.
  - When to use: to build intuition about what layers do, and to understand why activations are required.

## Key Concepts
- **Tensor**: a container for numbers that generalizes matrices to any number of axes. Its three key attributes are **rank** (`ndim`, the number of axes), **shape** (a tuple of dimensions per axis), and **dtype** (`float32`, `uint8`, and so on).
- **Rank naming**: scalar = rank-0, shape `()`. Vector = rank-1. Matrix = rank-2. Then rank-3 and higher. A "5D vector" (one axis with 5 entries) is not a "5D tensor" (5 axes).
- **Samples axis / batch axis**: axis 0 of every data tensor. Models process data in **batches**, for example `train_images[128*n : 128*(n+1)]`.
- **Class / sample / label**: a category / a data point / the class assigned to a specific sample.
- **Element-wise operations**: relu, `+`, `*` and so on, applied independently to each entry. They vectorize to BLAS on CPU or CUDA on GPU. In the book's timing test, NumPy took 0.02 s against 2.45 s for naive Python loops.
- **Broadcasting**: when the shapes differ, NumPy adds broadcast axes to the smaller tensor and repeats it virtually, without copying memory. A tensor of shape `(n, …, m)` broadcasts against one of shape `(a, b, …, n, …, m)`.
- **Tensor product (dot)**: unlike the element-wise `*`, it requires `x.shape[1] == y.shape[0]` and gives `(a, b)·(b, c) → (a, c)`. In general, `(a, b, c, d)·(d, e) → (a, b, c, e)`. It is not symmetric once either input has rank > 1.
- **Reshaping**: rearranges entries into a target shape with the same total size. **Transposition** is a special case: `x[i, :]` becomes `x[:, i]`.
- **Dense layer**: `output = relu(dot(input, W) + b)`. `W` is the **kernel** and `b` the **bias**. Without an activation, a Dense layer is an affine layer.
- **Activation function**: a non-linearity such as relu (`max(x, 0)`). Without it, stacked Dense layers collapse into one affine transform: `W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)`.
- **Derivative / gradient**: the derivative is the slope of a local linear approximation, `f(x + ε) ≈ f(x) + a·ε`. The gradient generalizes this to tensor inputs: `grad(loss, W0)` has the shape of `W` and points in the direction of steepest ascent. Each of its entries is a **partial derivative**.
- **Learning rate**: the step size. Too small: slow progress and a risk of getting stuck in a local minimum. Too large: updates jump to random places on the loss curve.
- **Mini-batch SGD vs. true SGD vs. batch gradient descent**: true SGD uses one sample per update and batch gradient descent uses all the data. Mini-batches are the efficient compromise between the two.
- **Momentum**: the update uses the previous velocity as well as the current gradient. This speeds up convergence and helps escape local minima, like a ball rolling with inertia. Optimizers built on related ideas include SGD with momentum, Adagrad and RMSprop.
- **Computation graph**: a directed acyclic graph of tensor operations. It treats "computation as data", which makes automatic differentiation and automatic distribution possible.
- **`tf.Variable` / `GradientTape`**: a Variable holds mutable state such as weights. The tape records the operations run inside its scope, and `tape.gradient(y, x)` (or `tape.gradient(y, [W, b])`) returns the gradients.
- **Epoch**: one full pass over the training data.
- **Overfitting (first sighting)**: the model does worse on new data than on its training data. In this chapter's MNIST example, train accuracy is 98.9% and test accuracy is 97.8%.

## Mental Models
- **Layers as data filters / a sieve**: each layer refines its input into a more useful representation, which is "progressive data distillation".
- **Uncrumpling a paper ball**: two sheets of colored paper crumpled together form the input manifold. The network learns a sequence of simple finger movements that uncrumple the ball until the classes separate. Each layer disentangles a little, and depth makes an intractable unfolding tractable. A **manifold** is a continuous surface.
- **Ball rolling down the loss curve**: a picture of gradient descent, with momentum as inertia that carries the ball over small dips.
- **Low-dimensional intuition is fragile**: loss surfaces with millions of dimensions do not look like 2D pictures. Treat 1D and 2D sketches as aids, not truth.
- **Weights are where knowledge persists**: the architecture is fixed and the weights are what get learned.

## Anti-patterns
- **Perturbing one weight at a time** to see how the loss moves: this costs two forward passes per parameter and is hopeless at millions of parameters. Use gradients, which update every weight in one pass.
- **Solving `grad(f(W), W) = 0` analytically**: this is intractable for real networks, which have thousands to tens of millions of parameters.
- **Stacking Dense layers without activations**: this gives "a linear model in disguise".
- **Writing Python loops for tensor math**: it is about 100× slower than vectorized NumPy or TensorFlow.
- **Hand-written weight updates in production**: use `optimizer.apply_gradients(...)` instead of `w.assign_sub(g * lr)`.
- **Confusing `*` (element-wise) with `dot` (tensor product)**.

## Code Examples
```python
# The canonical first Keras model (MNIST)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)  # ~0.978
```
- **What it demonstrates**: the complete workflow: load the data, reshape and scale it to [0, 1] as float32, stack the layers, compile with optimizer, loss and metrics, then fit and evaluate.

```python
# One training step "from scratch" with GradientTape
import tensorflow as tf
from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)                     # forward pass
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)    # backward pass
    optimizer.apply_gradients(zip(gradients, model.weights))  # weight update
    return average_loss
```
- **What it demonstrates**: what `fit()` does for each batch. The manual alternative to `apply_gradients` is `w.assign_sub(g * learning_rate)`.

```python
# A minimal Dense layer: output = activation(dot(input, W) + b)
class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        w_init = tf.random.uniform((input_size, output_size), minval=0, maxval=1e-1)
        self.W = tf.Variable(w_init)
        self.b = tf.Variable(tf.zeros((output_size,)))

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]
```
- **What it demonstrates**: a layer is just weight Variables plus a forward function. Chain such layers (as `NaiveSequential` does) and collect their weights, and you have a model.

## Reference Tables

| Data type | Tensor rank | Shape convention | Example |
|---|---|---|---|
| Vector data | 2 | `(samples, features)` | 500 docs × 20,000-word counts → `(500, 20000)` |
| Timeseries / sequence | 3 | `(samples, timesteps, features)`; time is always axis 1 | 250 days × 390 min × 3 prices → `(250, 390, 3)` |
| Images | 4 | `(samples, height, width, channels)` (channels-last, the TF default) | 128 RGB 256×256 → `(128, 256, 256, 3)`; grayscale keeps channel=1 |
| Video | 5 | `(samples, frames, height, width, channels)` | 4 clips × 240 frames × 144×256×3 → 106,168,320 values ≈ 405 MB in float32 |

| Geometric operation | Tensor operation |
|---|---|
| Translation | add a vector |
| Rotation by θ | dot with `[[cos θ, -sin θ], [sin θ, cos θ]]` |
| Scaling | dot with a diagonal matrix `[[sx, 0], [0, sy]]` |
| Linear transform | dot with any matrix |
| Affine transform | `W·x + b` (Dense with no activation) |
| Non-linear fold | `relu(W·x + b)` |

Channels-first `(samples, channels, height, width)` is falling out of favor. Keras supports both conventions.

## Worked Example
**Backprop by hand on a scalar graph.** Take `x1 = w*x`, `x2 = x1 + b`, `loss_val = abs(y_true - x2)`, with `x=2, w=3, b=1, y_true=4`.
- Forward pass: `x1 = 6`, `x2 = 7`, `loss_val = 3`.
- Edge derivatives in the backward graph: `grad(loss_val, x2) = 1`, `grad(x2, x1) = 1`, `grad(x2, b) = 1`, `grad(x1, w) = x = 2`.
- Chain rule along the paths: `grad(loss_val, w) = 1·1·2 = 2` and `grad(loss_val, b) = 1·1 = 1`.
- The update moves `w` and `b` against these gradients, scaled by the learning rate.

**Counting updates.** With 60,000 samples and a batch size of 128, one epoch is 469 batches. Five epochs give 2,345 gradient updates.

## Key Takeaways
- Tensors are defined by rank, shape and dtype. Axis 0 is the samples or batch axis.
- Every deep learning transformation reduces to a handful of tensor operations: element-wise operations, broadcasting, dot and reshape. Each of them has a geometric meaning.
- Non-linear activations are what give stacked layers more expressive power than a single affine map.
- Learning means finding the weights that minimize the loss on the training data, by mini-batch SGD with gradients from backpropagation.
- You need two things before you feed data to a model: a **loss** (what success means) and an **optimizer** (how its gradient updates the weights).
- Autodiff (`GradientTape`) means you only write the forward pass.

## Connects To
- Chapter 1: this chapter makes the "three figures" (weights, loss, optimizer) concrete.
- Chapter 3: Keras and TensorFlow APIs (Variables, GradientTape, layers, `compile`/`fit`) in depth.
- Chapter 4: choosing the loss and the last-layer activation for classification and regression.
- Chapter 5: overfitting, generalization, and the manifold hypothesis (which builds on the "uncrumpling" picture).
- Chapter 7: custom training loops built on `one_training_step`.
- Chapter 9: residual connections and batch normalization as remedies for vanishing gradients.
