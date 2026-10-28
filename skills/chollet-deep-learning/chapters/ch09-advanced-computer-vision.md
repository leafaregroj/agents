# Chapter 9: Advanced Deep Learning for Computer Vision

## Core Idea
Image classification is only one of three essential computer vision tasks. The others are image segmentation and object detection. What separates the toy convnets of chapter 8 from the ones experts build is a small set of architecture patterns: residual connections, batch normalization, and depthwise separable convolutions. All of them are applications of the modularity-hierarchy-reuse (MHR) formula. The chapter also argues that convnets are "the opposite of black boxes." Their representations are visual, so three techniques can inspect them: plotting intermediate activations, visualizing filters via gradient ascent, and building class activation heatmaps (Grad-CAM).

## Frameworks Introduced
- **The three essential computer vision tasks**: image classification (one or more labels per image), image segmentation (a class per pixel), object detection (class-labeled bounding boxes). Most vision applications reduce to one of these.
  - When to use: as the first step in framing any vision problem. Ask what the output is: a label, a mask, or boxes.
  - How: single-label or multi-label classification uses a softmax or sigmoid head. Segmentation uses an encoder-decoder with a per-pixel softmax. Detection is specialized (e.g., RetinaNet) and is not covered in the book.
- **Modularity-hierarchy-reuse (MHR) formula**: to make a complex system simpler, structure it into modules, organize the modules into a hierarchy, and reuse modules where appropriate ("reuse" means abstraction here).
  - When to use: when designing any model architecture.
  - How: build the model from repeated blocks (e.g., VGG16's "conv, conv, max pooling"). Arrange them in a pyramid where the filter count grows as the feature-map size shrinks. Reuse weights where the data allows it (convolution reuses weights across spatial locations).
- **Residual connections**: `x = add([block(x), residual])`. Each block gets a noiseless information shortcut, so gradients survive great depth (ResNet, He et al., 2015).
  - When to use: on any network deep enough for vanishing gradients to appear, i.e., most nontrivial convnets.
  - How: save `residual = x`, run the block, then add. If shapes differ, project the residual with a 1x1 `Conv2D` with no activation. Add `strides=2` to that projection if the block contains max pooling. Use `padding="same"` inside the block.
- **Batch normalization**: normalizes intermediate activations using batch statistics during training and an exponential moving average of them at inference (Ioffe and Szegedy, 2015).
  - When to use: after Conv2D or Dense layers in deep networks. Some very deep networks only train with it.
  - How: `Conv2D(..., use_bias=False)`, then `BatchNormalization()`, then `Activation("relu")`, so the activation comes last.
- **Depthwise separable convolution** (`SeparableConv2D`): a per-channel spatial convolution followed by a pointwise (1x1) convolution that mixes channels. Its prior is that spatial locations are highly correlated while channels are highly independent.
  - When to use: as a drop-in replacement for Conv2D after the first layer. It matters most when training small models from scratch on limited data.
  - How: keep a regular `Conv2D` as the first layer, because RGB channels are highly correlated and break the assumption. Use `SeparableConv2D` from there on.
- **Ablation studies**: systematically remove parts of a system to find where its performance actually comes from.
  - When to use: before believing that any added complexity is necessary.
  - How: if X + Y + Z works, also try X, Y, Z, X + Y, X + Z, and Y + Z.

## Key Concepts
- **Semantic vs. instance segmentation**: semantic segmentation maps every cat pixel to a single "cat" class. Instance segmentation separates "cat 1" from "cat 2".
- **Segmentation mask**: the label for segmentation. It is an image the same size as the input, with one channel of integer class IDs. In Oxford-IIIT Pets the IDs are 1 = foreground, 2 = background, 3 = contour; subtract 1 to get 0..2.
- **Strided convolutions vs. max pooling**: 2x2 max pooling discards which location inside the window produced the value. Strided convolutions downsample while keeping location information. Use strides in any model that cares about feature location (segmentation, the generative models of chapter 12).
- **Conv2DTranspose**: "a kind of convolution layer that learns to upsample." `Conv2D(128, 3, strides=2, padding="same")` maps (100,100,64) to (50,50,128). `Conv2DTranspose(64, 3, strides=2, padding="same")` maps it back.
- **Hypothesis space / architecture as prior**: an architecture defines the functions gradient descent can search. A good one encodes prior knowledge and shrinks the search space, and "gradient descent is a pretty stupid search process, so it needs all the help it can get."
- **Vanishing gradients**: the "game of Telephone" analogy. Each function in a deep chain adds noise until the gradient signal drowns.
- **Bias redundancy under BN**: normalization re-centers outputs, so the preceding layer's bias vector does nothing. Pass `use_bias=False`.
- **Co-evolution of hardware, software, and algorithms**: cuDNN micro-optimizes standard convolutions. As a result, separable convolutions run only about as fast on GPU despite using quadratically fewer FLOPs, and leaving the beaten path is expensive.
- **Information distillation pipeline**: deeper activations carry less information about the specific input and more about the target. They become more abstract and sparser.
- **Gradient ascent in input space**: adjust the input image, not the weights, to maximize a filter's response.
- **Class activation map (CAM) / Grad-CAM**: weight each channel of the last conv feature map by the mean gradient of the class score with respect to that channel, then average over channels. The result is a heatmap of "how intensely the input image activates the class."
- **`model(x)` vs. `model.predict(x)`**: `predict()` batches, scales to large arrays, and returns NumPy, but it is not differentiable. Use `model(x)` only inside a `GradientTape` loop.

## Mental Models
- **Deep learning is MHR applied to continuous optimization**: layers are the modules, the stack is the hierarchy, and convolution is the reuse.
- **Deep and narrow beats broad and shallow**: depth encourages feature reuse and abstraction, but only up to the vanishing-gradient limit. Residuals and BN push that limit back.
- **"We know the how but not the why"**: there is no settled explanation for why batch normalization works (the "internal covariate shift" story is unproven). Treat it as an empirical best practice.
- **Stronger accurate priors give better models**: separable convs assume channel independence. That holds for learned representations but not for raw RGB.
- **Filters as a basis**: each layer learns a filter bank in which its inputs can be expressed, much as the Fourier transform decomposes signals onto cosines. Early layers learn edges and colors, middle layers learn textures, and deep layers learn natural-image textures such as feathers, eyes, and leaves.
- **The bicycle-drawing effect**: like a deep network, the brain abstracts away visual detail, so people cannot draw a bicycle from memory.

## Anti-patterns
- Using MaxPooling2D to downsample in segmentation models, which destroys location information.
- Putting the activation before BatchNormalization, as in `Conv2D(32, 3, activation="relu")` followed by `BatchNormalization()`. The author calls the ordering "not exactly critical" but recommends BN before ReLU.
- Leaving BatchNormalization layers trainable during fine-tuning. Their moving statistics interfere with the small updates to the surrounding layers, so freeze them.
- Using SeparableConv2D as the very first layer on RGB input.
- Adding complexity without ablation, i.e., papers "optimized for peer review" that use math "as a signal of seriousness, like an expensive suit on a salesman."
- Calling `predict()` inside a GradientTape when you need gradients.
- Expecting separable convs to train faster on GPU. They do not, but they still overfit less and converge faster per sample.

## Code Examples
```python
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    return layers.add([x, residual])

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1./255)(inputs)
x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)  # GAP follows; no pooling needed
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
```
- **What it demonstrates**: a reusable residual block that handles both kinds of shape mismatch: a filter-count change (1x1 projection) and spatial downsampling (strided 1x1 projection).

```python
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)  # regular conv for RGB
for size in [32, 64, 128, 256, 512]:
    residual = x
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
```
- **What it demonstrates**: the mini Xception-like model, which combines all the patterns: blocks, a filter pyramid, BN placed before ReLU, separable convs, residuals, GlobalAveragePooling instead of Flatten, and dropout.

```python
import tensorflow as tf
with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
grads = tape.gradient(top_class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()   # channel importance
last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
heatmap = np.mean(last_conv_layer_output, axis=-1)
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)               # normalize to [0, 1]
```
- **What it demonstrates**: the core of Grad-CAM. The model is split into a "last conv layer" model and a classifier model, the gradient is taken with respect to the conv feature map, and the channels are weighted by importance.

## Reference Tables
| Pattern | Problem solved | Keras recipe | Caveat |
|---|---|---|---|
| Residual connection | Vanishing gradients in deep stacks | `layers.add([block_out, residual])` | 1x1 conv projection on shape mismatch; `strides=2` when pooling |
| Batch normalization | Gradient propagation, deeper nets | `Conv2D(use_bias=False)`, then `BatchNormalization()`, then `Activation("relu")` | Freeze during fine-tuning |
| SeparableConv2D | Parameter and FLOP efficiency, overfitting | Replace `Conv2D` after the first layer | No GPU speedup (cuDNN); not for raw RGB |
| Strided conv | Downsampling while keeping location | `Conv2D(..., strides=2, padding="same")` | Prefer over max pooling for segmentation and generation |
| Conv2DTranspose | Learned upsampling | `Conv2DTranspose(..., strides=2, padding="same")` | Mirror the encoder's downsampling |

| Interpretation technique | Question answered | Model used |
|---|---|---|
| Intermediate activations | How do layers transform an input? | Multi-output model over all Conv/Pool layer outputs |
| Filter visualization | What pattern does a filter respond to? | Feature extractor plus gradient ascent on the input |
| Grad-CAM heatmap | Which image regions drove the class decision, and where is the object? | Full classifier with top included |

Parameter count, 3x3 window, 64 input and 64 output channels: regular conv = 3·3·64·64 = 36,864; depthwise separable = 3·3·64 + 64·64 = 4,672.

## Worked Example
**Oxford-IIIT Pets semantic segmentation** (7,390 images; 1,000 held out for validation; images resized to 200x200).
1. Shuffle the input and mask path lists with the same seed (1337) so they stay aligned. Load the masks with `color_mode="grayscale"` and subtract 1.
2. Encoder: `Rescaling(1/255)`, then Conv2D layers at 64, 128, and 256 filters, with every other layer at `strides=2`, ending at (25, 25, 256). Use `padding="same"` everywhere.
3. Decoder: mirrored `Conv2DTranspose` layers (256, 128, 64; every other one at `strides=2`) back to 200x200, then `Conv2D(num_classes=3, 3, activation="softmax", padding="same")`. The model has 2.88M parameters.
4. Compile with `rmsprop` and `sparse_categorical_crossentropy`. Train for 50 epochs at batch size 64 with `ModelCheckpoint(save_best_only=True)`. Overfitting starts around epoch 25, and the checkpoint keeps the best model.
5. Predict, take `argmax(axis=-1)` per pixel, and display the result.

**Architecture payoff**: on dogs vs. cats, the mini Xception reaches 90.8% test accuracy, compared with 83.5% for the naive chapter-8 convnet, with 721,857 parameters versus 991,041. Architecture best practices alone deliver a sizable gain before any hyperparameter tuning.

**Grad-CAM on elephants**: Xception predicts African elephant (87%), tusker (7%), and Indian elephant (2%), with class index 386. The heatmap from `block14_sepconv2_act` lights up the calf's ears, which is probably the cue that separates African from Indian elephants.

## Key Takeaways
- Frame vision problems as classification, segmentation, or detection.
- For segmentation, use an encoder of strided convs and a decoder of Conv2DTranspose layers, ending in a per-pixel softmax over the classes.
- Build convnets from repeated blocks. Let the filter count grow as the feature maps shrink, and prefer deep and narrow.
- Wrap blocks in residual connections. Add BN after convs (before the activation, with `use_bias=False`). Use SeparableConv2D after the first layer.
- Use GlobalAveragePooling2D plus Dropout instead of Flatten at the head.
- Freeze BN layers when fine-tuning.
- Run ablation studies before trusting complexity.
- Use activation grids, filter visualizations, and Grad-CAM to debug and localize.

## Connects To
- Chapter 8: convolution strides, the dogs-vs-cats data pipeline and augmentation, and pretrained Xception and VGG16.
- Chapter 7: multi-output Functional models (activation model, feature extractor) and GradientTape loops.
- Chapter 2: low-level gradient descent with GradientTape.
- Chapter 10: the LSTM carry track is "pretty much the same idea" as residual connections.
- Chapter 12: strided convs instead of pooling in generative models. DeepDream and style transfer build on filter visualization by gradient ascent.
- Chapter 13: systematic hyperparameter tuning as the next step after architecture best practices.
