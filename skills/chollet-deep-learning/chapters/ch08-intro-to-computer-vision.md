# Chapter 8: Introduction to Deep Learning for Computer Vision

## Core Idea
Convolutional neural networks (**convnets**) dominate computer vision because their architecture matches two structural properties of the visual world: it is **translation-invariant** and **spatially hierarchical**. Convolutions learn local patterns that can be recognized anywhere; stacked convolution + max-pooling stages compose them into increasingly abstract concepts over increasingly large windows. On the most common real-world setting, a *small* image dataset, overfitting is the central problem, and the toolbox is: train a small convnet from scratch with **data augmentation**, then do **feature extraction** with a pretrained model, then **fine-tuning**. On a 2,000-image dogs-vs-cats task these climb from about 70% to 83.5% to 97.5% to 98.5% test accuracy.

## Frameworks Introduced
- **Small-data image classification toolbox** (three strategies, in increasing power):
  1. **Train a small convnet from scratch** + data augmentation + dropout.
     - When to use: no suitable pretrained model, or as a baseline.
     - How: alternating `Conv2D(relu)`/`MaxPooling2D` stages, `Rescaling(1./255)` input, `Flatten`, `Dropout(0.5)`, sigmoid/softmax head.
  2. **Feature extraction with a pretrained model**: reuse the **convolutional base** of a model trained on a large dataset (e.g., ImageNet), train a new classifier on its outputs.
     - When to use: small dataset, a large generic pretrained model exists.
     - How (fast, no augmentation): run `conv_base.predict()` once over the data, save features as NumPy arrays, train a dense classifier on them. How (end to end, with augmentation): freeze the base (`trainable = False`), chain augmentation -> preprocessing -> base -> dense head, train on images (needs a GPU).
  3. **Fine-tuning**: unfreeze a few top layers of the base and jointly train them with the new head at a very low learning rate.
     - When to use: after feature extraction has converged, to adapt the more abstract representations to the new task.
     - How (5 steps): add custom head on trained base -> freeze base -> train head -> unfreeze some top base layers (never `BatchNormalization` layers) -> jointly train with a low learning rate.
- **Why downsample**: max pooling both (a) reduces the number of feature-map coefficients (fights overfitting and cost) and (b) induces spatial-filter hierarchies by making later convolutions see larger fractions of the input.

## Key Concepts
- **Dense vs convolution**: Dense layers learn global patterns over the whole input; convolution layers learn local patterns in small 2D windows (typically 3x3).
- **Translation invariance**: a pattern learned in one location is recognized anywhere, making convnets data-efficient on perceptual problems.
- **Spatial hierarchy of patterns**: early layers learn edges and textures, later layers combine them into parts (eyes, ears) and then concepts ("cat").
- **Feature map**: a rank-3 tensor (height, width, channels/depth). After the first layer, channels stand for **filters**, not colors.
- **Filter / response map**: each output channel is a 2D map of where that filter's pattern occurs in the input.
- **Convolution kernel**: one learned weight matrix applied to every extracted patch of shape (window_h, window_w, input_depth), producing an output vector of size output_depth at each location. Key parameters: patch size (3x3 or 5x5) and output depth (`Conv2D(output_depth, kernel_size)`).
- **Border effects and padding**: without padding (`padding="valid"`, the default) a 3x3 conv shrinks each spatial dimension by 2 (28 -> 26). `padding="same"` pads so output matches input size.
- **Strides**: distance between successive windows (default 1). Stride 2 downsamples by 2. Strided convs are rare in classifiers but used in some models (next chapter).
- **Max pooling**: takes the max of each channel over (usually) 2x2 windows with stride 2, halving feature maps. Like convolution but with a hardcoded max instead of a learned transform. Beats average pooling and strided convs because max presence of a feature is more informative than its average presence.
- **Rescaling layer**: maps [0, 255] pixel values to [0, 1] inside the model.
- **`image_dataset_from_directory()`**: turns a directory with one subfolder per class into a shuffled, resized, batched `tf.data.Dataset` of (images, integer labels).
- **`tf.data.Dataset`**: iterable input pipeline with `.batch()`, `.shuffle()`, `.prefetch()` (async prefetching), `.map()`, `.take(N)`; pass directly to `fit()`.
- **Data augmentation**: random, believable transformations (`RandomFlip`, `RandomRotation`, `RandomZoom`) applied at training time so the model never sees the same image twice. Like dropout, these layers are inactive at inference.
- **Pretrained model / feature reuse**: features learned on a large, general dataset act as a generic model of the visual world and transfer across problems, one of deep learning's greatest strengths.
- **Convolutional base**: the conv + pooling part of a convnet; reused because its features are generic presence maps that retain spatial information. The dense classifier is task-specific and discards location, so it is generally not reused.
- **Freezing**: setting `trainable = False` empties a layer's trainable weights so they are not updated. Changes take effect only after `compile()`; recompile after changing trainability.
- **`preprocess_input`**: model-specific input scaling (e.g., `keras.applications.vgg16.preprocess_input`) that must be applied before a pretrained base.

## Mental Models
- **Depth up, size down**: in almost all convnets, feature-map depth grows (32 -> 256) while spatial size shrinks (180x180 -> 7x7).
- **Generality decreases with depth**: early layers are generic (edges, colors, textures), late layers are specialized ("dog eye"). If the new dataset differs a lot from the original, extract features from earlier layers only; when fine-tuning, adjust only the top, most specialized layers.
- **Augmentation remixes, it does not create**: augmented images remain highly intercorrelated, so augmentation alone may not eliminate overfitting; combine with dropout.
- **Large random gradients destroy pretrained representations**: a randomly initialized head propagates large error signals, which is why you freeze the base first and fine-tune only after the head is trained.

## Anti-patterns
- Removing max-pooling layers: three unpooled 3x3 convs only see 7x7 input windows (no global view), and flattening a 22x22x128 map gives 61,952 coefficients and a 619,530-parameter Dense layer, causing intense overfitting.
- Reusing the pretrained dense classifier when your classes differ or object location matters.
- Training a new head on an unfrozen pretrained base (destroys learned representations).
- Fine-tuning before the classifier on top is trained.
- Fine-tuning the entire base on a small dataset: diminishing returns from lower layers and high overfitting risk with ~15 million parameters.
- Changing `trainable` after `compile()` without recompiling.
- Fine-tuning with a normal learning rate; use a very low one (1e-5) to limit representation changes.
- Fast feature extraction (precomputed features) and expecting augmentation benefits; it overfits almost from the start.
- Trusting validation accuracy as the final number: feature extraction with augmentation hit over 98% validation but 97.5% test; results depend on the sample set evaluated.

## Code Examples
```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),   # fraction of a full circle: +/-36 degrees
    layers.RandomZoom(0.2),
])
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
for filters in (32, 64, 128, 256):
    x = layers.Conv2D(filters, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy", optimizer="rmsprop",
              metrics=["accuracy"])
callbacks = [keras.callbacks.ModelCheckpoint(
    "convnet_from_scratch_with_augmentation.keras",
    save_best_only=True, monitor="val_loss")]
model.fit(train_dataset, epochs=100, validation_data=validation_dataset,
          callbacks=callbacks)
```
- **What it demonstrates**: the from-scratch small-data recipe (loop is a compact rewrite of the book's explicit layer stack); reaches 83.5% test accuracy.

```python
conv_base = keras.applications.vgg16.VGG16(weights="imagenet",
                                           include_top=False)
conv_base.trainable = False
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy", optimizer="rmsprop",
              metrics=["accuracy"])
model.fit(train_dataset, epochs=50, validation_data=validation_dataset)

# Fine-tuning: unfreeze only the last conv block, then retrain with a tiny LR
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False
model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])
model.fit(train_dataset, epochs=30, validation_data=validation_dataset)
```
- **What it demonstrates**: feature extraction with augmentation on a frozen base (97.5% test), then fine-tuning block5 (98.5% test). Note the recompile after each trainability change.

## Reference Tables
| Strategy | Test acc. (dogs vs cats, 2k train imgs) | Cost | Augmentation? |
|---|---|---|---|
| Small convnet from scratch | 69.5% (val peaks ~75%, overfits by epoch ~10) | low | no |
| + augmentation + dropout | 83.5% (val 80-85%, overfits ~epoch 60-70) | low | yes |
| Fast feature extraction (VGG16 features to NumPy) | val ~97%, overfits immediately | very low (<1 s/epoch on CPU) | no |
| Feature extraction end to end, frozen base | 97.5% (val >98%) | high, GPU only | yes |
| Fine-tuning top 3 conv layers, LR 1e-5 | 98.5% | high | yes |

| Default | Value |
|---|---|
| Conv window | 3x3, stride 1 |
| Pooling | 2x2 max, stride 2 |
| `Conv2D` padding default | `"valid"` |
| Binary head | `Dense(1, activation="sigmoid")` + `binary_crossentropy` |
| Integer labels, multiclass | `sparse_categorical_crossentropy` |
| Batch size / image size in example | 32 / 180x180 |
| Dropout before head | 0.5 |
| Fine-tuning LR | 1e-5 (RMSprop) |

MNIST: simple convnet (104,202 params, 5 epochs, batch 64) reaches 99.1% test vs 97.8% for the dense net of chapter 2, about a 60% relative error reduction. VGG16 base: 14,714,688 params, output (5, 5, 512) for 180x180 input; 26 trainable weight tensors before freezing, 0 after. Other `keras.applications` models: Xception, ResNet, MobileNet, EfficientNet, DenseNet.

## Worked Example
Dogs vs. Cats (Kaggle 2013, 25,000 images) subsampled to a balanced set: 1,000 train / 500 validation / 1,000 test images per class, so accuracy is an appropriate metric. Images loaded with `image_dataset_from_directory(..., image_size=(180, 180), batch_size=32)`. A 5-conv-layer model (991,041 params) with `ModelCheckpoint(save_best_only=True, monitor="val_loss")` overfits fast: training accuracy approaches 100% while validation loss bottoms out near epoch 10; reloading the best checkpoint gives 69.5% test. Adding augmentation + dropout and training 100 epochs gives 83.5%; further tuning might reach ~90%, but more requires pretrained features. VGG16 feature extraction reaches 97.5%, and fine-tuning its last block reaches 98.5%, using about 10% of the competition's data (2,000 vs 20,000 samples). Caveat: ImageNet already contains cat and dog classes, which flatters the pretrained results.

## Key Takeaways
- Convnets are the default for vision; their inductive biases (locality, translation invariance, hierarchy) make them data-efficient.
- Stack conv + max-pool stages: depth increases, spatial size decreases; pooling keeps parameter counts sane and builds hierarchy.
- On small image datasets overfitting is the main issue; data augmentation is the vision-specific remedy, used almost universally.
- Reuse pretrained convolutional bases, not their classifiers. Freeze before training a new head.
- Fine-tune only the top few layers, only after the head is trained, with a very low learning rate, and recompile after any trainability change.
- Always checkpoint on `val_loss` and evaluate the best saved model on a held-out test set.

## Connects To
- Chapter 2 (dense MNIST baseline) and Chapter 4 (dense classifiers on top of features).
- Chapter 5 (overfitting, regularization, dropout) and Chapter 6 (choosing loss functions).
- Chapter 7 (Functional API, `ModelCheckpoint`, freezing via `trainable` and trainable weights).
- Chapter 9 (strided convolutions, batch normalization and fine-tuning, residual connections, depthwise separable convs, interpreting convnets using the saved augmentation model).
