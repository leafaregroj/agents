---
name: chollet-deep-learning
description: "Knowledge base from \"Deep Learning with Python\" (2nd ed.) by François Chollet. Use when applying Chollet's frameworks for the universal workflow of machine learning, overfitting and generalization, Keras/TensorFlow model building, convnets and transfer learning, timeseries RNNs, text models and Transformers, generative models (VAE, GAN, style transfer), or hyperparameter tuning and scaling, studying the book, or referencing its concepts and Keras code."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Deep Learning with Python (2nd ed.)
**Author**: François Chollet | **Pages**: ~499 | **Chapters**: 14 | **Generated**: 2026-10-28

## How to Use This Skill

- **Without arguments**: load the core frameworks below
- **With a topic**: e.g. `fine-tuning`, `residual connections`, `Transformer`, `recurrent dropout`, `GAN`. I find and read the relevant chapter
- **With a chapter**: e.g. `ch09`. I load that chapter, including its Keras code
- **Browse**: ask "what chapters do you have?"

For a topic not covered in Core Frameworks, read the relevant `chapters/` file before answering. For defaults and quick decisions, use [cheatsheet.md](cheatsheet.md).

---

## Core Frameworks & Mental Models

### 1. Deep learning is learning representations
Classical programming turns rules + data into answers. ML turns data + answers into rules. Learning means **searching a hypothesis space for useful representations, guided by a feedback signal**. A deep network is a chain of layers of geometric transformations, each weight-parameterized. The loss scores predictions, and the optimizer uses backpropagation to nudge weights. Think of it as **uncrumpling a paper ball**. Use DL for perceptual data (images, sound, text). For tabular data, prefer gradient boosted trees. (Ch 1, 2)

### 2. The manifold hypothesis and interpolation
Natural data lies on low-dimensional **latent manifolds**. DL generalizes by **interpolating** between training points on the learned manifold, which is local generalization. It cannot achieve **extreme generalization** (abstraction and reasoning about real novelty). So **dense sampling of the input space is the best lever**, and more or better data beats cleverer models. (Ch 5, 14)

### 3. Optimization vs. generalization: overfit first, then regularize
- Training always trades fit against generalization. "To achieve the perfect fit, you must first overfit."
- **Phase 1, statistical power**: beat a common-sense baseline.
- **Phase 2**: scale up until validation loss turns.
- **Phase 3, regularize and tune**, in order: more or better data → better features → early stopping → less capacity → L1/L2 (small models) → dropout (large models).
- Tuning on validation **leaks information**, so test exactly once.
- Use K-fold or iterated K-fold when data is scarce. (Ch 4, 5)

### 4. The universal workflow of machine learning
1. **Define the task**: frame inputs and targets, collect and annotate data, understand it, and choose a success metric.
2. **Develop a model**: prepare tensors, pick an evaluation protocol, beat the baseline, scale up, regularize.
3. **Deploy the model**: explain results to stakeholders in FP/FN terms, ship it (REST API, on-device, or browser), monitor it, and maintain it.

The cardinal sins are **non-representative data**, **sampling bias** and **target leaking**. Plan for **concept drift**. (Ch 6)

### 5. Last layer, loss and evaluation by task
| Task | Last layer | Loss |
|---|---|---|
| Binary classification | `Dense(1, "sigmoid")` | `binary_crossentropy` |
| Single-label N-class | `Dense(N, "softmax")` | `(sparse_)categorical_crossentropy` |
| Multilabel | `Dense(N, "sigmoid")` | `binary_crossentropy` |
| Scalar regression | `Dense(1)` | `mse` (monitor `mae`) |

Normalize features with **training statistics only**. Avoid intermediate layers narrower than the number of classes, because they create an information bottleneck. (Ch 4, 6)

### 6. Keras: progressive disclosure of complexity
Keras offers a **spectrum of workflows**. Pick the simplest that works:
- **Model building**: `Sequential` → **Functional API** (the default: inspectable, multi-input and multi-output, plottable, good for feature extraction) → **subclassing** (arbitrary `call()`, but a black box).
- **Training**: `fit()` → custom metrics and callbacks (EarlyStopping + ModelCheckpoint) → override `train_step()` → a from-scratch GradientTape loop under `@tf.function`.
- **Layers**: create state in `build()` and compute in `call()`. The model topology *is* the hypothesis space. (Ch 3, 7)

### 7. Convnets: modularity, hierarchy, reuse
- **Why convnets**: they learn **translation-invariant**, **spatially hierarchical** patterns, from edges to parts to objects.
- **Small-data toolbox**: from scratch with augmentation + dropout → **feature extraction** on a frozen pretrained base → **fine-tuning** of the top layers at a low learning rate.
- **Modern blocks**: **residual connections** fight vanishing gradients; **BatchNorm** goes between conv and activation (freeze it when fine-tuning); **depthwise separable convolutions** cut parameters about 8×.
- **Scaling**: deep and narrow beats broad and shallow.
- **Location-sensitive tasks**: use strides, not max pooling.
- **Interpretability**: intermediate activations, filter visualization and **Grad-CAM**. (Ch 8, 9)

### 8. Timeseries: baseline first, recurrence second
- Split chronologically, and never let validation or test data be older than training data.
- Compute a **common-sense baseline** first ("same as 24 h ago": MAE 2.62), then a small dense model, then LSTM or GRU.
- Fight overfitting with **recurrent dropout** (and double the units), then stack layers.
- Bidirectional RNNs suit text, not forecasting where recent data matters most.
- About 10% better than baseline is realistic. Don't apply this to markets. (Ch 10)

### 9. Text: sets vs. sequences, then Transformers
- **Pipeline**: standardize → tokenize → index → encode (`TextVectorization`, run on CPU inside `tf.data`).
- **Rule of thumb**: if samples ÷ mean words per sample < **1,500**, a **bag-of-bigrams** model wins. Above that, use a **sequence model**.
- **Attention (query–key–value)** builds context-aware tokens, and multi-head attention splits them into subspaces. A Transformer needs **positional embeddings** to be order-aware.
- **Seq2seq**: an encoder plus a decoder with a **causal mask**, trained with targets offset by one step and decoded token by token at inference. (Ch 11)

### 10. Generative deep learning
Generation is sampling from a learned latent space:
- **Text**: a language model with **softmax temperature** sampling. Sweep the temperature.
- **DeepDream**: runs a convnet in reverse, by gradient ascent over octaves.
- **Neural style transfer**: content loss + Gram-matrix style loss + total variation loss.
- **VAEs**: give structured, continuous latent spaces with **concept vectors**, good for editing.
- **GANs**: a generator vs. a discriminator. They give realism but are unstable; use the bag of tricks. (Ch 12)

### 11. Real-world best practices
- **Automate hyperparameter search** (KerasTuner, BayesianOptimization) over a well-crafted search space. It is "automation, not magic", and it overfits validation.
- **Ensemble diverse models**: "diversity is strength".
- **Scaling**: mixed precision, then MirroredStrategy for multi-GPU, then TPUStrategy with step fusing. (Ch 13)

### 12. Chollet's view of the limits of AI
Current DL is **cognitive automation**, not intelligence. Intelligence is the efficiency of turning experience and priors into new skills (ARC). The **shortcut rule** says that optimizing a single metric sacrifices everything else. The future is **hybrid**: DL modules plus program synthesis, built on reusable abstractions (the kaleidoscope hypothesis). (Ch 1, 14)

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-what-is-deep-learning.md) | What is deep learning? | Learning representations; DL in three figures; history; why now |
| [ch02](chapters/ch02-mathematical-building-blocks.md) | The mathematical building blocks of neural networks | Tensors; tensor ops; gradient descent; backprop; GradientTape |
| [ch03](chapters/ch03-intro-to-keras-and-tensorflow.md) | Introduction to Keras and TensorFlow | Variables; GradientTape; Layer build/call; compile/fit/evaluate |
| [ch04](chapters/ch04-classification-and-regression.md) | Getting started with neural networks: Classification and regression | IMDB, Reuters, Boston Housing; last-layer/loss table; K-fold |
| [ch05](chapters/ch05-fundamentals-of-ml.md) | Fundamentals of machine learning | Manifold hypothesis; evaluation protocols; overfit-then-regularize |
| [ch06](chapters/ch06-universal-workflow.md) | The universal workflow of machine learning | Define → develop → deploy; leaks and drift; deployment options |
| [ch07](chapters/ch07-working-with-keras.md) | Working with Keras: A deep dive | Sequential / Functional / subclassing; callbacks; custom train_step |
| [ch08](chapters/ch08-intro-to-computer-vision.md) | Introduction to deep learning for computer vision | Convolution and pooling; augmentation; feature extraction; fine-tuning |
| [ch09](chapters/ch09-advanced-computer-vision.md) | Advanced deep learning for computer vision | Segmentation; residuals; BatchNorm; separable convs; Grad-CAM |
| [ch10](chapters/ch10-deep-learning-for-timeseries.md) | Deep learning for timeseries | Baselines; LSTM/GRU; recurrent dropout; stacking; bidirectional |
| [ch11](chapters/ch11-deep-learning-for-text.md) | Deep learning for text | TextVectorization; bag-of-words vs sequences; embeddings; Transformer; seq2seq |
| [ch12](chapters/ch12-generative-deep-learning.md) | Generative deep learning | Temperature sampling; DeepDream; style transfer; VAE; GAN |
| [ch13](chapters/ch13-best-practices-real-world.md) | Best practices for the real world | KerasTuner; ensembling; mixed precision; multi-GPU; TPU |
| [ch14](chapters/ch14-conclusions.md) | Conclusions | Recap templates; limits of DL; extreme generalization; future of AI |

## Topic Index

- **Attention / Transformer / positional embedding** → ch11, ch12, ch14
- **Backpropagation / gradient descent / GradientTape** → ch02, ch03, ch07
- **Baselines (common-sense)** → ch05, ch06, ch10
- **BatchNorm / residual connections / separable convs** → ch09
- **Callbacks / EarlyStopping / TensorBoard** → ch07
- **Concept drift / target leaking / sampling bias** → ch06
- **Convnets / pooling / strides** → ch08, ch09
- **Custom training loops / train_step** → ch07
- **Data augmentation** → ch08
- **Deployment (REST, TF Lite, TF.js) / quantization** → ch06
- **DeepDream / style transfer** → ch12
- **Dropout / L1–L2 / regularization** → ch05, ch10
- **Embeddings (word, GloVe)** → ch11
- **Feature extraction / fine-tuning / transfer learning** → ch08
- **Functional API / subclassing / Sequential** → ch07
- **GAN / VAE / latent space** → ch12
- **Grad-CAM / filter visualization** → ch09
- **Hyperparameter tuning / KerasTuner / ensembling** → ch13
- **K-fold validation / holdout** → ch04, ch05
- **Limits of DL / ARC / extreme generalization** → ch05, ch14
- **Manifold hypothesis / interpolation** → ch05, ch14
- **Mixed precision / multi-GPU / TPU** → ch13
- **Loss and last-layer choice** → ch04, ch06, ch14
- **RNN / LSTM / GRU / recurrent dropout** → ch10, ch11
- **Segmentation** → ch09
- **Seq2seq / translation** → ch11
- **Temperature sampling / text generation** → ch12
- **Tensors / broadcasting** → ch02
- **Text vectorization / bag-of-words / N-grams** → ch11

## Supporting Files

- [glossary.md](glossary.md): all key terms with definitions
- [patterns.md](patterns.md): recipes (workflow, vision, timeseries, text, generative, tuning, scaling)
- [cheatsheet.md](cheatsheet.md): model choice, defaults, decision rules, warning signs

---

## Scope & Limits

This skill covers the book's content (2nd ed., 2021, TensorFlow 2.6 / Keras era). Code snippets were rebuilt from a PDF text extraction in which listings and callouts were interleaved, and known book typos were fixed. Keras 3 and newer TensorFlow have changed some APIs since (e.g. the `kerastuner` package is now `keras_tuner`), so check them against current docs before production use.
