# Chapter 14: Conclusions

## Core Idea
Deep learning is a simple idea: large parametric models trained with gradient descent on many examples. They learn chains of differentiable geometric transformations that map an input vector space onto a target space, which lets them "uncrumple" data manifolds. That makes deep learning very good at **local generalization** (interpolating on a learned manifold) and **value-centric abstraction** (perception, intuition). It has essentially no ability for **program-centric abstraction** (reasoning, planning), which **extreme generalization** requires. Use deep learning where the manifold hypothesis holds and data is plentiful. Don't expect it to reason, and don't anthropomorphize it. Chollet expects the field to move toward hybrid systems that combine deep learning with **program synthesis**, guided by learned intuition and built from a library of reusable subroutines.

## Frameworks Introduced
- **The universal machine learning workflow (recap)**: (1) define the problem and the data; (2) choose a reliable measure of success; (3) set up the validation process with train/val/test sets and no leakage (for temporal data, val/test come after train); (4) vectorize and normalize; (5) beat a common-sense baseline; (6) overfit first, then regularize and tune using validation data only, watching for validation-set overfitting; (7) deploy and monitor.
  - When to use: on every new project. The hard part is everything before and after model design.
- **Input modality → architecture map**: a network architecture encodes assumptions about the data's structure (a hypothesis space). It works only when those assumptions match the data.
  - How: vector → Dense; image → 2D convnet; timeseries → RNN; discrete sequences (text) → Transformer; continuous translation-invariant sequences (e.g., birdsong) → 1D convnet; video → 3D convnet, or frame-level 2D convnet + sequence model; volumetric → 3D convnet.
- **Spectrum of generalization**: no generalization (a hardcoded dict or if-then-else) → local generalization (ML/deep learning: "known unknowns") → broad generalization ("unknown unknowns" within one domain, e.g., a fully robust self-driving car, the Woz coffee test) → extreme generalization (humans: radically new situations with little or no new data).
  - When to use: to judge what a system can realistically handle, and to evaluate claims about AI.
- **Intelligence as an efficiency ratio**: intelligence is the conversion ratio between the relevant information available (experience plus priors) and the **future operating area** (the set of new situations handled well).
  - How to measure: fix the system's experience and priors, then test it only on tasks it wasn't trained or programmed for, ideally tasks its creators couldn't anticipate. ARC (Abstraction and Reasoning Corpus) implements this with Core Knowledge priors.
- **The two poles of abstraction**: value-centric analogy (similarity by distance → prototypes) and program-centric analogy (exact structural match → isomorphic subprograms). Cognition interleaves both.
  - When to use: to decide between deep learning and discrete programs/search for a given (sub)problem.

## Key Concepts
- **AI ⊃ machine learning ⊃ deep learning**: AI is "all attempts to automate human cognitive processes." ML builds programs (models) from training data. Deep learning uses models that are long chains of geometric transformations (layers) parameterized by weights.
- **Everything is a vector**: inputs and targets are vectorized, and each layer is a simple geometric transform. The chain must be differentiable, which means a smooth, continuous morphing, and that is a significant constraint.
- **Meaning as pairwise relationships captured by a distance function**: the core assumption behind deep learning. Chollet calls "neural network" a misleading name. "Layered representations learning," "hierarchical representations learning," "deep differentiable models," or "chained geometric transforms" would be more accurate.
- **Key enabling technologies**: incremental algorithmic innovations (starting with backpropagation), large perceptual datasets, cheap parallel hardware (NVIDIA GPUs), and a software stack (CUDA, TensorFlow autodiff, Keras).
- **Limits of deep learning**: reasoning, long-term planning, and algorithmic data manipulation are out of reach "no matter how much data you throw at them." Most programs cannot be expressed as continuous geometric morphing of a data manifold, and scaling up only superficially helps.
- **Anthropomorphizing risk**: people project understanding through theory of mind. Adversarial examples (a panda plus a "gibbon gradient" is classified as a gibbon) and texture shortcuts (a leopard-print sofa is classified as a leopard) show that models map inputs to targets without understanding.
- **Automaton vs. intelligent agent**: an automaton runs fixed "if this, then that" behavioral programs. An intelligent agent adapts on the fly. Trained deep nets are automatons, as Descartes' 1637 remark anticipates.
- **Local generalization**: generalization by interpolation on a learned manifold. It works only for factors of variation that appear densely in the training data, which is why data augmentation helps.
- **Extreme generalization**: adapting to never-before-seen situations using abstraction and reasoning, with little or no data.
- **Cognitive automation vs. "Artificial Intelligence"**: today's AI is cognitive automation. Chollet calls the "intelligence" label a category error and proposes "Artificial Cognition," with Cognitive Automation and Artificial Intelligence as two nearly independent subfields.
- **The shortcut rule**: if you optimize one success metric, you reach it at the expense of everything the metric doesn't cover. Examples: the Netflix Prize winner was never deployed; Deep Blue taught little about cognition.
- **Kaleidoscope hypothesis**: the world looks endlessly new, but it is built from a small number of "atoms of meaning" recombined. Abstractions are those atoms, and analogy-making is how you extract them.
- **Intelligence as sensitivity to abstract analogies**: "Abstraction is the engine of intelligence, and analogy-making is the engine that produces abstraction."
- **Program synthesis**: discrete search (e.g., genetic programming) over a vocabulary of building blocks for a program that satisfies an input-output specification. It is data-efficient but faces combinatorial explosion.
- **Hybrid systems**: AlphaGo (hand-written Monte Carlo Tree Search plus learned value and policy networks) and self-driving cars (a hardcoded 3D world model updated by deep learning perception). RNNs are an early step: geometric transforms inside a hardcoded for loop.
- **Lifelong learning / modular subroutine reuse**: multitask training improves each task (e.g., English-German plus French-Italian translation; classification plus segmentation sharing a conv base). Eventually, meta-learners will draw on a global library of geometric and algorithmic subroutines.

## Mental Models
- **Uncrumpling a paper ball**: the crumpled ball is the input data manifold, and each layer is one move that flattens it. "Deep learning models are mathematical machines for uncrumpling complicated manifolds of high-dimensional data."
- **Layers as LEGO bricks for information processing**: plug architectures together into multimodal models that map almost anything to anything, given data and a continuous mapping of reasonable complexity.
- **A dim image in a mirror**: real world → embodied human experience → abstract concepts → labeled data → model. Each step loses fidelity. The model matches the training data, not the human mental model behind it.
- **Memorizing student vs. understanding student**: you tell them apart by giving a brand-new problem.
- **Rocket to the moon**: a deep net needs dense sampling (millions of launch trials). Humans derive physics and solve it in one or a few tries.
- **Two poles, one mind**: perception involves implicit reasoning, and a mathematician's proof is guided by intuition. Deep learning has only the value-centric half.

## Anti-patterns
- **Believing a model "understands" its task**: models break absurdly on anything that deviates from the training distribution.
- **Training a deep net for algorithmic tasks**: sorting five numbers is "an exercise in frustration" that needs massive data, still makes errors, and must be retrained for lists of ten. A few lines of Python generalize to any length.
- **Hand-coding perception**: a pure-Python MNIST classifier takes thousands of lines for about 90% accuracy. "If you have lots of data and you're faced with a problem where the manifold hypothesis applies, go with deep learning."
- **Fixing the task as the goal of AI research**: by the shortcut rule, you get a skilled but non-intelligent system. Skill can be "bought" with data or hardcoded rules without adding generalization. With enough data, even nearest-neighbor can play games at a superhuman level.
- **Optimizing a single competition metric for production**: Kaggle and Netflix-style winners ignore inference cost, maintainability, and explainability.
- **Expecting scale alone to solve reasoning**: more layers and data only "superficially palliate" the problem.

## Code Examples
```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(num_input_features,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
# Binary:            Dense(1, "sigmoid")                 + "binary_crossentropy" (targets 0/1)
# Single-label:      Dense(num_classes, "softmax")       + "categorical_crossentropy" (one-hot)
#                                                         or "sparse_categorical_crossentropy" (ints)
# Multilabel:        Dense(num_classes, "sigmoid")       + "binary_crossentropy" (multi-hot)
# Regression:        Dense(num_values)  (no activation)  + "mse"
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```
- **What it demonstrates**: the last-layer activation and loss pairing for each task type on vector data.

```python
inputs = keras.Input(shape=(height, width, channels))
x = layers.SeparableConv2D(32, 3, activation="relu")(inputs)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.SeparableConv2D(64, 3, activation="relu")(x)
x = layers.SeparableConv2D(128, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# Stacked RNN: every recurrent layer but the last returns full sequences.
seq_in = keras.Input(shape=(num_timesteps, num_features))
y = layers.LSTM(32, return_sequences=True)(seq_in)
y = layers.LSTM(32, return_sequences=True)(y)
y = layers.LSTM(32)(y)
seq_out = layers.Dense(num_classes, activation="sigmoid")(y)
```
- **What it demonstrates**: the template image convnet (separable convolutions, pooling to downsample, global pooling before Dense) and the stacking rule for RNNs. For very deep convnets, add BatchNormalization and residual connections. The Transformer templates (a PositionalEmbedding → TransformerEncoder → GlobalMaxPooling1D classifier, and encoder-decoder seq2seq) use the custom layers from chapter 11.

## Reference Tables
| Value-centric abstraction | Program-centric abstraction |
|---|---|
| Relates things by distance | Relates things by exact structural match |
| Continuous, grounded in geometry | Discrete, grounded in topology |
| "Averages" instances into prototypes | Isolates isomorphic substructures (subgraph isomorphism) |
| Underlies perception and intuition | Underlies reasoning and planning |
| Immediate, fuzzy, approximate | Slow, exact, rigorous |
| Needs lots of experience | Experience-efficient (as few as two instances) |

| Machine learning | Program synthesis |
|---|---|
| Model: differentiable parametric function | Model: graph of operators from a programming language |
| Engine: gradient descent | Engine: discrete search (e.g., genetic) |
| Needs lots of data | Data-efficient (a couple of examples) |

| Architecture | Use when | Notes |
|---|---|---|
| Dense | vector/categorical data; final classification or regression head | assumes no input structure |
| Convnet (Conv1D/2D/3D, SeparableConv2D) | spatially local, translation-invariant patterns | separable convolutions are leaner |
| RNN (GRU, LSTM) | sequences where order and recency matter (timeseries) | GRU is the cheaper LSTM; skip SimpleRNN |
| Transformer | sets and discrete sequences, especially seq2seq | needs positional encoding for order |

## Worked Example
**Choosing between deep learning and a program for a subproblem.** Sorting lists is non-interpolative and has an exact structure. A deep net needs huge data, still makes occasional mistakes, and fails on longer lists. A short program, validated on a couple of examples, works on any list. Choose the program. Digit recognition is perceptual and fits the manifold hypothesis. Hand-written rules stall around 90% after thousands of lines, while a fitted model uses abundant data robustly. Choose deep learning. Real systems such as AlphaGo and self-driving cars combine the two: learned perception modules feeding hand-built algorithmic modules (tree search, 3D world models).

## Key Takeaways
- Deep learning maps space X to space Y through a smooth transform learned from dense sampling. Any disruption in X or Y invalidates the mapping.
- Pick the architecture by input modality. The architecture is a hypothesis space and must match the data's structure.
- Don't anthropomorphize. Adversarial examples and texture shortcuts show there is no human-like understanding.
- Intelligence is skill-acquisition efficiency, not skill. Measure it only on tasks the system and its creators couldn't anticipate.
- The shortcut rule says you get exactly what your metric rewards and nothing else. Choose objectives carefully.
- Chollet's direction for the field: models as programs, deep learning plus program synthesis (with deep learning guiding the search), and lifelong reuse of modular subroutines.
- To keep learning: practice on Kaggle (mostly won with XGBoost for shallow ML and Keras for deep learning), follow arXiv (not peer-reviewed, so filter it, e.g., with Google Scholar author alerts), and use the Keras docs, guides, and examples.

## Connects To
- Ch 1: AI/ML/DL definitions, AI summers, why deep learning took off.
- Ch 4: last-layer activation and loss choices.
- Ch 5: manifold hypothesis, interpolation as the source of generalization, data augmentation.
- Ch 6: the universal workflow of machine learning.
- Ch 8–11: convnets, residual connections and BatchNormalization, RNNs, Transformers (PositionalEmbedding, TransformerEncoder/Decoder).
- Ch 9 and 12: gradient ascent in input space (filter visualization, DeepDream), the basis of adversarial examples.
- Ch 13: hyperparameter tuning, validation-set overfitting, ensembling, all practiced on Kaggle.
