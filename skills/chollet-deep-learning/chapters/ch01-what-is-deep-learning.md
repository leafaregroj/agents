# Chapter 1: What Is Deep Learning?

## Core Idea
Deep learning is a subfield of machine learning, which is itself a subfield of AI. It learns **successive layers of increasingly meaningful representations** from data. Machine learning reverses classical programming: you supply data and answers, and the system finds the rules. Learning means searching for useful transformations of the data, inside a predefined space of possibilities and guided by a feedback signal. Deep learning took off after 2012 because hardware, data, and algorithms improved together. Its staying power comes from **simplicity, scalability, and versatility/reusability**. Chollet's stance is: "Don't believe the short-term hype, but do believe in the long-term vision."

## Frameworks Introduced
- **Machine learning as a new programming paradigm**: classical programming is rules + data → answers. Machine learning is data + answers → rules.
  - When to use: when nobody can write the rules by hand, because the problem is fuzzy or perceptual (images, speech, language), or because hand-written rules would be brittle and costly to maintain.
  - How: give the system many labelled examples. It finds statistical structure in them, and that structure becomes a rule you can reuse. The system is *trained*, not explicitly programmed.
- **The three ingredients of machine learning**: (1) input data points, (2) examples of the expected output, (3) a way to measure whether the algorithm is doing a good job.
  - When to use: as the first check on any ML problem. If one ingredient is missing, the problem is not yet ready for ML.
  - How: the measure in (3) is the feedback signal. It measures the distance between the current output and the expected output, and adjusting the algorithm with it *is* learning.
- **Learning = search for representations**: "searching for useful representations and rules over some input data, within a predefined space of possibilities, using guidance from a feedback signal."
  - When to use: when you need to explain why a model can or cannot learn something. It can only find what lies inside its **hypothesis space**.
  - How: choose a family of transformations (coordinate changes, projections, non-linearities). Then let the feedback signal pick the member that makes the task solvable with a simple rule.
- **How deep learning works, in three figures**: (1) layers are parameterized by **weights**; (2) a **loss function** compares the predictions with the true targets and produces a loss score; (3) an **optimizer** uses that score to update the weights via **backpropagation**. Repeating this is the **training loop**.
  - When to use: as the mental skeleton for every model in the book.
  - How: start from random weights, which give random transformations and a high loss. On each batch, nudge the weights in the direction that lowers the loss. After enough iterations you have a trained network.
- **Three technical forces driving ML progress**: hardware, datasets and benchmarks, algorithmic advances.
  - When to use: when judging why an idea failed or succeeded in a given era. ML is an engineering science, so ideas can only be tested when data and compute exist.
- **Why deep learning will last**: simplicity, scalability, versatility and reusability.
  - Simplicity: no feature engineering. You train an end-to-end model built from five or six tensor operations.
  - Scalability: the work parallelizes on GPUs and TPUs, and mini-batch training handles datasets of any size.
  - Versatility and reusability: you can keep training on new data without restarting, which suits online learning, and you can reuse trained models. This makes deep learning viable on fairly small datasets.

## Key Concepts
- **Artificial intelligence**: "the effort to automate intellectual tasks normally performed by humans". It includes approaches that involve no learning at all.
- **Symbolic AI**: hand-written explicit rules over explicit knowledge bases. It dominated from the 1950s to the late 1980s and peaked with the expert-systems boom. It works on well-defined logic problems such as chess but fails on fuzzy perception.
- **Representation**: a different way to encode the same data, for example RGB vs. HSV. "Select red pixels" is easy in RGB, while "desaturate" is easy in HSV. A good representation makes the task solvable with a simple rule.
- **Hypothesis space**: the predefined set of operations a learning algorithm searches through. ML algorithms are not creative; they only search this space.
- **Depth**: the number of layers that contribute to a model. Other names for the field could have been "layered representations learning" or "hierarchical representations learning".
- **Shallow learning**: methods that learn only one or two layers of representation.
- **Neural network**: the layered model family used in deep learning. It is *not* a model of the brain. Treat it as "a mathematical framework for learning representations from data."
- **Weights / parameters**: the numbers that specify a layer's transformation. Learning means finding weight values that map inputs to targets.
- **Loss function** (also called objective function or cost function): turns predictions and targets into a distance score.
- **Optimizer**: applies the backpropagation algorithm to update the weights.
- **Feature engineering**: the manual creation of good representations for shallow models. Deep learning automates it.
- **Joint feature learning**: all layers are learned at the same time, not greedily one after another. When one feature changes, every dependent feature adapts, and everything is supervised by one feedback signal.
- **Kernel trick**: to find a separating hyperplane in a high-dimensional space, you only need the distances between pairs of points, which a kernel function computes without building the new representation.
- **Gradient boosting**: train new models one after another so that each one fixes the weak points of the ones before it. Usually applied to decision trees.
- **AI winter**: a period of funding collapse after hype that did not deliver. The first came after the symbolic-AI optimism of the 1960s; the second followed the expert-systems wave of the 1980s.

## Mental Models
- **Information distillation**: a deep network is "a multistage information-distillation process". Each layer acts as a filter, and the information gets more purified for the task as it moves up.
- **Simple mechanisms, scaled, look like magic**: nothing in a deep network is mysterious. It is a long chain of simple, learned transformations plus a feedback loop.
- **Deep learning hammer**: if deep learning is the only tool you know, every problem starts to look like a nail. Learn the classical methods so you can tell when not to use it.
- **The sigmoid of a scientific revolution**: progress is fast at first, then plateaus as hard limits appear. As of 2021, Chollet places deep learning in the second half of that curve: significant gains are still coming, but the explosive phase is over.
- **The internet in 1995**: AI today is like the internet then. It is under-deployed and its impact is easy to underestimate.

## Anti-patterns
- **Anthropomorphizing networks** ("works like the brain"): this is misleading. Drop the "just like our minds" mystique.
- **Believing short-term hype**: inflated near-term promises have twice led to AI winters. Set realistic expectations with non-technical stakeholders.
- **Hand-crafting rules for perception**: rule-based digit recognizers (loop counts, pixel histograms) are brittle and become a maintenance nightmare.
- **Greedily stacking shallow models to imitate depth**: returns drop off fast, because the best first layer of a 3-layer model is not the best first layer of a 1-layer model.
- **Using deep learning for everything**: on structured/tabular data, gradient boosted trees often win. With very little data, deep learning may not apply at all.

## Reference Tables

| Era | Approach | Key idea | Limitation |
|---|---|---|---|
| 1950s–80s | Symbolic AI / expert systems | Hand-coded rules | Fails on fuzzy perception; costly to maintain |
| 1950s→ (by hand earlier) | Probabilistic modeling (Naive Bayes, logistic regression) | Bayes' theorem with independent features; logreg is the "Hello World" of ML and is a *classification* algorithm | Shallow |
| 1989 | LeNet (LeCun, Bell Labs) | Convnets + backprop, used for USPS ZIP codes | Could not train deep stacks |
| 1990s–2000s | Kernel methods / SVM (Vapnik & Cortes, 1995) | Map data to high dimensions, maximize the margin, kernel trick | Hard to scale; needs feature engineering for perception |
| 2000s–2014 | Decision trees → random forests | Ensembles of trees; "almost always the second-best algorithm" | Shallow |
| 2014→ | Gradient boosting machines (XGBoost, LightGBM) | Each new model fixes the errors of the previous ones | Best for nonperceptual data |
| 2012→ | Deep convnets (ImageNet: 74.3% → 83.6% top-5 in 2012; 96.4% by 2015) | End-to-end learned hierarchies | Needs data and compute |

| Problem type | Default tool family (Kaggle and industry, 2016–2020) | Libraries |
|---|---|---|
| Structured / tabular data | Gradient boosted trees | Scikit-learn, XGBoost, LightGBM |
| Perceptual (images, audio, text) | Deep learning | Keras (with TensorFlow) |

**Algorithmic breakthroughs that made depth trainable** (the core problem was the feedback signal fading as layers were added):
- 2009–2010: better activation functions, better weight initialization (layer-wise pretraining, quickly abandoned), and better optimizers (RMSProp, Adam). Together these allowed models of 10 or more layers.
- 2014–2016: batch normalization, residual connections, and depthwise separable convolutions. Models of arbitrary depth could now be trained from scratch.

**Hardware numbers**: CPUs got about 5,000× faster between 1990 and 2010. NVIDIA released CUDA in 2007. A Titan RTX (2019, $2,500) delivers 16 TFLOPS, about 500× the fastest supercomputer of 1990. A TPU v3 card delivers 420 TFLOPS, and a pod of 1,024 cards delivers about 100 PFLOPS. **Investment**: AI venture capital went from under $1B (2011) to over $5B (2015) to $16B (2017).

## Worked Example
**Learning a representation for 2D points.** You have black and white points in (x, y) space, and the goal is to predict each point's color.
- Inputs: coordinates. Targets: colors. Feedback: percentage of points classified correctly.
- A coordinate change (rotate or shift the axes) gives a new representation where one simple rule works: "black if x > 0".
- Here a human chose the transformation. Machine learning automates this: it searches the space of coordinate changes and keeps the one that scores best on the feedback signal. Deep learning does the same through many stacked, learned transformations. That is what lets it handle tasks like telling a handwritten 6 from an 8, where no human can write down the right transformation.

## Key Takeaways
- ML finds rules from data plus answers. The central problem is to *meaningfully transform data* into useful representations.
- The "deep" in deep learning means successive layers of representations, not deeper understanding.
- Training loop: random weights → predictions → loss → optimizer update via backpropagation → repeat.
- Deep learning's real advantages: it automates feature engineering, and it learns all layers jointly from a single feedback signal.
- Use gradient boosted trees for structured data and deep learning for perceptual data.
- Progress needed hardware (GPUs and TPUs, subsidized by gaming), data (the internet, ImageNet, competitions), and algorithms (better gradient propagation). Easy tooling (Theano, then TensorFlow and Keras) opened the field to anyone who can write Python.
- Moderate short-term expectations, but stay confident in the long-term impact.

## Connects To
- Chapter 2: the math behind the three figures (tensors, gradients, backpropagation, the GradientTape).
- Chapter 3: Keras and TensorFlow, the tools behind the democratization of deep learning.
- Chapter 5: generalization, the manifold hypothesis, and the limits of the hypothesis space.
- Chapter 14: the limits of deep learning and the future of AI (it revisits the hype and the long-term vision).
