# Chapter 10: Deep Learning

## Core Idea
A neural network builds non-linear features (hidden-layer activations) from its inputs and feeds them into a linear model. Everything is fitted together by gradient descent with backpropagation. Specialized architectures such as CNNs for images and RNNs for sequences exploit structure through weight sharing. Deep learning shines when n is huge and interpretability matters less. When simpler models do as well, follow Occam's razor.

## Frameworks Introduced
- **Single layer neural network**: f(X) = β₀ + Σₖ₌₁ᴷ βₖhₖ(X), where the activations are Aₖ = hₖ(X) = g(wₖ₀ + Σⱼ wₖⱼXⱼ).
  - Think of it as a linear regression on K *learned* basis functions. Without the non-linear g, the model collapses to a linear one.
  - The activation g is the sigmoid g(z) = eᶻ/(1+eᶻ), or the now-preferred **ReLU** g(z) = max(0, z), which is cheaper to compute and store.
- **Multilayer neural network**: hidden layers are stacked, with A⁽¹⁾ = g(W₁X) and A⁽²⁾ = g(W₂A⁽¹⁾). The output is Zₘ = βₘ₀ + Σ βₘₗA⁽²⁾ₗ. For K classes, apply the softmax fₘ(X) = e^{Zₘ}/Σₗ e^{Zₗ} and minimize cross-entropy.
  - When to use: very large n with complex structure, such as images, text or speech.
  - Prefer several modest layers over one huge layer. Use many units, and control overfitting with regularization.
- **Convolutional neural network (CNN)**: alternates *convolution layers* (banks of small learned filters, e.g. 3×3, slid over the image with shared weights, followed by ReLU) with *pooling layers* (2×2 max pooling, which condenses the image and gives some location invariance). The number of filters usually grows as the image shrinks. The result is flattened into dense layers and a softmax.
  - When to use: image classification. Convolutions find local features such as edges and stripes, and later layers combine them hierarchically.
  - **Data augmentation**: replicate each training image with random distortions such as zoom, shift, shear, rotation and flips, which leave the label unchanged. This acts like ridge regularization, and it fits naturally into SGD minibatches.
  - **Pretrained classifier / transfer learning**: reuse a network trained on ImageNet, such as resnet50, either as-is or by freezing its convolution layers and retraining only the last layers.
- **Document classification (bag-of-words)**: represent each document as a binary vector over a dictionary of the M most frequent words (M = 10,000 for IMDb), which gives a sparse matrix. Fit a two-hidden-layer network or a lasso logistic regression. Both reach about 88% test accuracy.
- **Recurrent neural network (RNN)**: process a sequence X₁…X_L with *shared* weights W (input→hidden), U (hidden→hidden) and B (hidden→output). The hidden state is A_ℓ = g(W X_ℓ + U A_{ℓ−1}), and the output is O_ℓ = β₀ + βᵀA_ℓ. Only the final output O_L enters the loss.
  - Uses: documents, with words passed through an **embedding layer** E (m × D, m ≈ low hundreds). E can be learned, or pretrained (word2vec, GloVe) and frozen, which is called *weight freezing*. RNNs are also used for time series forecasting and for speech and translation (seq2seq).
  - The **LSTM** variant keeps both long-term and short-term memory tracks so that early signals are not washed out. On IMDb it reached 87%.
- **Autoregressive (AR) framing of time series**: create (X, Y) pairs from lagged mini-series of length L and regress v_t on v_{t−1},…,v_{t−L} (plus other series). An RNN on the same lags is closely related. Feeding the lags into a feedforward network gives a *non-linear AR* model.
- **Fitting: gradient descent + backpropagation + SGD + regularization**:
  - How: start θ⁰ at small random values. Repeat θᵐ⁺¹ ← θᵐ − ρ∇R(θᵐ) until the objective stops decreasing.
  - **Backpropagation**: the chain rule assigns a fraction of each residual yᵢ − f(xᵢ) to every parameter.
  - **Minibatch SGD**: sample a small batch for each gradient step. An *epoch* is one pass through the equivalent of the full training set.
  - Regularization: a ridge or lasso penalty λΣθⱼ², possibly per layer; **dropout**, which zeroes a random fraction φ of units each time an observation is processed and scales the survivors by 1/(1−φ); **early stopping** on a validation set; and data augmentation.
  - Why it works / failure mode: the objective is *non-convex* with multiple local minima. Slow learning (a small learning rate ρ, many epochs) together with regularization guards against overfitting. Networks often have more parameters than observations (235K weights vs. 60K MNIST images), so regularization is essential.

## Key Formulas
- Single layer: f(X) = β₀ + Σₖ βₖ g(wₖ₀ + Σⱼ wₖⱼXⱼ).
- Sigmoid: g(z) = eᶻ/(1 + eᶻ) = 1/(1 + e⁻ᶻ). ReLU: g(z) = (z)₊ = max(0, z).
- Regression loss: Σᵢ (yᵢ − f(xᵢ))².
- Softmax: fₘ(X) = Pr(Y = m | X) = e^{Zₘ} / Σ_{ℓ=0}^{9} e^{Z_ℓ}.
- Cross-entropy: −Σᵢ Σₘ yᵢₘ log fₘ(xᵢ), where yᵢₘ is a one-hot indicator.
- Ridge-penalized objective: R(θ; λ) = −Σᵢ Σₘ yᵢₘ log fₘ(xᵢ) + λ Σⱼ θⱼ².
- Gradient step: θᵐ⁺¹ ← θᵐ − ρ ∇R(θᵐ), where ρ is the learning rate.
- Backprop for a single layer, with zᵢₖ = wₖ₀ + Σⱼ wₖⱼxᵢⱼ and Rᵢ = ½(yᵢ − f(xᵢ))²: ∂Rᵢ/∂βₖ = −(yᵢ − f(xᵢ)) g(zᵢₖ), and ∂Rᵢ/∂wₖⱼ = −(yᵢ − f(xᵢ)) βₖ g′(zᵢₖ) xᵢⱼ.
- RNN: A_ℓₖ = g(wₖ₀ + Σⱼ wₖⱼX_ℓⱼ + Σₛ uₖₛA_{ℓ−1,s}), O_ℓ = β₀ + Σₖ βₖA_ℓₖ, and the loss is (Y − O_L)².
- With two classes, the softmax gives log[Pr(Y=1|X)/Pr(Y=0|X)] = Z₁ − Z₀, which is logistic regression on the learned features.
- MAE = (1/n) Σ|yᵢ − ŷᵢ|.

## Key Concepts
- **Hidden unit / activation**: a learned non-linear transformation of a linear combination of the inputs.
- **Weights and biases**: the ML community's names for coefficients and intercepts. This "bias" is not statistical bias.
- **Feature map / channel**: a 2-D array per channel (for example R, G and B), stacked into a 3-D tensor.
- **Weight sharing**: the same filter weights (CNN) or the same W, U, B (RNN) are reused across positions, which drastically cuts the number of parameters.
- **Embedding**: a dense m-dimensional representation of each word in place of a one-hot vector of length D.
- **Epoch / minibatch**: with 48,000 training images and a batch of 128, an epoch is about 375 SGD updates.
- **Dropout**: inspired by random forests. Random units are removed during training to prevent over-specialization.
- **Interpolation threshold / double descent**: once the number of basis functions or parameters reaches n, the training error hits zero. Test error spikes there, then *descends again* as the minimum-norm interpolating solution becomes smoother.

## Mental Models
- **Think of a hidden layer as automatic feature engineering.** It replaces hand-built bases such as splines (Ch 7) with ones learned from the data.
- **Use deep learning when n is very large, there is spatial or sequential structure, the signal-to-noise ratio is high, and interpretability is not a priority.** Otherwise, try linear models, the lasso, GAMs and boosting first.
- **Regularization is the real complexity knob, not the architecture size.** Use wide networks, then rein them in with dropout, ridge, early stopping and augmentation.
- **Double descent does not refute the bias-variance trade-off.** Counting parameters mismeasures flexibility. A minimum-norm interpolant with d = 42 has lower variance than one with d = 20. A properly tuned ridge fit would never interpolate and would do even better.

## Anti-patterns
- **Reaching for a neural network on small tabular data**: on Hitters (n = 263), an NN with 1,345 parameters did no better than OLS or the lasso, and it is a black box.
- **Reporting p-values from a model selected on the same data**: this introduces selection bias. The authors refit the lasso-chosen model on the held-out test data.
- **Training for too many epochs without monitoring validation error**: on MNIST, the validation objective starts rising by about 30 epochs. Use early stopping.
- **Deliberately relying on double descent / interpolation**: it depends on a high signal-to-noise ratio. Regularized fits usually do better.
- **Forgetting to standardize inputs**: the scale affects the fit and the penalties. The lab standardizes before fitting the lasso and the NN.
- **Tinkering with hyperparameters against the test set**: the tuning process itself can overfit if done carelessly.

## Code Examples
```python
import torch
from torch import nn
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256),
                                    nn.ReLU(), nn.Dropout(0.4))
        self.layer2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self._forward = nn.Sequential(self.layer1, self.layer2, nn.Linear(128, 10))
    def forward(self, x):
        return self._forward(x)
```
- **What it demonstrates**: the Fig. 10.4 MNIST network (784→256→128→10, with 40% and 30% dropout) as a `torch` `nn.Module`. The softmax and cross-entropy are handled by the classification module.

```python
from pytorch_lightning import Trainer
from ISLP.torch import SimpleDataModule, SimpleModule, ErrorTracker
from torchmetrics import MeanAbsoluteError
hit_dm = SimpleDataModule(hit_train, hit_test, batch_size=32,
                          num_workers=min(4, max_num_workers), validation=hit_test)
hit_module = SimpleModule.regression(hit_model, metrics={'mae': MeanAbsoluteError()})
hit_trainer = Trainer(deterministic=True, max_epochs=50, log_every_n_steps=5,
                      logger=hit_logger, callbacks=[ErrorTracker()])
hit_trainer.fit(hit_module, datamodule=hit_dm)
hit_trainer.test(hit_module, datamodule=hit_dm)    # test_mae ≈ 229.5
```
- **What it demonstrates**: the ISLP + pytorch_lightning training loop. A data module handles batching, `SimpleModule.regression` or `.classification` supplies the loss and metrics, and `Trainer` runs SGD by epoch. Here the Hitters model is 19→50 ReLU→Dropout(0.4)→1, with 1,051 parameters.

## Reference Tables
**MNIST test error (Table 10.1)**
| Method | Test error |
|---|---|
| NN + ridge regularization | 2.3% |
| NN + dropout regularization | 1.8% |
| Multinomial logistic regression | 7.2% |
| LDA | 12.7% |

**Hitters test results (Table 10.2; 176 train / 87 test)**
| Model | # Parameters | Mean abs. error | Test R² |
|---|---|---|---|
| Linear regression | 20 | 254.7 | 0.56 |
| Lasso (10-fold CV, 12 variables) | 12 | 252.3 | 0.51 |
| Neural network (64 ReLU, dropout) | 1,345 | 257.4 | 0.54 |
| Relaxed lasso, 4 variables | — | **224.8** | — |

**Architecture vs. data type**
| Data | Architecture | Key structural trick |
|---|---|---|
| Tabular | Feedforward (dense) | — (often no better than linear models) |
| Images | CNN | convolution filters + pooling, weight sharing, augmentation |
| Text | Bag-of-words NN / RNN with embeddings / LSTM | sparse features; sequence and order |
| Time series | RNN / AR / non-linear AR on lags | lagged mini-series (lag L) |

Other results: IMDb accuracy was about 88% (bag-of-words NN or lasso logistic), 76% for a simple RNN with a learned embedding (m = 32), and 87% for an LSTM. The NYSE log-volume forecast (L = 5 lags, RNN with K = 12 hidden units) reached test R² = 0.42. A linear AR model got 0.41, the "yesterday's value" baseline 0.18, and adding day_of_week raised the RNN and AR models to 0.46 and the non-linear AR to 0.47. Lab test accuracies: MNIST NN 96.8%, MNIST multiclass logistic regression 92.4%, CIFAR100 CNN 42%, IMDb lasso logistic regression about 85.5% and NN 84.8%. The CIFAR100 state of the art at the time of writing was just above 75%.

## Worked Example
**Double descent with natural splines.** Simulate n = 20 points from Y = sin(X) + ε, with X ~ U[−5, 5] and σ = 0.3, so the signal-to-noise ratio is about 5.9. Fit natural splines with d basis functions by least squares, and for d > n pick the **minimum-norm** solution (the smallest Σβ̂ⱼ²). At d = 8 the fit is good. At d = 20 = n the fit interpolates the data, makes wild excursions, and the test error explodes. At d = 42 and d = 80 the fits still interpolate (training error 0), but they are much smoother, and the test error descends a second time. The cause: only one interpolant exists at d = n, while for d > n there are infinitely many, and the smoothest of them is chosen. SVMs and SGD-trained networks implicitly prefer similarly smooth, minimum-norm solutions. A ridge-penalized spline would avoid the spike entirely and give lower test error.

## Key Takeaways
1. A neural network is a linear model on learned non-linear features. ReLU plus softmax and cross-entropy is the standard classification recipe.
2. Fit by gradient descent with backprop, using minibatch SGD. Regularize with ridge or lasso, dropout, early stopping and data augmentation.
3. CNNs (convolution + pooling) are built for images and RNNs/LSTMs (shared weights over time, embeddings) for sequences. Both rely on weight sharing.
4. Pretrained networks (resnet50 on ImageNet) and pretrained embeddings (GloVe, word2vec) transfer to new tasks.
5. On modest tabular data, linear models and the lasso often match neural networks. Apply Occam's razor.
6. Double descent can happen when interpolating with minimum-norm solutions. The bias-variance trade-off still holds, so do not rely on interpolation.

## Connects To
- **Ch 3/4**: the output layer is linear or logistic regression, and the softmax is multinomial logistic regression.
- **Ch 5**: the validation set approach is used for early stopping and tuning.
- **Ch 6**: ridge and lasso penalties on weights. The lasso is a competitive baseline (Hitters, IMDb).
- **Ch 7**: hidden units are learned basis functions. Natural splines illustrate double descent.
- **Ch 8**: dropout is inspired by random forests' random feature removal.
- **Ch 9**: SVMs' minimum-norm solutions relate to good interpolation behaviour.
- **External**: transformers and attention (the successors to RNNs for language), Adam/RMSprop optimizers, batch normalization, transfer learning.
