# Cheatsheet — Chollet Decision Rules

## Which model for which data
| Data | Reach for | Notes |
|---|---|---|
| Structured or tabular | Gradient boosted trees (XGBoost/LightGBM) | DL is for perceptual data; random forests are "almost always the second-best" |
| Vector data | Dense stack | Normalize features: mean 0, std 1 |
| Images | Convnet; residual + BatchNorm + separable convs | Pretrained base when data is small |
| Per-pixel labels | Strided conv encoder + Conv2DTranspose decoder | Strides, not max pooling |
| Timeseries | Baseline → LSTM/GRU with recurrent_dropout | Not bidirectional; never market prediction |
| Text, samples/mean words < 1,500 | Bag-of-bigrams + Dense | Beat the sequence models on IMDB (90.4%) |
| Text, ratio > 1,500, or seq2seq | Transformer | Positional embedding required |
| Video | 3D convnet, or a per-frame convnet + sequence model | |
| Algorithmic or exact tasks | Write a program | DL only interpolates |

## Defaults
- Optimizer `rmsprop` (Adam is also standard). Batch 32–512; powers of 2.
- Convs: 3×3, stride 1, `padding="same"` when stacking deep. MaxPool 2×2 stride 2. Increase filters as maps shrink.
- Dropout 0.2–0.5. L2 ≈ 0.001–0.002. EarlyStopping `patience=2`–10 with `ModelCheckpoint(save_best_only=True)`.
- Fine-tuning learning rate 1e-5. Unfreeze only the top few layers. Keep BatchNorm frozen.
- Text: vocab 20k, embed 256, cap length near the 95th percentile of document lengths (IMDB: 600).
- Text sampling temperature: sweep it; ≈0.7 worked best (0.2 loops; 1.5 is near random).
- GAN: LeakyReLU(0.2), kernel 4 with stride 2, normal latent, label noise 0.05, Dropout(0.2) in the discriminator, Adam 1e-4.
- Mixed precision gives up to 3× on GPU; keep the softmax output in float32.
- Multi-GPU speed-up: 2 GPUs ≈ 2×, 4 ≈ 3.8×, 8 ≈ 7.3×.

## Decision rules
- **Beat a common-sense baseline first** (MNIST 10%, IMDB 50%, Reuters ≈19%, timeseries "same as 24 h ago" MAE 2.62).
- **Model can't overfit?** Add capacity. **Training loss stuck?** Lower the learning rate or change the batch size. **No better than baseline?** The data or the architecture priors are wrong.
- **Overfitting?** In order: more or better data, better features, early stopping, smaller model, L1/L2 (small models), dropout (large models).
- **Small dataset?** Small model, K-fold, augmentation, pretrained features.
- **Validation metric got better after many tuning rounds?** Suspect a leak. Test once, at the end.
- **Timeseries?** Never shuffle across time. Validation and test data must be newer than training data.
- **Normalize** with training-set statistics only. Fill missing numeric values with the mean or median, never 0.
- **Use the Functional API** whenever the model is a graph of layers without cycles; use subclassing only when you need an arbitrary `call()`.
- **Need custom training?** Override `train_step` before writing a loop from scratch.
- **Deploy where it fits**: REST API for online, latency-tolerant use; on-device (TF Lite) for latency, offline or privacy; browser (TF.js) to keep data local.
- **An extra 50 hours?** Spend them collecting data, not tuning.

## Tells & smells
- Validation loss is lowest after 1–4 epochs → model too big for the data.
- Training accuracy ≈ 100% while validation stays flat → memorizing; regularize.
- The discriminator's loss → 0 while the generator's loss rises → weaken the discriminator (lower its learning rate, raise its dropout).
- GAN images show checkerboard artifacts → make kernel size divisible by stride.
- RNN training is suddenly 2–5× slower → `recurrent_dropout` disabled cuDNN.
- Feature present at training time but not in production → target leak.
- Model accuracy decays in production → concept drift; keep collecting hard samples and retrain.
- Ensembles of the same network with different seeds → little gain; diversify model families.
