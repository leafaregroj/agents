# Patterns & Techniques — Deep Learning with Python

## Last Layer + Loss by Task
**When to use**: Every new model (Ch 4, 6, 14).
**How**:
- Binary classification: `Dense(1, sigmoid)` + `binary_crossentropy`.
- Single-label N-class: `Dense(N, softmax)` + `categorical_crossentropy`, or `sparse_categorical_crossentropy` for integer labels.
- Multilabel: `Dense(N, sigmoid)` + `binary_crossentropy`.
- Scalar regression: `Dense(1)` with no activation, `mse` loss and `mae` metric.
**Trade-offs**: If the metric you care about isn't differentiable (ROC AUC), optimize a proxy loss and monitor the metric.

## Overfit First, Then Regularize
**When to use**: Developing any model (Ch 5, 6).
**How**:
1. Beat a common-sense baseline (statistical power).
2. Scale up (more layers, units, epochs) until validation loss turns up.
3. Regularize and tune: more or better data, better features, early stopping, smaller capacity, L1/L2 weight penalties, dropout.
4. Retrain on train+val and evaluate once on test.
If training loss stalls, fix the learning rate or batch size. If the model trains but can't beat the baseline, check the data and the architecture priors.
**Trade-offs**: Each round of tuning on validation leaks information into the model.

## Evaluation Protocol
**When to use**: Choosing how to estimate generalization (Ch 4, 5).
**How**: Use simple holdout when data is plentiful. Use K-fold (K = 4–5) with little data, and iterated K-fold with shuffling for precise estimates. Normalize with training statistics only. Split timeseries chronologically, with validation and test data newer than training data.
**Trade-offs**: K-fold multiplies training cost by K.

## Small-Data Vision Toolbox
**When to use**: An image classifier with a few thousand images (Ch 8).
**How**: Train a small convnet from scratch with data augmentation (RandomFlip, RandomRotation(0.1), RandomZoom(0.2)) and Dropout(0.5). Next, try feature extraction on a frozen pretrained base, either fast (`predict()` once) or end-to-end with augmentation. Then fine-tune:
1. Add a head.
2. Freeze the base.
3. Train the head.
4. Unfreeze the top few layers.
5. Retrain jointly at learning rate 1e-5.
**Trade-offs**: Fast feature extraction can't use augmentation. End-to-end extraction needs a GPU. Fine-tuning too many layers overfits.

## Modern Convnet Blocks (Mini-Xception)
**When to use**: Building a vision model beyond a toy (Ch 9).
**How**:
- Structure the model as stacked blocks: Conv(use_bias=False) → BatchNormalization → Activation, using SeparableConv2D after the first layer.
- Add a residual connection around each block, with a 1×1 projection when shapes change.
- Increase filters as the spatial size shrinks.
- End with GlobalAveragePooling → Dropout → Dense.
**Trade-offs**: Never use separable convs on raw RGB input. Freeze BatchNormalization when fine-tuning.

## Segmentation Encoder–Decoder
**When to use**: Per-pixel classification (Ch 9).
**How**: Downsample with strided Conv2D (not max pooling, which loses location), upsample with mirrored Conv2DTranspose, and output a per-pixel softmax with sparse crossentropy.

## Timeseries Forecasting
**When to use**: Predicting future values from sequences (Ch 10).
**How**: Use a chronological 50/25/25 split and normalize with training statistics. Window the data with `timeseries_dataset_from_array`. Compute a naive baseline first. Then try a small dense model, then LSTM or GRU, then add `recurrent_dropout` with doubled units, then stack layers (`return_sequences=True` on every layer except the last). Use MSE loss with an MAE metric, and checkpoint the best model.
**Trade-offs**: `recurrent_dropout` disables cuDNN (2–5× slower). Bidirectional RNNs don't help forecasting where recent data matters most. Conv1D ignores order sensitivity.

## Text Classification
**When to use**: Classifying documents (Ch 11).
**How**: Run `TextVectorization` inside `tf.data.map()`. Choose by the ratio of samples to mean words per sample: below 1,500, use bag-of-bigrams (`ngrams=2`, multi_hot or tf_idf) → Dense(16) → Dropout(0.5); above 1,500, use a sequence model (PositionalEmbedding → TransformerEncoder → GlobalMaxPooling). For export, wrap the vectorizer into an inference model that takes raw strings.
**Trade-offs**: A Transformer without positional information is a bag-of-words model. Pretrained embeddings help only on small datasets.

## Sequence-to-Sequence
**When to use**: Translation and other tasks that map a sequence to a sequence (Ch 11).
**How**: Use inputs `{source, target[:, :-1]}` and targets `target[:, 1:]`. Use a Transformer encoder + decoder with a causal mask, or GRU seq2seq with the encoder state passed as the decoder's `initial_state`. At inference, decode greedily from `[start]` until `[end]`.

## Text Generation by Temperature Sampling
**When to use**: Generating text from a language model (Ch 12).
**How**: Train next-token prediction with `x = seq[:, :-1]` and `y = seq[:, 1:]`. Sample with `exp(log p / T)`, renormalize, and draw a multinomial sample. Sweep several temperatures.

## Generative Image Models
**When to use**: Image synthesis or editing (Ch 12).
**How**:
- **Neural style transfer**: VGG19, content loss + Gram-matrix style loss + total variation loss, optimized on the image.
- **VAE**: a sampler layer and a custom `train_step` combining reconstruction loss and KL loss.
- **GAN**: override `compile` with two optimizers and alternate discriminator and generator steps.
- **DeepDream**: gradient ascent on layer activations over octaves.
**Trade-offs**: VAEs give structured latent spaces for editing and animation. GANs give realism but are unstable to train.

## Custom Training
**When to use**: You need control beyond `fit()` (Ch 7).
**How**: Escalate only as needed:
1. Built-in `fit()`.
2. Custom metrics and callbacks.
3. Override `train_step()`, which keeps `fit()`'s machinery.
4. A full GradientTape loop wrapped in `@tf.function`.
Debug eagerly, then compile.

## Hyperparameter Search & Ensembling
**When to use**: Squeezing out the last points of accuracy (Ch 13).
**How**: Write `build_model(hp)` and run KerasTuner `BayesianOptimization` on a validation metric with EarlyStopping. Retrain the top configurations on train+val for about 1.2× the best epoch. Ensemble diverse model families with weights learned on validation data.
**Trade-offs**: Tuning overfits the validation set. Ensembling the same network with different seeds isn't worth it.

## Scaling Training
**When to use**: Training is too slow (Ch 13).
**How**: Use mixed precision (`mixed_float16`) first. Next, use `MirroredStrategy` on multiple GPUs, building and compiling inside `scope()`. On TPUs, use `TPUStrategy` with `steps_per_execution`. Add `prefetch` and `cache` to the input pipeline.
**Trade-offs**: Larger batches need a higher learning rate. TPUs need big batches and fast I/O.
