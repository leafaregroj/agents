# Chapter 10: Deep Learning for Timeseries

## Core Idea
Timeseries data, meaning measurements at regular intervals, has dynamics: periodic cycles, trends, a regular regime, and sudden spikes. Dense networks flatten away time, and 1D convnets assume translation invariance and pool away order, so neither fits it well. Recurrent neural networks (RNNs) process a sequence step by step while keeping a state, which respects order and causality. The chapter works through a single task, predicting the temperature 24 hours ahead on the Jena climate dataset. The method is disciplined: a common-sense baseline first, then cheap models, then LSTM/GRU with recurrent dropout, stacking, and (instructively) bidirectional layers. On this dataset the realistic ceiling is about a 10% improvement over the baseline.

## Frameworks Introduced
- **Kinds of timeseries tasks**: forecasting (the most common), classification, event detection (e.g., "hotword detection"), and anomaly detection (typically unsupervised, because you don't know what anomalies look like).
  - When to use: to frame the problem before choosing data windows and outputs.
  - How: forecasting and classification are supervised regression or classification over windows. Anomaly detection usually needs unsupervised methods.
- **Common-sense baseline, then simple model, then complex model**: before any deep model, establish a non-ML baseline. Then try a small, cheap ML model. Only then try expensive architectures.
  - When to use: always, on any new problem with no known solution.
  - How: for continuous, periodic series the baseline is "the value 24 hours from now equals the value now." Every later model must beat its MAE, or the added complexity is not legitimate.
- **Recurrent neural network (RNN)**: "a for loop that reuses quantities computed during the previous iteration of the loop." Its step function is `output_t = activation(W·input_t + U·state_t + b)`, and `state_t = output_t`.
  - When to use: for data where order matters, especially timeseries.
  - How: use `layers.LSTM` or `layers.GRU`. `SimpleRNN` is too simplistic because of vanishing gradients.
- **LSTM (Long Short-Term Memory)** (Hochreiter and Schmidhuber, 1997): a SimpleRNN plus a **carry track**, "a conveyor belt running parallel to the sequence." Information can jump on, ride to a later timestep, and jump off intact.
  - When to use: as the default recurrent layer for long-range dependencies.
  - How: `layers.LSTM(units)`. Internally the cell computes `c_{t+1} = i_t * k_t + c_t * f_t`.
- **Recurrent dropout** (Yarin Gal, 2016): apply the same dropout mask at every timestep, to both the inputs and the inner recurrent activations. A mask that changes randomly per timestep disrupts the error signal through time.
  - When to use: whenever a recurrent model overfits.
  - How: pass `dropout=` (input units) and `recurrent_dropout=` (recurrent units). Add a `Dropout` layer before the Dense head, give the network more units, and train for about 5x more epochs.
- **Stacking recurrent layers**: increases capacity once overfitting is under control and performance has plateaued.
  - When to use: when you are regularized but hitting a performance bottleneck, following the universal workflow's "increase capacity until overfitting becomes the primary obstacle."
  - How: every intermediate recurrent layer gets `return_sequences=True`.
- **Bidirectional RNN**: two RNN instances, one reading chronologically and one antichronologically, with their representations merged. It is the "Swiss Army knife of deep learning for natural language processing."
  - When to use: when order matters but which order you read in does not (text).
  - How: `layers.Bidirectional(layers.LSTM(16))`. Avoid it when the recent past dominates the prediction (weather).

## Key Concepts
- **Timeseries split by time**: validation and test data must be more recent than the training data. Use 50% / 25% / 25% chronologically, because "some problems happen to be considerably simpler if you reverse the time axis."
- **Per-feature normalization from training data only**: compute the mean and std on the first `num_train_samples` rows only.
- **`timeseries_dataset_from_array()`**: generates overlapping windows on the fly instead of materializing redundant samples. `targets` is the `data` array offset by the forecast delay.
- **`sampling_rate`, `sequence_length`, `delay`**: with readings every 10 minutes, `sampling_rate=6` gives one point per hour, `sequence_length=120` gives 5 days, and `delay = sampling_rate * (sequence_length + 24 - 1)` points the target 24 hours past the window's end.
- **MSE loss, MAE metric**: MSE is smooth around zero, which suits gradient descent. MAE is easier to interpret, in degrees. The regression head is a Dense layer with no activation.
- **Translation invariance assumption (1D)**: Conv1D assumes a window's content has the same properties wherever it slides. Weather breaks this, because morning, evening, and night behave differently.
- **Feedforward network**: has no memory and processes each input independently. A sequence has to be shown all at once, as one flattened point.
- **`return_sequences`**: `False` (the default) returns `(batch, units)`, the last output, which already summarizes the whole past. `True` returns `(batch, timesteps, units)`.
- **Variable-length input**: `keras.Input(shape=(None, num_features))`. Prefer a fixed length when you have one, because it enables summary shapes and `unroll`.
- **GRU (Gated Recurrent Unit)** (Cho et al., 2014): a "slightly simpler, streamlined version" of LSTM.
- **cuDNN kernel trade-off**: LSTM and GRU on GPU with default arguments use fast cuDNN kernels. `recurrent_dropout` is not supported by those kernels and forces a fallback that is 2-5x slower.
- **Unrolling**: `unroll=True` inlines the loop. It needs a fixed sequence length and is viable only for about 100 steps or fewer, at a memory cost.
- **Information arbitrage**: all trading is gaining an advantage from data or insights others lack. Public data plus well-known ML techniques gives no edge.

## Mental Models
- **A good solution existing in the hypothesis space does not mean gradient descent will find it**: the dense model cannot reliably rediscover the trivial "same as now" heuristic. "Your common sense contains a lot of valuable information to which a machine learning model doesn't have access." The remedy is feature engineering and the right architecture priors.
- **An RNN cell spec is a set of constraints, not a design**: what gates "do" is decided by learned weights, so interpretations like "f_t forgets" don't mean much. You only need to know the LSTM's purpose, which is to let past information be reinjected later and so fight vanishing gradients.
- **The carry track is a temporal residual connection**: it is the same idea as chapter 9's residuals.
- **Reversed sequences give different but useful representations**: the more different the representations, the more value in combining them. This is the intuition behind both bidirectional RNNs and ensembling.
- **Always look for periodicity**: data shows daily and yearly cycles, and human-generated data also shows weekly cycles. The scale of the forecast decides whether the problem is easy (monthly averages) or chaotic (daily).
- **Machine learning works where the past predicts the future**: weather, electricity load, and store foot traffic qualify. Markets do not: "looking in the rear-view mirror is a bad way to drive."

## Anti-patterns
- Randomly shuffling timesteps across splits, or validating on data older than the training data.
- Computing normalization statistics over the full dataset, which leaks validation and test information.
- Flattening a timeseries into a Dense network, which removes the notion of time.
- Applying Conv1D plus pooling to data that is not translation-invariant at the window's scale, or where recency matters. The pooling destroys order.
- Using a standard `Dropout` layer before a recurrent layer, which hinders learning, or a per-timestep random mask.
- Using `SimpleRNN` for real work, because long-term dependencies are unlearnable in practice.
- Using bidirectional RNNs on recency-dominated forecasting. The antichronological half adds capacity, overfits early, and contributes nothing.
- Skipping the baseline, in which case you can't tell whether you are making progress.
- Applying public-data ML to stock or forex price prediction.

## Code Examples
```python
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)
# val: start_index=num_train_samples, end_index=num_train_samples + num_val_samples
# test: start_index=num_train_samples + num_val_samples

def evaluate_naive_method(dataset):
    total_abs_err, samples_seen = 0., 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]   # last temp, un-normalized
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen
```
- **What it demonstrates**: windowed forecasting datasets with chronological splits, and the common-sense baseline evaluated in original units.

```python
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = layers.GRU(32, recurrent_dropout=0.5)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)                     # no activation: regression
model = keras.Model(inputs, outputs)

callbacks = [keras.callbacks.ModelCheckpoint("jena_stacked_gru_dropout.keras",
                                             save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=50,
                    validation_data=val_dataset, callbacks=callbacks)
model = keras.models.load_model("jena_stacked_gru_dropout.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
```
- **What it demonstrates**: the best model in the chapter, a stacked, dropout-regularized GRU. Intermediate layers use `return_sequences=True`, a Dropout sits before the head, the loss is MSE with MAE as the metric, and the best checkpoint is reloaded for testing.

```python
import numpy as np
timesteps, input_features, output_features = 100, 32, 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)  # (timesteps, output_features)
```
- **What it demonstrates**: an RNN is just a loop with state. This is the forward pass that Keras's `SimpleRNN` implements.

## Reference Tables
| Model (Jena, 24h-ahead temperature) | Val MAE (°C) | Test MAE (°C) | Notes |
|---|---|---|---|
| Common-sense baseline ("same as now") | 2.44 | 2.62 | Surprisingly hard to beat |
| Flatten, Dense(16), Dense(1) | ~baseline, unreliable | — | Can't find the simple solution |
| Conv1D (windows 24/12/6) + pooling | ~2.9 | — | Worse than baseline: breaks translation invariance and order |
| LSTM(16) | 2.36 | 2.55 | First to beat baseline; overfits fast |
| LSTM(32, recurrent_dropout=0.25) + Dropout(0.5), 50 epochs | 2.27 | 2.45 | 7% / 6.5% better than baseline |
| 2x GRU(32, recurrent_dropout=0.5) + Dropout(0.5) | — | 2.39 | 8.8% better; diminishing returns |
| LSTM on reversed sequences | — | — | Strongly underperforms the baseline |
| Bidirectional(LSTM(16)) | — | — | Worse than plain LSTM here |

| Setting | Value | Reason |
|---|---|---|
| Split | 50 / 25 / 25 chronological | Predict the future from the past |
| Samples | 210,225 / 105,112 / 105,114 | 420,551 rows of 14 features |
| `sampling_rate` | 6 | One point per hour (raw data every 10 min) |
| `sequence_length` | 120 | 5 days of hourly data |
| Batch size | 256 | Batch shape (256, 120, 14) |
| Realistic ceiling | ~10% over baseline | Single-location data; weather depends on surrounding regions |

## Worked Example
**Jena climate temperature forecasting.**
1. Parse the CSV (420,551 rows, 14 features, Date Time dropped). Temperature is column 1.
2. Plot the series. It shows yearly periodicity over 8 years and daily periodicity across 10 days (144 points per day).
3. Normalize every feature using training-set mean and std. Build train/val/test datasets with `timeseries_dataset_from_array`.
4. The baseline gives a validation MAE of 2.44. The dense model fails to beat it reliably. The 1D convnet does worse, at about 2.9.
5. LSTM(16) gives a validation MAE of 2.36 and a test MAE of 2.55: machine learning finally adds value, though the model overfits within a few epochs.
6. Add `recurrent_dropout=0.25` and `Dropout(0.5)`, double the units, and train for 50 epochs. Test MAE reaches 2.45 with no overfitting for the first 20 epochs.
7. Performance has now plateaued, so add capacity by stacking two GRUs with dropout. Test MAE is 2.39.
8. Probe processing order. Reversed sequences perform badly, which confirms that recency matters, so Bidirectional doesn't help here.
9. Further options: tune units, dropout, and learning rate; use a Dense stack as the regressor; change sequence length or sampling rate; do feature engineering. "You must iterate."

## Key Takeaways
- Establish a common-sense baseline and beat it with the simplest model possible before going bigger.
- Split timeseries chronologically, and normalize with training statistics only.
- Use LSTM or GRU for ordered data. They easily outperform models that flatten time first.
- Regularize recurrent layers with `recurrent_dropout` (time-constant masks), not with Dropout placed before the layer.
- Stack recurrent layers (`return_sequences=True`) only after overfitting is under control. Expect diminishing returns on small problems.
- Use bidirectional RNNs for text-like data where reading direction doesn't matter. Avoid them for recency-driven forecasting.
- Small RNNs often run faster on a multicore CPU than on a GPU. `recurrent_dropout` disables cuDNN.
- Don't use these techniques to forecast markets.

## Connects To
- Chapter 5: common-sense baselines and the hypothesis space.
- Chapter 6: the universal workflow of machine learning, i.e., increase capacity until overfitting, then regularize.
- Chapter 8: Conv1D/SeparableConv1D as 1D analogues of 2D convnets.
- Chapter 9: residual connections, the same idea as the LSTM carry track, and cuDNN co-evolution.
- Chapter 11: bidirectional LSTMs for text and the Transformer that superseded them.
- Chapter 13: ensembling, which exploits diverse representations like reversed-order RNNs.
