# Chapter 12: Generative Deep Learning

## Core Idea
Perceptual data, language, and art all have statistical structure, and deep learning is very good at learning it. Once a model has learned a **latent space** of images, music, or text, you can **sample** from it to make new artifacts that resemble the training data. Chollet treats this as *augmented intelligence*: sampling is a mathematical operation with no grounding in human experience. Meaning comes from the person interpreting or steering the output. Latent-space sampling becomes a brush that "factors art apart from craft."

The chapter covers five techniques. Three are trained generative models: language models for text, VAEs, and GANs. Two optimize an input image against a pretrained convnet: DeepDream and neural style transfer. All of them come down to the same deep learning move: define a loss that says what you want, then optimize it, either over the model's weights or over the input pixels.

## Frameworks Introduced
- **Language-model sampling loop**: train a model to predict the next token(s) from previous tokens. Then feed it **conditioning data** (a prompt), sample the next token, append it, and repeat.
  - When to use: any discrete sequence generation, such as text, musical notes, brush-stroke timeseries, dialogue, or speech.
  - How: use a Transformer decoder or RNN, a softmax over the vocabulary, and a **sampling strategy** that includes a temperature.
- **Softmax temperature**: reweight the model's output distribution to control entropy. Low temperature gives predictable, repetitive output. High temperature gives surprising and eventually random output.
  - When to use: always, when you sample from a generative sequence model.
  - How: compute `p' ∝ exp(log(p)/T)`, renormalize, and sample. Try several values of T, because humans judge the result and there's no way to know the best entropy in advance.
- **Sequence-to-sequence language modeling with causal masking**: feed tokens 0..N and predict tokens 1..N+1. A causal mask ensures position i only sees tokens 0..i.
  - When to use: text generation, instead of plain "N words → word N+1".
  - How: use a `TransformerDecoder` with no encoder. Targets are the input shifted by one token.
- **DeepDream** is a multiscale form of filter visualization. It runs gradient *ascent* on an existing image to maximize the activations of entire layers, across several **octaves**, with **detail reinjection**.
- **Neural style transfer (Gatys et al., 2015)**: `loss = distance(style(ref) − style(combo)) + distance(content(orig) − content(combo))`. Content is measured with upper-layer activations. Style is measured with **Gram matrices** across many layers.
  - When to use: texture transfer from a strongly textured, self-similar style image onto a content image that stays recognizable without fine detail.
- **Variational autoencoder (VAE)**: the encoder maps an input to a distribution `(z_mean, z_log_var)`. You sample `z = z_mean + exp(0.5·z_log_var)·ε`, and the decoder reconstructs the input. Training uses a **reconstruction loss** plus a **KL-divergence regularization loss**.
  - When to use: when you need a structured, continuous latent space for image editing, **concept vectors**, or latent-space animation.
- **Generative adversarial network (GAN)**: a generator ("forger") and a discriminator ("art dealer") are trained against each other. Training seeks an equilibrium, not a minimum.
  - When to use: when you need highly realistic single frames and latent-space structure doesn't matter.
  - How: write a custom `train_step` that alternates a discriminator update on mixed real and fake images with a generator update through the frozen discriminator using "misleading" labels.
- **A bag of tricks (GANs)**: heuristic, "more alchemy than science." See the Reference Tables section.

## Key Concepts
- **Language model**: any network that models the probability of the next token given the previous ones. It captures the statistical structure, or latent space, of language.
- **Conditioning data**: the initial text, or prompt, that seeds generation.
- **Greedy sampling**: always picking the most likely token. It produces repetitive, loop-prone text and is a minimum-entropy distribution.
- **Stochastic sampling**: sampling according to the softmax probabilities, so a word with p = 0.3 is picked 30% of the time.
- **Entropy**: uniform sampling has maximum entropy, and greedy sampling has minimum entropy. The model's raw softmax sits in between.
- **Octaves**: the successive image scales at which DeepDream runs gradient ascent, from small to large, with each scale 1.4× the previous one.
- **Detail reinjection**: after upscaling, add back `resize(original, L) − resize(resize(original, S), L)`, the detail lost when going from size S to size L.
- **Content loss**: the L2 distance between upper-layer activations (`block5_conv2` in VGG19) of the base image and of the combination image.
- **Gram matrix**: the inner product of a layer's feature maps, which gives a map of feature correlations. Its statistics correspond to textures at that layer's spatial scale.
- **Style loss**: the L2 distance between Gram matrices of the style image and the combination image, taken over layers from low to high.
- **Total variation loss**: a regularizer on the combination image's pixels that encourages spatial continuity and avoids pixelation.
- **Latent space / generator / decoder**: a low-dimensional vector space in which every point maps to a valid-looking image. In a GAN the mapping module is called the generator. In a VAE it is called the decoder.
- **Concept vector**: a direction in latent space that encodes a meaningful axis of variation. For example, if z is a face, `z + s` is the same face smiling, where s is the smile vector.
- **Reparameterization (sampling layer)**: drawing ε ~ N(0, I) and computing z from `z_mean` and `z_log_var`, so that gradients can flow through the sampling step.
- **KL divergence**: the VAE regularizer. It nudges the encoder's distribution toward a well-rounded normal distribution centered on 0.
- **DCGAN**: a deep convolutional GAN, in which both the generator and the discriminator are convnets.
- **Checkerboard artifacts**: uneven pixel coverage caused by a kernel size that isn't divisible by the stride.

## Mental Models
- **"Generating sequential data is the closest computers get to dreaming"** (Graves). All sequence generation follows the same pattern: predict next, sample, append, repeat. GPT-3 is the chapter's model "with a deep stack of Transformer decoders, and a much bigger training corpus."
- **Language models are all form and no substance.** They capture the statistical structure of the artifacts that language use leaves behind. They don't capture communication, agency, or thought. Language is "the operating system of the mind," so it needs a mind to carry meaning. If human languages were optimally compressed, they would have no statistical structure left to model.
- **Running a convnet in reverse.** DeepDream and style transfer keep the network fixed and optimize the *input*. Lower layers produce geometric patterns. Higher layers produce recognizable ImageNet objects such as dog eyes and bird feathers.
- **Layers as a decomposition over spatial scales.** Content is global and abstract, so you measure it with one upper layer. Style is texture at every scale, so you measure it with correlations across many layers.
- **VAE stochasticity forces continuity.** The encoder is penalized for anything near `z_mean` that decodes badly, so nearby points decode to similar images. Combined with low dimensionality, this makes every direction meaningful.
- **A GAN is a landscape that moves under your feet.** Every step downhill changes the terrain. You are looking for an equilibrium between two forces, and that is why GANs are notoriously hard to train.
- **The generator never sees real data.** Everything it knows about the dataset comes through the discriminator's gradients.
- **VAE vs. GAN.** VAEs give structured, continuous latent spaces, which suit editing and animation. GANs give realism with less continuity. Chollet notes that most practical image applications he has seen rely on VAEs, while GANs dominate academic research.

## Anti-patterns
- **Using only greedy or only temperature-1.0 sampling.** Low temperature gives boring loops, and very high temperature breaks local structure. Always sweep temperatures. About 0.7 worked best for the IMDB model.
- **Training a next-word model on N-word windows only.** The model can't handle short prompts, and the overlapping windows make it redo the same encoding work. Use seq-to-seq with causal masking instead.
- **Over-optimizing DeepDream.** Stop gradient ascent once the loss passes `max_loss`, or you get unwanted artifacts. Exclude border pixels from the loss to avoid edge artifacts.
- **Expecting style transfer to be magic.** It is "closer to classical signal processing than to AI." It retextures images and can't perform abstract transfers such as moving one portrait's style onto another. Widely different image sizes also make it harder.
- **Using max pooling where location matters.** VAE encoders and GAN discriminators should use strided convolutions.
- **Using ReLU and max pooling in GANs.** Both make gradients sparse. Use LeakyReLU and strided convolutions.
- **Using a kernel size that isn't divisible by the stride** in strided Conv2D or Conv2DTranspose layers. This causes checkerboard artifacts.
- **Using plain autoencoders to get a useful latent space.** Classical autoencoders give unstructured latent spaces and compress poorly, and they have largely fallen out of fashion.
- **Letting the discriminator dominate.** If the generator loss climbs while the discriminator loss heads to 0, lower the discriminator's learning rate and raise its dropout.

## Code Examples
```python
import numpy as np

def sample_next(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature   # reweight
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)       # renormalize
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Language-model dataset: targets are inputs shifted by one token
def prepare_lm_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    x = vectorized_sequences[:, :-1]
    y = vectorized_sequences[:, 1:]
    return x, y

# Decoder-only Transformer language model (blocks from chapter 11)
inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
```
- **What it demonstrates**: temperature sampling, the shifted-target dataset setup, and the fact that a GPT-style generator is just a causal Transformer decoder trained on next-token prediction.

```python
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    return tf.matmul(features, tf.transpose(features))

def style_loss(style_img, combination_img):
    S, C = gram_matrix(style_img), gram_matrix(combination_img)
    channels, size = 3, img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base_img, combination_img):
    return tf.reduce_sum(tf.square(combination_img - base_img))

# The combination image is a tf.Variable optimized directly
optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))
for i in range(1, 4001):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image)
    optimizer.apply_gradients([(grads, combination_image)])
```
- **What it demonstrates**: style as Gram-matrix correlations, content as raw activation distance, and gradient descent on pixels with a learning-rate schedule. The book substitutes SGD for L-BFGS, which TensorFlow doesn't provide.

```python
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):  # __init__ / metrics omitted
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                              - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"total_loss": self.total_loss_tracker.result()}
```
- **What it demonstrates**: the self-supervised custom `train_step` pattern from chapter 7. Compile with no loss and call `fit(x)` with no targets. It also shows the two-part VAE loss.

## Reference Tables
**Sampling temperature (IMDB word-level LM, 200 epochs)**
| T | Behavior |
|---|---|
| 0.2 | Boring, repetitive, gets stuck in loops |
| 0.5–0.7 | Coherent and more interesting; about 0.7 was the best balance |
| 1.0 | Creative, with looser structure |
| 1.5 | Local structure breaks down and output is near random |

**Defaults used in the chapter**
| Technique | Settings |
|---|---|
| Text LM | vocab 15,000; sequence length 100; embed_dim 256; latent_dim 2048; 2 heads; RMSprop; 200 epochs; prompt "this movie" |
| DeepDream | InceptionV3; layers mixed4–7 weighted 1.0/1.5/2.0/2.5; step 20; 3 octaves; scale 1.4; 30 iterations; max_loss 15; works best on images of about 300–400 px |
| Style transfer | VGG19; content layer block5_conv2; style layers block1–5_conv1; content_weight 2.5e-8; style_weight 1e-6; TV weight 1e-6; height 400 px; SGD with learning rate 100, decayed ×0.96 every 100 steps; 4,000 iterations |
| VAE (MNIST) | latent_dim 2; strided Conv2D 32 and 64; Dense 16; Conv2DTranspose decoder; sigmoid output; Adam; 30 epochs; batch 128 |
| DCGAN (CelebA 64×64) | latent_dim 128; kernel 4 with stride 2; LeakyReLU(0.2); discriminator Dropout 0.2; Adam with lr 1e-4 for both networks; label noise 0.05; about 100 epochs, with interesting results after about epoch 20 |

**GAN bag of tricks**
| Trick | Why |
|---|---|
| Strided convs, not pooling | Avoid sparse gradients and keep location information |
| Sample latents from a normal distribution, not uniform | Empirically works better |
| Add random noise to labels | Stochasticity keeps the dynamic equilibrium from getting stuck |
| LeakyReLU instead of ReLU | Relaxes sparsity |
| Kernel size divisible by stride | Prevents checkerboard artifacts |
| Dropout in the discriminator | Stops the generator from getting stuck producing noise |

## Worked Example
**Training a DCGAN on faces.** Load CelebA with `image_dataset_from_directory(label_mode=None, image_size=(64, 64), smart_resize=True)`. Smart resize crops as well as scales, so face proportions aren't distorted. Then rescale the images to [0, 1].

The discriminator is three Conv2D(k=4, s=2) layers with LeakyReLU, followed by Flatten, Dropout(0.2), and Dense(1, sigmoid). The generator starts with Dense(8·8·128) and a Reshape, then three Conv2DTranspose(k=4, s=2) layers with 128, 256, and 512 filters plus LeakyReLU, and ends with Conv2D(3, k=5, sigmoid) to produce 64×64×3 images.

The `GAN(keras.Model)` subclass overrides `compile` to accept two optimizers and a loss function. Each `train_step` does the following:
1. Sample normal latent vectors and generate fake images.
2. Concatenate the fake and real images, with labels 1 for fake and 0 for real, plus 0.05 uniform noise.
3. Update the discriminator.
4. Sample new latent vectors and pass them through `discriminator(generator(z))` with "misleading" labels that say real (all zeros).
5. Update only the generator's weights.

A `GANMonitor` callback saves sample images every epoch. If the discriminator loss collapses toward 0 while the generator loss rises, lower the discriminator's learning rate or raise its dropout.

## Key Takeaways
- To generate sequences, train next-token prediction and then sample in a loop. The sampling strategy matters as much as the model, so sweep temperatures.
- Seq-to-seq training with causal masking trains N overlapping prediction problems at once and lets the model generate from very short prompts.
- DeepDream (gradient ascent on activations) and style transfer (gradient descent on content, style, and TV losses) optimize pixels against a frozen pretrained convnet. The idea also applies to speech and music.
- Style transfer is texture transfer. To make it fast, train a small feedforward convnet on input-output pairs generated with the slow method for a fixed style.
- Image generation means learning a latent space and decoding sampled points from it. VAEs give continuity and concept vectors. GANs give realism.
- GANs are difficult to train and depend on heuristic tricks and extensive tuning.

## Connects To
- **Chapter 7**: subclassing `Model` with a custom `train_step` and `metrics` property, used here for the VAE and the GAN. Callbacks are used for monitoring generation.
- **Chapter 9**: filter visualization (DeepDream is its multiscale form). Strided convolutions vs. pooling come from the segmentation example. Pretrained ImageNet convnets.
- **Chapter 10**: LSTMs and sequence prediction. A temperature-forecasting model could be reframed as seq-to-seq.
- **Chapter 11**: `TextVectorization`, `PositionalEmbedding`, `TransformerDecoder`, and causal masking. Word embeddings anticipate concept vectors.
- **Chapter 14**: limits of deep learning. Language models capture form without meaning.
