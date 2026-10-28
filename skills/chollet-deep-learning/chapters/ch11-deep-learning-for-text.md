# Chapter 11: Deep Learning for Text

## Core Idea
Modern NLP doesn't try to make machines "understand" language. It is pattern recognition applied to words, sentences and paragraphs: ingest text, return something useful (topic, sentiment, next word, translation, summary). Every text model first runs a vectorization pipeline (standardize, tokenize, index, encode). After that, the central design question is **how to represent word order**. Three answers produce three model families. Bag-of-words models discard order. RNNs process tokens strictly in order. The Transformer is a hybrid: order-agnostic attention plus injected position information. Choosing between bag-of-words and sequence models is an empirical question with a measurable rule of thumb. For sequence-to-sequence tasks such as translation, the encoder-decoder Transformer is the tool of choice.

Historical arc: handcrafted rules (ELIZA era) gave way to machine learning in the late 1980s (decision trees, then logistic regression). LSTMs dominated 2015-2017, and bidirectional LSTMs set the state of the art. The Transformer took over from 2017-2018.

## Frameworks Introduced
- **The text vectorization template**: standardization, then tokenization, then indexing, then one-hot encoding or embedding.
  - When to use: always. Neural networks only take numeric tensors.
  - How: use the Keras `TextVectorization` layer. Call `adapt()` on a dataset of raw strings to build the vocabulary (and TF-IDF weights). Pick `output_mode` from `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`. `ngrams=N` adds N-grams. `output_sequence_length` pads or truncates.
- **Sets vs. sequences (bag-of-words vs. sequence models)**: bag-of-words models treat text as an unordered set of tokens or N-grams and run them through Dense layers. Sequence models (RNNs, 1D convnets, Transformers) look at ordered word sequences and learn order-based features themselves.
  - When to use: decide with the samples-to-length ratio below.
  - How: for bag-of-words, use `multi_hot` or `tf_idf` output with `ngrams=2` and a small Dense stack. For sequence models, use `int` output, then an Embedding layer, then RNN or Transformer layers.
- **The samples-to-mean-length ratio heuristic** (from Chollet's 2017 systematic study): compute (number of training samples) / (mean words per sample).
  - When to use: at the start of any new text-classification task.
  - How: if the ratio is below about 1,500, use a bag-of-bigrams model, which is also faster to train and iterate on. If it is above 1,500, use a sequence model. The heuristic is for text classification only; it is a rule of thumb, not a law.
- **The query-key-value model of attention**: `outputs = sum(values * pairwise_scores(query, keys))`. For each element of the query, score its relation to every key, then use those scores to weight a sum of values. The terminology comes from search engines and recommender systems.
  - When to use: self-attention (query = keys = values = the same sequence) for classification and encoding. Cross-attention (query = target, keys = values = encoded source) for translation.
- **The Transformer encoder**: MultiHeadAttention, then residual add and LayerNormalization, then a dense projection (Dense relu, then Dense embed_dim), then residual add and LayerNormalization.
- **The Transformer decoder**: causal self-attention over the target, then add and norm, then cross-attention whose keys and values are the encoder output, then add and norm, then the dense projection, then add and norm.
- **Sequence-to-sequence learning template**: during training, an encoder turns the source into an intermediate representation, and a decoder predicts target token i from target tokens 0..i-1 plus the encoded source ("teacher forcing" with the target offset by one step). During inference, start from a seed token `[start]`, predict the next token, feed it back in, and repeat until `[end]` or the maximum length.

## Key Concepts
- **Text standardization**: basic feature engineering that erases encoding differences you don't want the model to deal with: lowercasing, stripping punctuation, normalizing accents (é to e), and, more rarely, **stemming**. It reduces the data needed and improves generalization, but it can erase signal. Keep "?" if you're extracting questions.
- **Tokenization**: word-level (optionally split into subwords such as "star+ing"), N-gram, or character-level (rare; used in text generation and speech). Sequence models use word-level tokens. Bag-of-words models use N-grams, which are "a way to artificially inject a small amount of local word order information."
- **Vocabulary indexing**: map each token to a unique integer. Restrict the vocabulary to the top 20,000-30,000 terms, because rare terms add features with almost no information.
- **Mask token (index 0) and OOV token (index 1, "[UNK]")**: 0 means "ignore me, I'm not a word" and is used for padding. 1 means "a word we did not recognize."
- **Multi-hot / count / TF-IDF encodings**: presence flags, raw counts, and counts reweighted by document frequency. Normalization must be **divide-only**, because subtracting the mean would wreck **sparsity**.
- **Word embeddings**: dense, low-dimensional (256, 512 or 1,024), learned, structured vectors in which geometric distance reflects semantic distance and directions carry meaning ("from pet to wild animal", gender, plural vectors). One-hot vectors are sparse, high-dimensional and hardcoded, and they assume all words are orthogonal. That assumption is a feature-engineering decision, and it is wrong for language.
- **Embedding layer**: effectively a dictionary lookup from integer index to dense vector. Input shape is `(batch, seq_len)` and output is `(batch, seq_len, embed_dim)`. It starts random and is trained by backpropagation.
- **Pretrained word embeddings**: Word2Vec (Mikolov, Google, 2013) and GloVe (Stanford, 2014, which factorizes a co-occurrence matrix). Load them via `Constant` initializer and freeze them. The idea traces to Bengio et al. (2003).
- **Masking**: `mask_zero=True` on Embedding generates a boolean `(batch, seq_len)` mask. Keras propagates it automatically to layers that support it: RNNs skip masked steps, and the loss skips them for sequence outputs.
- **Self-attention**: modulates each token's representation using the representations of related tokens. Score with the dot product, scale by sqrt(dim), apply softmax, then take the weighted sum. This produces **context-aware** token vectors ("station" absorbs part of "train").
- **Multi-head attention**: query, key and value each go through independent learned dense projections per head. Each head attends separately, and the results are concatenated. This lets the layer learn and factor features into independent subspaces, the same principle as depthwise separable and grouped convolutions.
- **LayerNormalization**: normalizes each sequence independently over the last axis. BatchNormalization pools over the batch axis and works poorly on sequences.
- **Positional embedding**: self-attention is a set-processing mechanism and is blind to order. Re-inject order by adding a learned position-embedding vector to each token embedding. The original paper used cyclic cosine positional encoding with values in [-1, 1]. Raw integer positions are bad because they produce large, discrete values. The downside of learned positions is that the sequence length must be known in advance.
- **Causal (attention) mask**: a lower-triangular mask over the decoder's self-attention. Without it, the order-agnostic decoder would learn to copy token N+1 to output N, reach perfect training accuracy, and be useless at inference.
- **BLEU score**: the proper metric for translation. It looks at whole generated sequences. Next-token accuracy is only a crude proxy because it assumes the earlier target tokens are correct.

## Mental Models
- **Feature engineering vs. learned features**: N-grams are hand-engineered order features. Sequence models replace them with hierarchical feature learning from raw sequences. This is the deep learning story repeated for text.
- **Order vs. context matrix (fig. 11.10)**:

| Model | Word-order aware | Context aware (cross-word interactions) |
|---|---|---|
| Bag-of-unigrams | No | No |
| Bag-of-bigrams | Very limited | No |
| RNN | Yes | No |
| Self-attention | No | Yes |
| Transformer | Yes | Yes |

- **Attention as soft retrieval**: the query is a search ("dogs on the beach"), the keys are the photo tags, and the values are the photos. Return a relevance-weighted sum of values.
- **Spectrum of attention**: max pooling is all-or-nothing attention. TF-IDF is continuous attention over tokens. Self-attention is learned, contextual attention.
- **RNN seq2seq as translating from memory**: the entire source must squeeze into a fixed state vector, and the network forgets the distant past. A Transformer decoder is the translator who glances back and forth at the source while writing.
- **Why the ratio heuristic works**: sequence inputs span a richer space that takes more data to map out. Short samples can't afford to lose word order ("this movie is the bomb" vs. "this movie was a bomb"). In long samples, word statistics alone become reliable.

## Anti-patterns
- Assuming bag-of-words is obsolete. On IMDB (about 20,000 samples, 233 mean words, ratio about 86) a bag-of-bigrams beat every sequence model in the chapter.
- One-hot encoding words as input to a sequence model. On IMDB this produced (600, 20000) = 12M floats per review, very slow training, and only 87% accuracy.
- Leaving padding unmasked with RNNs. The forward RNN spends hundreds of steps on zeros, and its state fades.
- Subtracting the mean from sparse bag-of-words vectors, which destroys sparsity.
- Putting `TextVectorization` inside a GPU-trained model during training. It runs only on CPU, so the GPU waits. Put it in the `tf.data` pipeline instead, then wrap it into an inference model for export so production doesn't need to reimplement preprocessing (which risks train/serve skew).
- Using a Transformer encoder without positional information and calling it a sequence model. Scrambled word order would give identical outputs.
- Training a Transformer decoder without a causal mask.
- Using BatchNormalization for sequence models instead of LayerNormalization.
- Writing custom layers without `get_config()`, so models can't be reloaded. Pass `custom_objects` when loading.
- Naive per-step RNN output for seq2seq: the source and target must have the same length, and token N is predicted from source tokens 0..N only.

## Code Examples
```python
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.dense_dim, self.num_heads = embed_dim, dense_dim, num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]   # 2D padding mask -> 3D for attention
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim,
                       "num_heads": self.num_heads, "dense_dim": self.dense_dim})
        return config
```
- **What it demonstrates**: the encoder block is attention plus a dense projection, each wrapped in a residual connection and LayerNorm. It is shape-invariant, so blocks can be stacked. `get_config` makes the layer serializable.

```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length, self.input_dim, self.output_dim = (
            sequence_length, input_dim, output_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)   # propagate padding mask downstream
    # get_config(): return output_dim, sequence_length, input_dim (as above)

# Text-classification Transformer (IMDB): 88.3% test accuracy
inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(600, 20000, 256)(inputs)
x = TransformerEncoder(256, 32, 2)(x)
x = layers.GlobalMaxPooling1D()(x)          # reduce sequence to one vector
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```
- **What it demonstrates**: learned position embeddings are added to token embeddings to make the set-processing encoder order-aware. A custom `compute_mask` keeps padding masking working.

```python
def call(self, inputs, encoder_outputs, mask=None):   # TransformerDecoder.call
    causal_mask = self.get_causal_attention_mask(inputs)  # lower-triangular (i >= j)
    padding_mask = None
    if mask is not None:
        padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        padding_mask = tf.minimum(padding_mask, causal_mask)
    attn_1 = self.attention_1(query=inputs, value=inputs, key=inputs,
                              attention_mask=causal_mask)
    attn_1 = self.layernorm_1(inputs + attn_1)
    attn_2 = self.attention_2(query=attn_1, value=encoder_outputs,
                              key=encoder_outputs, attention_mask=padding_mask)
    attn_2 = self.layernorm_2(attn_1 + attn_2)
    proj_output = self.dense_proj(attn_2)
    return self.layernorm_3(attn_2 + proj_output)
```
- **What it demonstrates**: causal self-attention over the target, then cross-attention in which the target is the query and the encoded source supplies keys and values. The decoder also sets `self.supports_masking = True` in `__init__`, because masking in Keras is opt-in.

## Reference Tables
**IMDB results in the chapter** (20k train, 5k val, 25k test; naive baseline 50%; about 95% is the best achievable without external data):

| Model | Test acc. |
|---|---|
| Binary unigram, Dense(16) | 89.2% |
| Binary bigram | **90.4%** (best) |
| TF-IDF bigram | 89.8% (usually about +1 point over binary on other datasets) |
| One-hot + BiLSTM(32) | 87% (very slow) |
| Embedding(256) + BiLSTM | 87% |
| Same with `mask_zero=True` | 88% |
| GloVe 100d frozen + BiLSTM | no gain here (helps on small datasets) |
| Transformer encoder, no positions | 87.5% |
| Transformer encoder + positional embedding | 88.3% |

**Defaults and numbers**: vocabulary 20,000 for classification. `max_length=600` for IMDB (mean review is 233 words; only 5% exceed 600). Validation split 20%. `num_parallel_calls=4`. Use `.cache()` on preprocessed datasets that fit in memory. Translation setup: vocab 15,000 per language, sequence length 20 (target 21 for the offset), batch 64, 15% val / 15% test, `embed_dim=256`, `dense_dim=2048`, `num_heads=8`. GRU seq2seq uses `latent_dim=1024` and reaches 64% next-token accuracy in 15 epochs. The Transformer reaches 67% in 30 epochs.

## Worked Example
**English-to-Spanish translation.** Wrap each Spanish sentence as `"[start] ... [end]"`. Use two `TextVectorization` layers. The Spanish one has custom standardization that keeps `[` and `]` and also strips `¿`. The dataset yields `({"english": eng, "spanish": spa[:, :-1]}, spa[:, 1:])`, so the decoder input and target are the same sentence offset by one step.

The **RNN baseline** encodes the source with a `Bidirectional(GRU(1024), merge_mode="sum")` and uses that as the `initial_state` of a decoder `GRU(return_sequences=True)`, followed by `Dense(vocab, softmax)`. Use GRU rather than LSTM because it has a single state vector. Masking is critical. Train with `sparse_categorical_crossentropy`.

Inference loops: vectorize the decoded-so-far string, predict, take the argmax at step i, append the token, and stop at `[end]`. This is inefficient, since it reprocesses everything at every step. A real system splits the encoder and decoder and runs one decoder step at a time with cached state.

The RNN approach has two limits. The whole source must fit in the state vector, and long-range memory fades. The **Transformer** replaces the RNN with `PositionalEmbedding`, `TransformerEncoder` on the source, and `PositionalEmbedding`, `TransformerDecoder(x, encoder_outputs)`, Dropout and softmax on the target. Translations are subjectively much better, but they show bias: a gender-neutral source is translated as male.

## Key Takeaways
- Every text pipeline follows the same steps: standardize, tokenize, index (with mask 0 and OOV 1), then encode. Use `TextVectorization`, running it in `tf.data` for GPU training and embedding it in the exported model for serving.
- Run the samples-to-mean-length ratio first. Below 1,500, use a bag-of-bigrams with Dense layers. Above 1,500, use a sequence model (Transformer).
- For sequence models, prefer learned embeddings over one-hot encoding, always enable masking, and use pretrained embeddings (GloVe, Word2Vec) only when data is scarce.
- Self-attention gives context-aware representations but no order. The Transformer equals attention plus positional information plus residual connections, LayerNorm and dense projections.
- Seq2seq uses an encoder, a decoder trained on the target offset by one step, and autoregressive inference from `[start]`. A Transformer decoder needs a causal mask. Evaluate translation with BLEU, not next-token accuracy.
- Depthwise-separable 1D convnet stacks can match BiLSTMs at much lower cost (the author's experience), though they are rarely used in NLP.

## Connects To
- Ch 4-5: the prevectorized IMDB data with multi-hot encoding and `num_words=10000`.
- Ch 8: `image_dataset_from_directory` mirrors `text_dataset_from_directory`, and pretrained convnets mirror pretrained embeddings.
- Ch 9: residual connections, normalization, and depthwise separable convolutions explain the Transformer's design choices.
- Ch 10: RNNs, LSTM/GRU and bidirectional RNNs; Google Translate circa 2017 used seven stacked LSTM layers.
- Ch 12: text generation with Transformers (language models, sampling strategies) builds directly on this decoder.
- Ch 7: subclassed layers, `get_config`, and `custom_objects` loading.
