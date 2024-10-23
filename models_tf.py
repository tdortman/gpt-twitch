import tensorflow as tf
from tensorflow.keras import layers, Model


class CausalSelfAttention(layers.Layer):
    def __init__(self, n_head, n_embd, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head

        self.qkv_proj = layers.Dense(3 * n_embd, use_bias=False)
        self.out_proj = layers.Dense(n_embd)
        self.attn_dropout = layers.Dropout(dropout)
        self.resid_dropout = layers.Dropout(dropout)

    def create_causal_mask(self, seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.cast(mask, dtype=tf.bool)[tf.newaxis, tf.newaxis, :, :]

    def call(self, inputs, training=False):
        B, T, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        causal_mask = self.create_causal_mask(T)

        qkv = self.qkv_proj(inputs)
        qkv = tf.reshape(qkv, [B, T, 3, self.n_head, self.head_size])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=0)

        scale = tf.cast(self.head_size, tf.float32) ** -0.5
        attn = tf.matmul(q, k, transpose_b=True) * scale

        attn = tf.where(causal_mask[:, :, :T, :T], tf.float32.min, attn)

        attn = tf.nn.softmax(attn)
        attn = self.attn_dropout(attn, training=training)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, T, C])

        return self.resid_dropout(self.out_proj(out), training=training)


class MLP(layers.Layer):
    def __init__(self, n_embd, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(4 * n_embd, activation="gelu")
        self.fc2 = layers.Dense(n_embd)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.dropout(x, training=training)


class Block(layers.Layer):
    def __init__(self, n_embd, n_head, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(n_head, n_embd, dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = MLP(n_embd, dropout)

    def call(self, inputs, training=False):
        x = inputs + self.attn(self.ln1(inputs), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


class GPTLanguageModel(Model):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = layers.Embedding(vocab_size, n_embd)
        self.position_embedding = layers.Embedding(block_size, n_embd)

        self.blocks = [Block(n_embd, n_head, dropout) for _ in range(n_layer)]

        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.out_proj = layers.Dense(vocab_size, use_bias=False)

        self.drop = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        _, T = tf.shape(inputs)[0], tf.shape(inputs)[1]

        tf.debugging.assert_less_equal(T, self.block_size)

        tok_emb = self.token_embedding(inputs)
        pos = tf.range(0, T, dtype=tf.int32)[tf.newaxis, :]
        pos_emb = self.position_embedding(pos)

        x = self.drop(tok_emb + pos_emb, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.ln_f(x)
        logits = self.out_proj(x)

        return logits

    def generate(self, context, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            context_cond = context[:, -self.block_size :]

            logits = self(context_cond, training=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = tf.nn.top_k(logits, min(top_k, tf.shape(logits)[1]))
                logits = tf.where(logits < v[:, [-1]], -float("inf"), logits)

            probs = tf.nn.softmax(logits)
            next_token = tf.random.categorical(probs, num_samples=1)

            context = tf.concat([context, next_token], axis=1)

        return context
