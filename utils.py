
import tensorflow as tf


# focal loss
def focal_loss(labels, y_pred, alpha=0.25, gamma=2):

    down = 1e-8
    up = 1.0 - 1e-8
    y_pred = tf.clip_by_value(y_pred, down, up)
    loss = -labels * (1 - alpha) * ((1 - y_pred) * gamma) * tf.math.log(y_pred) - \
           (1 - labels) * alpha * (y_pred ** gamma) * tf.math.log(1 - y_pred)

    return loss


## gated attention based multi-instance pooling layer
class MIL_gated_attention(tf.keras.layers.Layer):

    def __init__(self, d_model):

        super(MIL_gated_attention, self).__init__()

        self.w1 = tf.keras.layers.Dense(d_model)
        self.w2 = tf.keras.layers.Dense(d_model)
        self.w3 = tf.keras.layers.Dense(d_model)


    def call(self, x):

        # linear projection
        alpha = tf.tanh(self.w1(x))

        # gate mechanism
        gate = tf.nn.sigmoid(self.w2(x))
        alpha = self.w3(tf.multiply(alpha, gate))

        # attention weights
        attention_weights = tf.nn.softmax(alpha)

        # output
        output = tf.multiply(x, attention_weights)
        output = tf.reduce_mean(output, axis=-1)

        return output, attention_weights


# multi-head attention layer
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):

        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    # scaled dot product attention
    def scaled_dot_product_attention(self, q, k, v):

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights
