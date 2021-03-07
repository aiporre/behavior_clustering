import tensorflow as tf


@tf.function
def triplet_loss(x_a, x_p, x_n, margin=1.0):
    d_p = tf.square(tf.norm(x_a - x_p, ord='euclidean', axis=-1))
    d_n = tf.square(tf.norm(x_a - x_n, ord='euclidean', axis=-1))
    L = tf.maximum(0.0, d_p - d_n + margin)
    return tf.reduce_mean(L)

