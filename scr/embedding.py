import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Activation


def circular_cross_correlation(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be of the same length.
    """
    return tf.math.real(tf.signal.ifft(
        tf.multiply(tf.math.conj(tf.signal.fft(tf.cast(x, tf.complex64))), tf.signal.fft(tf.cast(y, tf.complex64)))))


def HolE(s, p, o):
    # sigm(p^T (s \star o))
    # dot product in tf: sum(multiply(a, b) axis = 1)
    l = Lambda(lambda x: tf.reduce_sum(tf.multiply(x[1], circular_cross_correlation(x[0], x[2])), axis=-1))
    score = l([s, p, o])
    score = Activation('sigmoid', name='score')(score)
    return score


def TransE(s, p, o):
    l = Lambda(lambda x: 1 / tf.norm(x[0] + x[1] - x[2], axis=-1))
    score = l([s, p, o])
    score = Activation('sigmoid', name='score')(score)  # Activation('tanh', name='score')(score)
    return score


def DistMult(s, p, o):
    l = Lambda(lambda x: K.sum(x[0] * x[1] * x[2], axis=-1))
    score = l([s, p, o])
    score = Activation('sigmoid', name='score')(score)
    return score