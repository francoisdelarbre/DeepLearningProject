"""losses used throughout the project"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


def dice_loss(y_true, y_pred, axis=(1, 2), smooth=1.):
    """(soft) dice loss"""
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    return 1 - tf.reduce_mean(numerator / denominator)


def bce_dice_loss(y_true, y_pred, weight=0.9):
    """combination of binary cross entropy and dice loss"""
    return (1 - weight) * binary_crossentropy(y_true, y_pred) + weight * dice_loss(y_true, y_pred)


def i_o_u_metric(y_true, y_pred, threshold=0.5, axis=(1, 2), smooth=1.):
    """computes the intersection over union metric without smoothing factor, we add smooth to both num and den"""
    y_pred = tf.cast(tf.math.greater(y_pred, threshold), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    return tf.reduce_mean(numerator / denominator)
