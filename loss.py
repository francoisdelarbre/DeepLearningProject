"""losses used throughout the project"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


def dice_loss(y_true, y_pred, axis=(1, 2), smooth=1.):
    """(soft) dice loss"""
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    return tf.reduce_mean(numerator / denominator)


def bce_dice_loss(y_true, y_pred, weight=0.9):
    """combination of binary cross entropy and dice loss"""
    return (1 - weight) * binary_crossentropy(y_true, y_pred) + weight * dice_loss(y_true, y_pred)
