"""losses used throughout the project"""
import tensorflow as tf
from keras.losses import binary_crossentropy


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
    """computes the intersection over union metric, we add smooth to both num and den"""
    y_pred = tf.cast(tf.math.greater(y_pred, threshold), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    return tf.reduce_mean(numerator / denominator)


def bce_dice_loss_unet(y_true, y_pred, weight=0.9, axis=(1, 2), smooth=1.):
    """inspired from https://stackoverflow.com/questions/42591191/keras-semantic-segmentation-weighted-loss-pixel-map
    same as bce dice loss but the last column of the y_pred is the weight map"""

    weight_map = y_true[:, :, :, -1:]
    y_true = y_true[:, :, :, :-1]

    normalize_weight_map = tf.reduce_mean(weight_map)

    # cross entropy
    loss_map = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true, logits=y_pred)  # loss_map of size
    # batch_size*width*height
    weighted_loss = tf.multiply(loss_map, weight_map)  # weight the loss of every pixel
    weighted_ce_loss = tf.reduce_mean(weighted_loss) / normalize_weight_map

    # dice loss
    intersection = tf.reduce_sum((y_true * y_pred) * weight_map, axis=axis) / normalize_weight_map
    union = tf.reduce_sum((y_true + y_pred) * weight_map, axis=axis) / normalize_weight_map
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    weighted_dice_loss = 1 - tf.reduce_mean(numerator / denominator)

    # total loss
    return (1 - weight) * weighted_ce_loss + weight * weighted_dice_loss


def i_o_u_metric_unet(y_true, y_pred):
    """iou metric for which we do not take the weight_masks into account"""
    return i_o_u_metric(y_true[:, :, :, :-1], y_pred)


def i_o_u_metric_first_mask(y_true, y_pred):
    """iou metric for the first predicted mask"""
    return i_o_u_metric(y_true[:, :, :, :1], y_pred[:, :, :, :1])


def i_o_u_metric_second_mask(y_true, y_pred):
    """iou metric for the second predicted mask"""
    return i_o_u_metric(y_true[:, :, :, 1:2], y_pred[:, :, :, 1:2])
