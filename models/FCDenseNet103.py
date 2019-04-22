"""FC-DenseNet103 model, as described in http://arxiv.org/abs/1611.09326, when in doubt with the details of the paper,
we inspired ourselves from https://github.com/0bserver07/One-Hundred-Layers-Tiramisu/blob/master/model-tiramasu-67.py
and the official implementation https://github.com/SimJeg/FC-DenseNet"""
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    Concatenate, ReLU, Dropout
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.regularizers import l2

REG = l2(1e-4)
GROWTH_RATE = 16

def fc_densenet103(inputs, num_classes):
    """function that builds a FC-DenseNet103 model, as described in http://arxiv.org/abs/1611.09326
        the function is taking the inputs as arguments"""
    x = Conv2D(48, kernel_size=3, padding='same', activation=None, kernel_initializer="he_uniform",
               kernel_regularizer=REG)(inputs)

    # downsampling
    x_224 = dense_subnetwork(x, 112, 4)(x)

    x_112 = transition_down(x_224, 112)
    x_112 = dense_subnetwork(x_112, 192, 5)

    x_56 = transition_down(x_112, 192)
    x_56 = dense_subnetwork(x_56, 304, 7)

    x_28 = transition_down(x_56, 304)
    x_28 = dense_subnetwork(x_28, 464, 10)

    x_14 = transition_down(x_28, 464)
    x_14 = dense_subnetwork(x_14, 656, 12)

    x_7 = transition_down(x_14, 656)

    # middle
    x_7 = dense_subnetwork(x_7, 896, 15)

    # upsampling
    x_14_up = transition_up(x_7, 1088)
    x_14_up = dense_subnetwork([x_14_up, x_14], 1088, 12)

    x_28_up = transition_up(x_14_up, 816)
    x_28_up = dense_subnetwork([x_28_up, x_28], 816, 10)

    x_56_up = transition_up(x_28_up, 578)
    x_56_up = dense_subnetwork([x_56_up, x_56], 578, 7)

    x_112_up = transition_up(x_56_up, 384)
    x_112_up = dense_subnetwork([x_112_up, x_112], 384, 5)

    x_224_up = transition_up(x_112_up, 256)
    x_224_up = dense_subnetwork([x_224_up, x_224], 256, 4)

    # classification layer
    return Conv2D(num_classes, kernel_size=1, padding='same', activation='sigmoid', kernel_initializer="he_uniform",
               kernel_regularizer=REG)(x_224_up)


def dense_subnetwork(inputs, num_out_channels, num_blocks):
    """applies a dense_subnetwork with the given number of output channels and the given number of blocks
    returns the output and takes the inputs as argument"""
    outputs = []
    inputs = [inputs]
    for i in range(num_blocks):
        x = Concatenate(inputs)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, beta_regularizer=REG,
                               gamma_regularizer=REG)(x)
        x = ReLU()(x)
        x = Conv2D(GROWTH_RATE, kernel_size=3, use_bias=False, padding='same', kernel_initializer="he_uniform",
                   kernel_regularizer=REG, activation=None)(x)
        x = Dropout(0.2)(x)
        inputs.append(x)
        outputs.append(x)
    return Concatenate(outputs)


def transition_down(inputs, num_channels):
    """applies a transition_down to the input"""
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, beta_regularizer=REG, gamma_regularizer=REG)(inputs)
    x = ReLU()(x)
    x = Conv2D(num_channels, kernel_size=1, use_bias=False, padding='same', kernel_initializer="he_uniform",
               kernel_regularizer=REG, activation=None)(x)
    x = Dropout(0.2)(x)
    return MaxPooling2D(pool_size=2, strides=2, padding='same')(x)


def transition_up(inputs, num_channels):
    """applies the transition_up to the list of inputs"""
    x = Concatenate(inputs)
    return Conv2DTranspose(num_channels, kernel_size=3, strides=2, padding='same', activation=None,
                           kernel_initializer="he_uniform", kernel_regularizer=REG)(x)