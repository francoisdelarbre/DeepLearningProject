"""FC-DenseNet103 model, as described in http://arxiv.org/abs/1611.09326, when in doubt with the details of the paper,
we inspired ourselves from https://github.com/0bserver07/One-Hundred-Layers-Tiramisu/blob/master/model-tiramasu-67.py
and the official implementation https://github.com/SimJeg/FC-DenseNet"""
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2D, MaxPooling2D, Conv2DTranspose, \
    Concatenate, Add, ReLU, Dropout
import tensorflow.keras.backend as keras_backend
from keras.regularizers import l2

REG = l2(1e-4)


def fc_densenet103(inputs, num_classes):
    """function that builds a FC-DenseNet103 model, as described in http://arxiv.org/abs/1611.09326
        the function is taking the inputs as arguments"""




def dense_subnetwork(inputs, num_out_channels, num_blocks):
    """applies a dense_subnetwork with the given number of output channels and the given number of blocks
    returns the output and takes the inputs as argument"""
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, beta_regularizer=REG, gamma_regularizer=REG)(inputs)
    x = ReLU()(x)
    x = Conv2D(320, kernel_size=1, use_bias=False, name='Conv_1_in')(x)
