"""the resnext related code is inspired/adapted from
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py, in particular since we
use it for the encoder and we need to get the names of the output channels right and for the auxiliary functions which
were basically copied"""
from keras.layers import Conv2D, Concatenate
from keras_applications.resnet import ResNet50
import keras.backend as backend
import keras.layers as layers
import keras.models as models
import keras.utils as utils


def unet_resnet50(inputs, num_classes, shape):
    """function that builds a UNet-like model whose encoder is resnet (weights pretrained on imagenet and whose
    decoder is a mirror image of the encoder
    the function is taking the inputs as arguments"""
    encoder = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs, input_shape=shape, pooling=None,
                       backend=backend, layers=layers, models=models, utils=utils)
    num_blocks = [3, 4, 6, 3]

    num_channels = [64, 64, 128, 256, 512]
    encoder_112 = encoder.get_layer('conv1_relu').output
    encoder_56 = encoder.get_layer('conv2_block3_out').output
    encoder_28 = encoder.get_layer('conv3_block4_out').output
    encoder_14 = encoder.get_layer('conv4_block6_out').output
    encoder_end = encoder.get_layer('conv5_block3_out').output

    # upsampling
    x_14_up = stack_with_upsampling(encoder_end, num_channels[4], num_blocks[3], name='conv5up')

    x_14_up = Concatenate()([encoder_14, x_14_up])
    x_28_up = stack_with_upsampling(x_14_up, num_channels[3], num_blocks[2], name='conv4up')

    x_28_up = Concatenate()([encoder_28, x_28_up])
    x_56_up = stack_with_upsampling(x_28_up, num_channels[2], num_blocks[1], name='conv3up')

    x_56_up = Concatenate()([encoder_56, x_56_up])
    x_112_up = stack_with_upsampling(x_56_up, num_channels[1], num_blocks[0], name='conv2up')

    x_112_up = Concatenate()([encoder_112, x_112_up])
    x_112_up = stack_with_upsampling(x_112_up, num_channels[0], num_blocks[0], up_stride_last=1, name='conv1up')

    x_224_up = stack_with_upsampling(x_112_up, num_channels[0], 2, up_stride_last=1, name='conv0up')  # added 2 blocks
    # at the end to process these images as well

    output = Conv2D(num_classes, kernel_size=1, activation="sigmoid")(x_224_up)

    return output


def block(x, filters, kernel_size=3, up_stride=1, conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        up_stride: default 1, stride of the first layer, a positive stride means upsampling
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        if up_stride == 1:
            shortcut = layers.Conv2D(4 * filters, 1, name=name + '_0_conv')(x)
        else:
            shortcut = layers.Conv2DTranspose(4 * filters, 1, strides=up_stride, use_bias=False,
                                              padding='same', name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    if up_stride == 1:
        x = layers.Conv2D(filters, 1, name=name + '_1_conv')(x)
    else:
        x = layers.Conv2DTranspose(filters, 1, strides=up_stride, use_bias=False,
                                   padding='same', name=name + '_0_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack_with_upsampling(x, filters, blocks, up_stride_last=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        up_stride_last: default 2, stride of the first layer in the last block, a stride > 1 means upsampling
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block(x, filters, name=name + '_block1')
    for i in range(2, blocks):
        x = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    x = block(x, filters, up_stride=up_stride_last, name=name + '_block' + str(blocks))
    return x
