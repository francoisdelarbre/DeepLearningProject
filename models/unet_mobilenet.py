"""the mobilenet-v2 related code is inspired/adapted from
https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py, in particular since we
use it for the encoder and we need to get the names of the output channels right and for the _make_divisible and
_inverted_res_block functions"""
from keras.layers import BatchNormalization, DepthwiseConv2D, Conv2D, MaxPooling2D, Conv2DTranspose, \
    Concatenate, Add, ReLU
from keras.applications.mobilenet_v2 import MobileNetV2
import keras.backend as keras_backend

from .vanilla_unet import conv_block


def unet_mobilenetv2(inputs, num_classes, shape, mobilenet_upsampling):
    """function that builds a UNet-like model whose encoder is mobilenetv2 (weights pretrained on imagenet and whose
    decoder can be (mobilenet_upsampling=True) a mirror image of the encoder or a classical UNet decoder.
    the function is taking the inputs as arguments"""
    encoder = MobileNetV2(input_shape=shape, include_top=False, weights='imagenet', input_tensor=inputs, pooling=None)
    encoder_64 = encoder.get_layer('expanded_conv_project_BN').output  # 16 channels
    num_channels_64 = 16
    encoder_32 = encoder.get_layer('block_2_add').output  # 24 channels
    num_channels_32 = 24
    encoder_16 = encoder.get_layer('block_5_add').output  # 32 channels
    num_channels_16 = 32
    encoder_8 = encoder.get_layer('block_12_add').output  # 96 channels
    num_channels_8 = 96
    output_encoder = encoder.get_layer('out_relu').output  # 1280 channels

    if mobilenet_upsampling:
        x_4_up = Conv2D(320, kernel_size=1, use_bias=False, name='Conv_1_in')(output_encoder)
        x_4_up = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='Conv_1_in_bn')(x_4_up)
        x_4_up = ReLU(6., name='Conv_1_in_relu')(x_4_up)

        x_4_up = _inverted_res_block_upsampling(x_4_up, filters=160, expansion=6, block_id=16)
        x_4_up = _inverted_res_block_upsampling(x_4_up, filters=160, expansion=6, block_id=15)
        x_4_up = _inverted_res_block_upsampling(x_4_up, filters=160, expansion=6, block_id=14)
        x_8_up = _inverted_res_block_upsampling(x_4_up, filters=num_channels_8, expansion=6, block_id=13,
                                                increase_fm_size=True)

        x_8_up = Concatenate()([encoder_8, x_8_up])
        x_8_up = _inverted_res_block_upsampling(x_8_up, filters=num_channels_8, expansion=6, block_id=12)
        x_8_up = _inverted_res_block_upsampling(x_8_up, filters=num_channels_8, expansion=6, block_id=11)
        x_8_up = _inverted_res_block_upsampling(x_8_up, filters=64, expansion=6, block_id=10)
        x_8_up = _inverted_res_block_upsampling(x_8_up, filters=64, expansion=6, block_id=9)
        x_8_up = _inverted_res_block_upsampling(x_8_up, filters=64, expansion=6, block_id=8)
        x_8_up = _inverted_res_block_upsampling(x_8_up, filters=64, expansion=6, block_id=7)
        x_16_up = _inverted_res_block_upsampling(x_8_up, filters=num_channels_16, expansion=6, block_id=6,
                                                 increase_fm_size=True)

        x_16_up = Concatenate()([encoder_16, x_16_up])
        x_16_up = _inverted_res_block_upsampling(x_16_up, filters=num_channels_16, expansion=6, block_id=5)
        x_16_up = _inverted_res_block_upsampling(x_16_up, filters=num_channels_16, expansion=6, block_id=4)
        x_32_up = _inverted_res_block_upsampling(x_16_up, filters=num_channels_32, expansion=6, block_id=3,
                                                 increase_fm_size=True)

        x_32_up = Concatenate()([encoder_32, x_32_up])
        x_32_up = _inverted_res_block_upsampling(x_32_up, filters=num_channels_32, expansion=6, block_id=2)
        x_64_up = _inverted_res_block_upsampling(x_32_up, filters=num_channels_64, expansion=6, block_id=1,
                                                 increase_fm_size=True)

        x_64_up = Concatenate()([encoder_64, x_64_up])
        x_128_up = _inverted_res_block_upsampling(x_64_up, filters=2*num_channels_64, expansion=6, block_id=0,
                                                  increase_fm_size=True)  # not exactly the mirror image + increase
        # number of output channels

    else:
        x_8_up = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='SAME')(output_encoder)

        x_8_up = Concatenate()([encoder_8, x_8_up])  # not the same number of channels
        x_8_up = conv_block(num_channels_8)(x_8_up)
        x_16_up = Conv2DTranspose(num_channels_8, kernel_size=3, strides=2, activation='relu', padding='SAME')(x_8_up)

        x_16_up = Concatenate()([encoder_16, x_16_up])
        x_16_up = conv_block(num_channels_16)(x_16_up)
        x_32_up = Conv2DTranspose(num_channels_16, kernel_size=3, strides=2, activation='relu', padding='SAME')(x_16_up)

        x_32_up = Concatenate()([encoder_32, x_32_up])
        x_32_up = conv_block(num_channels_32)(x_32_up)
        x_64_up = Conv2DTranspose(num_channels_32, kernel_size=3, strides=2, activation='relu', padding='SAME')(x_32_up)

        x_64_up = Concatenate()([encoder_64, x_64_up])
        x_64_up = conv_block(num_channels_64)(x_64_up)
        x_128_up = Conv2DTranspose(num_channels_64, kernel_size=3, strides=2, activation='relu',
                                   padding='SAME')(x_64_up)

        x_128_up = conv_block(num_channels_64)(x_128_up)  # uses num_channels_64

    output = Conv2D(num_classes, kernel_size=1, activation="sigmoid")(x_128_up)
    return output


def _inverted_res_block_upsampling(inputs, expansion, filters, block_id, increase_fm_size=False):
    """function producing an inverted residual block, simplified version of the one used in the repo mentionned on
    top of this file; at the moment a dephthwise separable transposed conv2D does not exists so in the case of
    increasing the feature map size, we perform the fm size increase in the expand layer"""
    in_channels = keras_backend.int_shape(inputs)[-1]
    x = inputs
    prefix = 'up_block_{}_'.format(block_id)

    # Expand
    if increase_fm_size:
        x = Conv2DTranspose(expansion * in_channels, kernel_size=3, strides=2, activation=None, padding='SAME')(x)
    else:
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, activation=None, use_bias=False, padding='same',
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == filters and not increase_fm_size:
        return Add(name=prefix + 'add')([inputs, x])
    return x
