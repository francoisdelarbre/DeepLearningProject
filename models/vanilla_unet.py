from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate


def unet_model(inputs, num_classes, num_channels_128):
    """function that builds a UNet-like model taking the inputs as arguments"""
    num_channels_64 = num_channels_128 * 2
    num_channels_32 = num_channels_128 * 4

    x_128 = conv_block(num_channels_128, starts_with_batchnorm=False, ends_with_batchnorm=False)(inputs)

    x_64 = MaxPooling2D(pool_size=2, padding='SAME')(x_128)
    x_64 = conv_block(num_channels_64, ends_with_batchnorm=False)(x_64)

    x_32 = MaxPooling2D(pool_size=2, padding='SAME')(x_64)
    x_32 = conv_block(num_channels_32)(x_32)
    x_64_up = Conv2DTranspose(num_channels_64, kernel_size=3, strides=2, activation='relu', padding='SAME')(x_32)

    x_64_up = Concatenate()([x_64, x_64_up])
    x_64_up = conv_block(num_channels_64)(x_64_up)
    x_128_up = Conv2DTranspose(num_channels_128, kernel_size=3, strides=2, activation='relu', padding='SAME')(x_64_up)

    x_128_up = Concatenate()([x_128, x_128_up])
    x_128_up = conv_block(num_channels_128)(x_128_up)
    output = Conv2D(num_classes, kernel_size=1, activation="sigmoid")(x_128_up)

    return output


def conv_block(num_channels, starts_with_batchnorm=True, ends_with_batchnorm=True):
    """function that build a conv_block composed of 1+starts_with_batchnorm+ends_with_batchnorm batchnorms layers
    and 2 conv layers
    :param num_channels: the number of channels outputted by the convolutions
    :param starts_with_batchnorm: whether we start with a batch normalization layer or not
    :param ends_with_batchnorm: whether we end with a batch normalization layer or not
    :returns: a function representing the block"""
    def block_fct(x):
        if starts_with_batchnorm:
            x = BatchNormalization()(x)
        x = Conv2D(num_channels, kernel_size=3, padding='SAME', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(num_channels, kernel_size=3, padding='SAME', activation='relu')(x)
        if ends_with_batchnorm:
            x = BatchNormalization()(x)
        return x
    return block_fct
