"""FC-DenseNet103 model, as described in http://arxiv.org/abs/1611.09326, when in doubt with the details of the paper,
we inspired ourselves from the official implementation https://github.com/SimJeg/FC-DenseNet. When in doubt between the
paper and the implementation, we followed the implementation"""
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    Concatenate, ReLU, Dropout
import keras.backend as keras_backend
from keras.regularizers import l2


class FCDenseNet:
    def __init__(self, num_channels=(4, 5, 7, 10, 12, 15), out_channels_first_conv=48,
                 growth_rate=16, regularizer=l2(1e-4)):
        self.num_channels = num_channels
        self.out_channels_first_conv = out_channels_first_conv
        self.growth_rate = growth_rate
        self.current_num_channels = 0
        self.reg = regularizer

    def compute_output(self, inputs, num_classes):
        """builds a FC-DenseNet103 model, as described in http://arxiv.org/abs/1611.09326
        the method is taking the inputs as arguments and returns the output"""

        x = Conv2D(self.out_channels_first_conv, kernel_size=3, padding='same', activation=None,
                   kernel_initializer="he_uniform", kernel_regularizer=self.reg)(inputs)
        self.current_num_channels = self.out_channels_first_conv

        # downsampling
        x_224 = self.dense_subnetwork(x, self.num_channels[0], concat_inputs_to_outputs=True)

        x_112 = self.transition_down(x_224)
        x_112 = self.dense_subnetwork(x_112, self.num_channels[1], True)

        x_56 = self.transition_down(x_112)
        x_56 = self.dense_subnetwork(x_56, self.num_channels[2], True)

        x_28 = self.transition_down(x_56)
        x_28 = self.dense_subnetwork(x_28, self.num_channels[3], True)

        x_14 = self.transition_down(x_28)
        x_14 = self.dense_subnetwork(x_14, self.num_channels[4], True)

        x_7 = self.transition_down(x_14)

        # middle
        x_7 = self.dense_subnetwork(x_7, self.num_channels[5], concat_inputs_to_outputs=False)

        # upsampling
        x_14_up = self.transition_up(x_7, self.num_channels[5] * self.growth_rate)
        x_14_up = self.dense_subnetwork(Concatenate()([x_14_up, x_14]), self.num_channels[4], False)

        x_28_up = self.transition_up(x_14_up, self.num_channels[4] * self.growth_rate)
        x_28_up = self.dense_subnetwork(Concatenate()([x_28_up, x_28]), self.num_channels[3], False)

        x_56_up = self.transition_up(x_28_up, self.num_channels[3] * self.growth_rate)
        x_56_up = self.dense_subnetwork(Concatenate()([x_56_up, x_56]), self.num_channels[2], False)

        x_112_up = self.transition_up(x_56_up, self.num_channels[2] * self.growth_rate)
        x_112_up = self.dense_subnetwork(Concatenate()([x_112_up, x_112]), self.num_channels[1], False)

        x_224_up = self.transition_up(x_112_up, self.num_channels[1] * self.growth_rate)
        x_224_up = Concatenate()([x_224_up,
                                  self.dense_subnetwork(Concatenate()([x_224_up, x_224]), self.num_channels[0],
                                                        False)])  # see original implementation

        # classification layer
        return Conv2D(num_classes, kernel_size=1, padding='same', activation='sigmoid', kernel_initializer="he_uniform",
                      kernel_regularizer=self.reg)(x_224_up)

    def dense_subnetwork(self, inputs, num_blocks, concat_inputs_to_outputs):
        """applies a dense_subnetwork and returns its output
        :param inputs: inputs of the dense subnetwork
        :param num_blocks: number of dense blocks in the dense subnetwork
        :param concat_inputs_to_outputs: True if the inputs are to be concatenated with the outputs, in that case,
        we increase the current_num_channels
        :return: the output of the subnetwork"""
        outputs = [inputs]
        for i in range(num_blocks):
            if i == 0:
                x = inputs
            else:
                x = Concatenate()(outputs)
            x = BatchNormalization(axis=-1, beta_regularizer=self.reg, gamma_regularizer=self.reg)(x)
            x = ReLU()(x)
            x = Conv2D(self.growth_rate, kernel_size=3, use_bias=False, padding='same', kernel_initializer="he_uniform",
                       kernel_regularizer=self.reg, activation=None)(x)
            x = Dropout(rate=0.2)(x)
            outputs.append(x)

        if concat_inputs_to_outputs:
            self.current_num_channels += num_blocks * self.growth_rate
            return Concatenate()(outputs)
        else:
            return Concatenate()(outputs[1:])

    def transition_down(self, inputs):
        """applies a transition_down to the inputs"""
        x = BatchNormalization(axis=-1, beta_regularizer=self.reg, gamma_regularizer=self.reg)(inputs)
        x = ReLU()(x)
        x = Conv2D(self.current_num_channels, kernel_size=1, use_bias=False, padding='same',
                   kernel_initializer="he_uniform", kernel_regularizer=self.reg, activation=None)(x)
        x = Dropout(rate=0.2)(x)
        return MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    def transition_up(self, inputs, num_channels):
        """applies the transition_up having num_channels out_channels to the inputs"""
        return Conv2DTranspose(num_channels, kernel_size=3, strides=2, padding='same', activation=None,
                               kernel_initializer="he_uniform", kernel_regularizer=self.reg)(inputs)
