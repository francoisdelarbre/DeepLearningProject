from datetime import datetime
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Lambda, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from pathlib import Path

from data_generator import DataGenerator

INPUT_SIZE = 128
NBR_DIMS = 3
NUM_CLASSES = 1
DATA_DIR = 'data/stage1_train'
BATCH_SIZE = 32

import numpy as np
def generate(x=np.zeros((32, 128, 128, 3)), y=np.ones((32, 128, 128, 1)), batch_size=32):
    """
    generator for the dataset

    :param x: training set
    :param y: testing set
    :param batch_size: size of a batch (without taking data augmentation into account
    :return: a batch of data taken from x_train and y_train
    """
    while True:
        for i in range(len(y) // batch_size):
            start, end = batch_size*i, batch_size*(i+1)
            # yield x[start:end, ...], to_categorical(y[start:end, ...], num_classes=NUM_CLASSES)
            yield x[start:end, ...], y[start:end, ...]

if __name__ == "__main__":
    log_dir = Path("tf_logs") / datetime.now().strftime("%Y.%m.%d-%H.%M")
    generator = DataGenerator(DATA_DIR, batch_size=BATCH_SIZE, resolution=(128, 128), n_channels_input=3,
                              output_masks=('union_mask',))

    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, NBR_DIMS))
    x = Conv2D(10, kernel_size=(3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x2 = MaxPooling2D(pool_size=(2, 2))(x)
    x2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = UpSampling2D(size=(2, 2))(x2)

    x = Concatenate()([x2, x])
    x = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(10, kernel_size=(3, 3), activation='relu')(x)
    output = Conv2D(NUM_CLASSES, (1, 1), activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(generate(), steps_per_epoch=len(generator), epochs=40,  # TODO replace with generator
                        verbose=2, callbacks=[TensorBoard(log_dir=log_dir)])
