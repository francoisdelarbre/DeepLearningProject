from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as keras_backend
from pathlib import Path

from data_generator import DataGenerator
from loss import bce_dice_loss
from TensorBoardPredictedImage import TensorBoardPredictedImage
from models import unet_model

INPUT_SIZE = 128
NBR_DIMS = 3
NUM_CLASSES = 1
DATA_DIR = 'data/stage1_train'
BATCH_SIZE = 8


if __name__ == "__main__":
    log_dir = Path("tf_logs") / datetime.now().strftime("%Y.%m.%d-%H.%M")
    generator = DataGenerator(DATA_DIR, output_masks=('union_mask',), batch_size=BATCH_SIZE, resolution=(128, 128),
                              n_channels_input=3)

    num_channels_128 = 32

    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, NBR_DIMS))
    output = unet_model(inputs, NUM_CLASSES, num_channels_128)
    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(lr=0.00001), loss=bce_dice_loss, metrics=['accuracy'])
    epoch_reduce_list = [20, 250, 400]
    red_fact = 0.5

    tensorboard_img, tensorboard_label = generator.get_unique_item(17)

    model.fit_generator(generator, epochs=450, verbose=2, callbacks=[
        TensorBoard(log_dir=log_dir),
        TensorBoardPredictedImage(img=tensorboard_img, label=tensorboard_label, model=model, log_dir=log_dir / 'img'),])
        #LearningRateScheduler(lambda x, y: y if x not in epoch_reduce_list else red_fact*y)])

    print("end training")
