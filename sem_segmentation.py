from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from pathlib import Path

from data_generator import DataGenerator
from loss import bce_dice_loss, i_o_u_metric
from TensorBoardPredictedImage import TensorBoardPredictedImage
from models.unet_mobilenet import unet_mobilenetv2
from models.vanilla_unet import unet_model

INPUT_SIZE = 128
NBR_DIMS = 3
NUM_CLASSES = 1
DATA_DIR = 'data/stage1_train'
BATCH_SIZE = 8


if __name__ == "__main__":
    log_dir = Path("tf_logs") / datetime.now().strftime("%Y.%m.%d-%H.%M")
    generator = DataGenerator(DATA_DIR, output_masks=('union_mask',), batch_size=BATCH_SIZE, resolution=128,
                              n_channels_input=3)

    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, NBR_DIMS))
    output = unet_mobilenetv2(inputs, NUM_CLASSES, shape=(INPUT_SIZE, INPUT_SIZE, NBR_DIMS), mobilenet_upsampling=True)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=0.00008), loss=bce_dice_loss, metrics=[i_o_u_metric])

    tensorboard_img, tensorboard_label = generator.get_unique_item(17)
    model.fit_generator(generator, epochs=720, verbose=2, callbacks=[
        TensorBoard(log_dir=log_dir),
        TensorBoardPredictedImage(img=tensorboard_img, label=tensorboard_label, model=model, log_dir=log_dir / 'img')])

    model.save(str(log_dir / 'model.h5'))

    print("end training")