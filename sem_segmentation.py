from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from pathlib import Path
import random
import argparse
import json

from data_generator import DataGenerator
from loss import bce_dice_loss, i_o_u_metric
from TensorBoardPredictedImage import TensorBoardPredictedImage
from models.unet_mobilenet import unet_mobilenetv2
from models.vanilla_unet import unet_model


parser = argparse.ArgumentParser(description='Computing table')
parser.add_argument('--save_file', default='model', type=str, help='file name of the model')
parser.add_argument('--input_size', default=128, type=int, help='width/height of the input')
parser.add_argument('--nbr_channels', default=3, type=int, help='number of channels')
parser.add_argument('--data_dir', default='data/stage1_train', type=str, help='directory containing the data')
parser.add_argument('--batch_size', default=8, type=int, help='size of a batch')
parser.add_argument('--train_prop', default=.9, type=float, help='proportion of training set w.r.t. complete dataset')
parser.add_argument('--out_masks', default='["union_mask"]', type=str, help='output masks as a json string')
parser.add_argument('--tensorboard_folder', default='', type=str, help='name of the tensorboard folder, if empty'
                                                                       'string, will use current time as name')

args = parser.parse_args()


if __name__ == "__main__":
    out_masks = json.loads(args.out_masks)
    log_dir = Path("tf_logs") / (datetime.now().strftime("%Y.%m.%d-%H.%M") if args.tensorboard_folder == ''
                                 else args.tensorboard_folder)
    ids_list = [directory.name for directory in Path(args.data_dir).iterdir() if directory.is_dir()]

    random.Random(17).shuffle(ids_list)
    last_train_element = int(0.9 * len(ids_list))
    ids_list_train, ids_list_val = ids_list[:last_train_element], ids_list[last_train_element:]

    train_gen = DataGenerator(args.data_dir, output_masks=out_masks, batch_size=args.batch_size,
                              resolution=args.input_size, performs_data_augmentation=True, ids_list=ids_list_train)
    val_gen = DataGenerator(args.data_dir, output_masks=out_masks, batch_size=args.batch_size,
                            resolution=args.input_size, performs_data_augmentation=False, ids_list=ids_list_val)

    inputs = Input(shape=(args.input_size, args.input_size, args.nbr_channels))
    output = unet_mobilenetv2(inputs, len(out_masks), shape=(args.input_size, args.input_size, args.nbr_channels),
                              mobilenet_upsampling=True)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=0.00008), loss=bce_dice_loss, metrics=[i_o_u_metric])

    tensorboard_img, tensorboard_label = val_gen.get_unique_item(17)
    model.fit_generator(train_gen, epochs=720, verbose=2, validation_data=val_gen, callbacks=[
        TensorBoard(log_dir=log_dir),
        TensorBoardPredictedImage(img=tensorboard_img, label=tensorboard_label, model=model, log_dir=log_dir / 'img')])

    model.save(str(log_dir / args.save_file /'.h5'))

    print("end training")
