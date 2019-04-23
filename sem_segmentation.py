from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from pathlib import Path
import argparse
import json

from data_generator import DataGenerator
from utils import split_train_val
from loss import bce_dice_loss, i_o_u_metric, bce_dice_loss_unet, i_o_u_metric_unet
from TensorBoardPredictedImages import TensorBoardPredictedImages
from models.unet_mobilenet import unet_mobilenetv2
from models.FCDenseNet import FCDenseNet
from models.vanilla_unet import unet_model


parser = argparse.ArgumentParser(description='Computing table')
parser.add_argument('--save_file', default='model', type=str, help='file name of the model, also used as tensorboard '
                                                                   'dir if different from model')
parser.add_argument('--model', default='fc-densenet', type=str, help='neural network to use',
                    choices=['fc-densenet', 'unet_mobilenet', 'vailla_unet'])
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'rmsprop'], help='optimizer to use')
parser.add_argument('--num_epochs', default=150, type=int, help='number of epochs')
parser.add_argument('--nbr_channels', default=3, type=int, help='number of channels')
parser.add_argument('--main_data_dir', default='data/stage1_train', type=str, help='directory containing the principal '
                                                                                   'dataset')
parser.add_argument('--sec_data_dir', default='data/extra_data', type=str,
                    help='directory containing another dataset that looks like the main one to improve results, set to '
                         '"" if only the main dataset is to be used')
parser.add_argument('--batch_size', default=8, type=int, help='size of a batch')
parser.add_argument('--train_prop', default=.9, type=float, help='proportion of training set w.r.t. complete dataset')
parser.add_argument('--out_masks', default='["union_mask", "weight_mask"]', type=str,
                    help='output masks as a json string, weight mask should be the last if it is present')

# parameters that are unrelevant for some choices of the other parameters
parser.add_argument('--input_size', default=128, type=int,
                    help='width/height of the input, used for networks with variable input size '
                         '(unet_mobilenet/vanilla_unet)')
parser.add_argument('--sec_data_dir_factor', default=4, type=int,
                    help='set this value to use more the secondary data set, this makes sense if, for example, the '
                         'images are bigger in the secondary dataset (you can ignore this if you do not use a secondary'
                         'dataset ')
parser.add_argument('--non_border_cells_weights', default=30, type=int,
                    help="if 'weight_mask' in output_masks: the weight to give to non_border_cells pixels (border cells"
                         " pixels have a weight of 255)")
parser.add_argument('--fcdensenet_num_channels', default=0, type=int,
                    help="number of blocks to use by subnetwork, if 0 is given, will use the values for a "
                         "fc-densenet103 instead, to be ignored if model isn't densenet")
parser.add_argument('--fcdensenet_growth_rate', default=16, type=int,
                    help="growth factor to use, to be ignored if model isn't fc-densenet")

args = parser.parse_args()


if __name__ == "__main__":
    out_masks = json.loads(args.out_masks)
    log_dir = Path("tf_logs") / (datetime.now().strftime("%Y.%m.%d-%H.%M") if args.save_file == 'model'
                                 else args.save_file)

    ids_list_train, ids_list_val = split_train_val(args.main_data_dir, args.train_prop)

    if args.sec_data_dir == '':
        data_dirs = (args.main_data_dir,)
        ids_list_train = (ids_list_train,)
    else:
        data_dirs = (args.main_data_dir, args.sec_data_dir)
        ids_list_train = (ids_list_train, None)  # uses all the extra set as training data

    if args.model == 'fc-densenet':
        input_size = 224
    else:
        input_size = args.input_size

    train_gen = DataGenerator(data_dirs, output_masks=out_masks, batch_size=args.batch_size,
                              resolution=input_size, performs_data_augmentation=True, ids_list=ids_list_train,
                              sec_data_dir_factor=args.sec_data_dir_factor)

    val_batch_size = args.batch_size // 5 + 1  # // 5 + 1 because we will perform validation on 5 crops (4 corners +
    # center)
    val_gen = DataGenerator((args.main_data_dir,), output_masks=out_masks, batch_size=val_batch_size,
                            resolution=input_size, performs_data_augmentation=False, ids_list=(ids_list_val,))
    tensorboard_imgs, tensorboard_labels = val_gen.get_some_items([-17])

    inputs = Input(shape=(input_size, input_size, args.nbr_channels))
    num_classes = len(out_masks) - (1 if 'weight_mask' in out_masks else 0)

    if args.model == 'fc-densenet':
        fcdensenet = FCDenseNet(growth_rate=args.fcdensenet_growth_rate) if args.fcdensenet_num_channels == 0 else \
            FCDenseNet(num_channels=[args.fcdensenet_num_channels]*6, growth_rate=args.fcdensenet_growth_rate)
        output = fcdensenet.compute_output(inputs, num_classes)
    elif args.model == 'unet_mobilenet':
        output = unet_mobilenetv2(inputs, num_classes, shape=(input_size, input_size, args.nbr_channels),
                                  mobilenet_upsampling=True)
    elif args.model == 'vailla_unet':
        output = unet_model(inputs, num_classes, 64)
    else:
        raise ValueError("invalid model")

    model = Model(inputs=inputs, outputs=output)

    if args.optimizer == 'adam':
        optimizer = Adam(lr=0.00008)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(lr=1e-4, decay=0.998)
    else:
        raise ValueError("invalid optimizer")

    if "weight_mask" in out_masks:
        model.compile(optimizer=optimizer, loss=bce_dice_loss_unet, metrics=[i_o_u_metric_unet])
        tensorboard_labels = tensorboard_labels[:, :, :, :-1]
    else:
        model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[i_o_u_metric])

    model.fit_generator(train_gen, epochs=args.num_epochs, verbose=2, validation_data=val_gen, callbacks=[
        TensorBoard(log_dir=log_dir, write_graph=False),
        TensorBoardPredictedImages(imgs=tensorboard_imgs, labels=tensorboard_labels,
                                   model=model, log_dir=log_dir / 'img')])

    model.save(str(Path('h5_files') / (args.save_file + '.h5')))

    print("end training")
