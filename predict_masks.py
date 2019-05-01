"""predict masks and save them as files in the corresponding folders; this script predict masks from the validation
set, to predict masks from the test sets, see make_submission (that directly outputs .csv files for submission)"""

import argparse
import cv2
import random
from utils import split_train_val
from pathlib import Path
import numpy as np
import os

from keras.models import load_model
from loss import bce_dice_loss, i_o_u_metric, bce_dice_loss_unet, i_o_u_metric_unet, i_o_u_metric_first_mask, \
    i_o_u_metric_second_mask
from data_generator import DataGenerator
from utils import predict_image

parser = argparse.ArgumentParser(description='showing net predictions on val data')
parser.add_argument('--model_name', default='model', type=str, help='name of the model file without extension, '
                                                                    'assumed to lay in h5_files folder')
parser.add_argument('--pred_mask_name', type=str, help='name of the mask we are predicting (without extension)')
parser.add_argument('--input_size', default=128, type=int, help='width/height of the input')
parser.add_argument('--nbr_channels', default=3, type=int, help='number of channels')
parser.add_argument('--data_dir', default='data/stage1_train', type=str, help='directory containing the data')
parser.add_argument('--train_prop', default=.9, type=float, help='proportion of training set w.r.t. complete dataset')
parser.add_argument('--val_first', action='store_true', help='takes the vlaidation set at the firsts offsets of the '
                                                             'shuffled array rather than the lasts')

args = parser.parse_args()

if __name__ == "__main__":
    custom_objects = {"bce_dice_loss": bce_dice_loss, "bce_dice_loss_unet": bce_dice_loss_unet,
                      "i_o_u_metric": i_o_u_metric, "i_o_u_metric_unet": i_o_u_metric_unet,
                      "i_o_u_metric_first_mask": i_o_u_metric_first_mask,
                      "i_o_u_metric_second_mask": i_o_u_metric_second_mask}  # so that
    # keras do not crash :/ https://github.com/keras-team/keras/issues/5916
    model = load_model(str(Path('h5_files') / f"{args.model_name}.h5"), custom_objects=custom_objects)

    _, ids_list_val = split_train_val(args.data_dir, args.train_prop, not args.val_first)

    path = Path(args.data_dir)

    for id_img in ids_list_val:
        dir_name = path / id_img / 'predicted_masks'
        if not os.path.exists(str(dir_name)):
            os.makedirs(str(dir_name))

        base_img = cv2.imread(str(path / id_img / 'images' / f"{id_img}.png"))
        base_img = base_img / 255.
        pred_img = predict_image(model, base_img, last_layer_res=args.input_size)
        pred_img = np.uint8(pred_img * 255)
        print(id_img)
        cv2.imwrite(str(dir_name / f"{args.pred_mask_name}.png"), pred_img)
