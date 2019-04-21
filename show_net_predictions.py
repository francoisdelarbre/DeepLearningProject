"""shows the predictions of the network on training data to compare it with the masks"""

import argparse
import cv2
import json
import random
from utils import split_train_val
from pathlib import Path
import numpy as np

from tensorflow.keras.models import load_model
from loss import bce_dice_loss, i_o_u_metric, bce_dice_loss_unet, i_o_u_metric_unet
from data_generator import DataGenerator


parser = argparse.ArgumentParser(description='showing net predictions on val data')
parser.add_argument('--model_name', default='model', type=str, help= 'name of the model file without extension, '
                                                                     'assumed to lay in h5_files folder')
parser.add_argument('--input_size', default=128, type=int, help='width/height of the input')
parser.add_argument('--nbr_channels', default=3, type=int, help='number of channels')
parser.add_argument('--data_dir', default='data/stage1_train', type=str, help='directory containing the data')
parser.add_argument('--train_prop', default=.9, type=float, help='proportion of training set w.r.t. complete dataset')
parser.add_argument('--out_masks', default='["union_mask", "weight_mask"]', type=str,
                    help='output masks as a json string, weight mask should be the last')

args = parser.parse_args()

if __name__ == '__main__':
    custom_objects = {"bce_dice_loss": bce_dice_loss, "bce_dice_loss_unet": bce_dice_loss_unet,
                      "i_o_u_metric": i_o_u_metric, "i_o_u_metric_unet": i_o_u_metric_unet}  # so that
    # keras do not crash :/ https://github.com/keras-team/keras/issues/5916

    model = load_model(str(Path('h5_files') / f"{args.model_name}.h5"), custom_objects=custom_objects)
    out_masks = json.loads(args.out_masks)

    _, ids_list_val = split_train_val(args.data_dir, args.train_prop)

    val_gen = DataGenerator(args.data_dir, output_masks=out_masks, batch_size=1,
                            resolution=args.input_size, performs_data_augmentation=False, ids_list=ids_list_val)

    print("several images will be predicted, press 'ctrl-c' to quit or 'n' to show next image")
    print("shows into the order: top: [in_img, pred_mask_1, pred_mask_2, ...], "
          "bottom: [in_img, true_mask_1, true_mask_2]")

    for img, mask in val_gen:

        crop_offset = random.randint(0, img.shape[0] - 1)  # take one of the crops at random
        img = img[crop_offset:crop_offset + 1, :, :, :]
        mask = mask[crop_offset, :, :, :]
        pred_mask = model.predict(img)

        line_1 = [img[0, :, :, :]]
        line_2 = [img[0, :, :, :]]
        for i in range(mask.shape[2]):
            # turn grayscale to rgb
            line_1.append(np.stack([(pred_mask[0, :, :, i] > 0.5).astype(np.float16)]*3, axis=-1))
            line_2.append(np.stack([mask[:, :, i]] * 3, axis=-1))

        line_1 = np.hstack(line_1)
        line_2 = np.hstack(line_2)
        img_to_plot = (np.vstack([line_1, line_2]) * 255).astype(np.uint8)

        rescaling_factor = 512 // max(img_to_plot.shape[0], img_to_plot.shape[1])
        img_to_plot = cv2.resize(img_to_plot, None, fx=rescaling_factor, fy=rescaling_factor)

        cv2.imshow('img and predicted masks', img_to_plot)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print("end of validation set reached")
