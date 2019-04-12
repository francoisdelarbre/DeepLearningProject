import numpy as np
import cv2
import os
from pathlib import Path

PATH = 'data/stage1_train'  # path to the training set
KERNEL_SIZE = 3

if __name__ == "__main__":
    """creates several new files containing different masks to train the model:
        - union_mask: union of the masks corresponding to each cell
        - border_mask: mask that is equal to 1 when two different cells touch one another
        - weight_mask: mask that is equal to 1 on the border of each cell to weight these points more heavily"""

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)  # kernel fot the dilation
    img_list = os.listdir(PATH)  # list of training images
    num_samples = len(img_list)
    for i, sample in enumerate(img_list):  # iterate over training images
        print("{} samples treated out of {}".format(i, num_samples))
        path_to_masks = Path(PATH) / sample / 'masks'  # path to the masks of the image
        masks_list = os.listdir(str(path_to_masks))  # list of the masks of the image

        based_img = cv2.imread(str(Path(path_to_masks) / masks_list[0]), cv2.IMREAD_GRAYSCALE)
        union_mask = np.zeros(based_img.shape).astype("float")
        border_mask = np.zeros(based_img.shape).astype("float")
        weight_mask = np.zeros(based_img.shape).astype("float")

        for mask in masks_list:  # iterate over the masks of the image
            path_to_mask = str(Path(path_to_masks) / mask)  # path to the current mask
            img = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)  # load the mask
            img = img.astype("float") / 255

            union_mask = union_mask + img

            dilated_img = cv2.dilate(img, kernel, iterations=1)  # dilate the mask
            border_mask += dilated_img  # make the sum of the dilated mask

            boundary = img - cv2.erode(img, kernel, iterations=1)
            weight_mask +=  boundary

        border_mask[border_mask == 1] = 0
        border_mask[border_mask > 1] = 1

        dir_name = Path(PATH) / sample / 'processed_masks'

        if not os.path.exists(str(dir_name)):
            os.makedirs(str(dir_name))

        union_mask = np.uint8(union_mask) * 255
        border_mask = np.uint8(border_mask) * 255
        weight_mask = np.uint8(weight_mask) * 255
        cv2.imwrite(str(dir_name / 'union_mask.png'), union_mask)
        cv2.imwrite(str(dir_name / 'border_mask.png'), border_mask)
        cv2.imwrite(str(dir_name / 'weight_mask.png'), weight_mask)
