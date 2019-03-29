import numpy as np
import cv2
import os
from pathlib import Path

PATH = 'data/stage1_train'  # path to the training set
KERNEL_SIZE = 3

if __name__ == "__main__":

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)  # kernel fot the dilation
    img_list = os.listdir(PATH)  # list of training images
    num_samples = len(img_list)
    for i, sample in enumerate(img_list):  # iterate over training images
        print(f"{i} samples treated out of {num_samples}")
        path_to_masks = Path(PATH) / sample / 'masks'  # path to the masks of the image
        masks_list = os.listdir(path_to_masks)  # list of the masks of the image

        based_img = cv2.imread(str(Path(path_to_masks) / masks_list[0]), cv2.IMREAD_GRAYSCALE)
        img_sum = np.zeros(based_img.shape).astype("float")
        dilated_sum = np.zeros(based_img.shape).astype("float")

        for mask in masks_list:  # iterate over the masks of the image
            path_to_mask = str(Path(path_to_masks) / mask)  # path to the current mask
            img = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)  # load the mask
            img = img.astype("float") / 255

            img_sum = img_sum + img
            dilated_img = cv2.dilate(img, kernel, iterations=1)  # dilate the mask

            dilated_sum = dilated_sum + dilated_img  # make the sum of the dilated mask

        dilated_sum[dilated_sum == 1] = 0
        dilated_sum[dilated_sum > 1] = 1

        dir_name = Path(PATH) / sample / 'processed_masks'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        img_sum = np.uint8(img_sum) * 255
        dilated_sum = np.uint8(dilated_sum) * 255
        cv2.imwrite(str(dir_name / 'union_mask.png'), img_sum)
        cv2.imwrite(str(dir_name / 'border_mask.png'), dilated_sum)
