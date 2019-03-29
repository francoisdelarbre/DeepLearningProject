import numpy as np
import cv2
import os

if __name__ == "__main__":

    kernel = np.ones((3, 3), np.uint8)  # kernel fot the dilation
    path = 'data/stage1_train'  # path to the training set
    img_list = os.listdir(path)  # list of training imgages
    sample = img_list[0]
    for sample in img_list:  # iterate over training images
        print(sample)
        path_to_masks = path + '/' + sample + '/masks'  # path to the masks of the image
        masks_list = os.listdir(path_to_masks)  # list of the masks of the image

        based_img = cv2.imread(path_to_masks + '/' + masks_list[0], cv2.IMREAD_GRAYSCALE)
        img_sum = np.zeros(based_img.shape).astype("float")
        dilated_sum = np.zeros(based_img.shape).astype("float")

        for mask in masks_list:  # iterate over the masks of the image
            path_to_mask = path_to_masks + '/' + mask  # path to the current mask
            img = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)  # load the mask
            img = img.astype("float") / 255

            img_sum = img_sum + img
            # cv2.imshow('image', img_sum)
            # cv2.waitKey(0)
            dilated_img = cv2.dilate(img, kernel, iterations=1)  # dilate the mask

            # cv2.imshow('dilated_img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            dilated_sum = dilated_sum + dilated_img  # make the sum of the dilated mask
            # cv2.imshow('sum_dilated', dilated_sum)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        dilated_sum[dilated_sum == 1] = 0
        dilated_sum[dilated_sum > 1] = 1
        # cv2.imshow('img', dilated_sum)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        dir_name = path + '/' + sample + '/processed_masks/'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        img_sum = np.uint8(img_sum) * 255
        dilated_sum = np.uint8(dilated_sum) * 255
        cv2.imwrite(dir_name + 'union_mask.png', img_sum)
        cv2.imwrite(dir_name + 'border_mask.png', dilated_sum)
