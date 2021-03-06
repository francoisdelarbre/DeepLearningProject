from watershed_transform import watershed_transform, watershed_transform_border, watershed_transform_center
from utils import split_train_val
import os
import pickle
from matplotlib import pyplot as plt
import cv2
import numpy as np

# list of directories
PATH = 'data/stage1_train/'

PRED_BORDER = '/predicted_masks/border.png'
PRED_CENTER = '/predicted_masks/center.png'
PRED_UNION_NO_WEIGHTS = '/predicted_masks/union_no_weights.png'
PRED_UNION_WEIGHTS = '/predicted_masks/union_weights.png'

TRUTH = '/masks/'


def separate_mask(union_mask, ret):
    """separate a mask with all nulei into a mask/nuclei
        :param union_mask: a mask where every nuclei have a different label
        :param ret: the number of different labels
        :return mask_list: a list of the nuclei masks"""
    mask_list = []
    for i in range(1, ret):
        mask = np.zeros(union_mask.shape, dtype=np.uint8)
        mask[union_mask == i] = 255
        mask_list.append(mask)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

    return mask_list


def iou(mask1, mask2):
    """compute the IOU of two masks
            :param mask1: a uint8 binary mask
            :param mask2: a uint8 binary mask
            :return IOU of mask1 and 2"""
    union = np.bitwise_or(mask1, mask2)
    area_union = np.sum(union) / 255

    intersection = np.bitwise_and(mask1, mask2)
    area_intersection = np.sum(intersection) / 255

    return area_intersection / area_union


def main():
    # load ids of validations images
    val_set = split_train_val(PATH, 0.9, False)

    mean_score = 0
    tresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # iterate over images of the validation set
    for k in range(len(val_set)):
        print('img', k, '/', len(val_set))

        # load predicted masks
        union_mask = cv2.imread(PATH + val_set[k] + PRED_UNION_NO_WEIGHTS, cv2.IMREAD_GRAYSCALE)
        union_w_mask = cv2.imread(PATH + val_set[k] + PRED_UNION_WEIGHTS, cv2.IMREAD_GRAYSCALE)
        border_mask = cv2.imread(PATH + val_set[k] + PRED_BORDER, cv2.IMREAD_GRAYSCALE)
        center_mask = cv2.imread(PATH + val_set[k] + PRED_CENTER, cv2.IMREAD_GRAYSCALE)

        # perform watershed
        # result = watershed_transform(union_mask)
        # result = watershed_transform(union_w_mask)
        # result = watershed_transform_border(union_mask, border_mask)
        # result = watershed_transform_center(union_mask, center_mask)
        # result = watershed_transform_border(union_w_mask, border_mask)
        result = watershed_transform_center(union_w_mask, center_mask)

        # assign a label/nuclei
        ret, result = cv2.connectedComponents(result, connectivity=4)

        # plt.imshow(result)
        # plt.show()

        # create a mask/nuclei
        predictions = separate_mask(result, ret)

        # load ground truth masks
        ground_truth_list = os.listdir(PATH + val_set[k] + TRUTH)
        ground_truths = [cv2.imread(PATH + val_set[k] + TRUTH + ground_truth, cv2.IMREAD_GRAYSCALE) for ground_truth in
                         ground_truth_list]

        # IOU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truths)))

        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                # fill the IOU matrix
                iou_matrix[i][j] = iou(predictions[i], ground_truths[j])

        score_img = 0
        for tresh in tresholds:
            hit_matrix = np.array((iou_matrix > tresh))

            # nb of prediction that matches a ground truth
            true_positives = np.sum(hit_matrix)

            false_positives = 0
            # number of prediction that have no match with a ground truth
            for i in range(len(predictions)):
                if np.sum(hit_matrix[i, :]) == 0:
                    false_positives += 1

            false_negatives = 0
            # number of ground truth that have no match with a prediction
            for j in range(len(ground_truths)):
                if np.sum(hit_matrix[:, j]) == 0:
                    false_negatives += 1

            score_tresh = true_positives / (true_positives + false_positives + false_negatives)
            score_img += 1 / len(tresholds) * score_tresh

        print(score_img)
        mean_score += score_img / len(val_set)

    print("mean score", mean_score)


if __name__ == '__main__':
    main()
