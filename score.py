import cv2
import numpy as np
import os


def IOU(mask1, mask2):
    union = np.bitwise_or(mask1, mask2)
    area_union = np.sum(union)/255  # remove 255 if mask are binary

    intersection = np.bitwise_and(mask1, mask2)
    area_intersection = np.sum(intersection)/255  # remove 255 if mask are binary

    return area_intersection/area_union


PATH1 = 'data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks/'
PATH2 = 'data/stage1_train/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe/masks/'
# list of the ground truths
ground_truth_list = os.listdir(PATH1)
ground_truths = [cv2.imread(PATH1+ground_truth, cv2.IMREAD_GRAYSCALE) for ground_truth in ground_truth_list]


# list of the predictions
prediction_list = os.listdir(PATH2)
predictions = [cv2.imread(PATH2+prediction, cv2.IMREAD_GRAYSCALE) for prediction in prediction_list]

# IOU matrix
IOU_matrix = np.zeros((len(predictions), len(ground_truths)))

for i in range(IOU_matrix.shape[0]):
    for j in range(IOU_matrix.shape[1]):
        # fill the IOU matrix
        IOU_matrix[i][j] = IOU(predictions[i], ground_truths[j])

# remove 0 from tresholds
tresholds = [0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
score_img = 0
for tresh in tresholds:
    hit_matrix = np.array((IOU_matrix > tresh))

    # nb of prediction that matches a ground truth
    TP = np.sum(hit_matrix)

    FP = 0
    # number of prediction that have no match with a ground truth
    for i in range(len(predictions)):
        if np.sum(hit_matrix[i, :]) == 0:
            FP += 1

    FN = 0
    # number of ground truth that have no match with a prediction
    for j in range(len(ground_truths)):
        if np.sum(hit_matrix[:, j]) == 0:
            FN += 1

    print("TP", TP, "FP", FP, "FN", FN)
    score_tresh = TP / (TP + FP + FN)
    print("score tresh", score_tresh)
    score_img += 1/len(tresholds) * score_tresh

print(score_img)
