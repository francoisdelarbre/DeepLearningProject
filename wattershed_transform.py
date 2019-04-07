import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PATH = 'data/stage1_train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/processed_masks/'


if __name__ == "__main__":
    mask = cv2.imread(PATH+'union_mask.png')
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    borders = cv2.imread(PATH+'border_mask.png')
    tmps_color = cv2.subtract(mask, borders)
    borders = cv2.cvtColor(borders, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('borders', borders)
    # cv2.waitKey(0)

    tmps = cv2.subtract(gray_mask, borders)
    cv2.imshow('borders', tmps)
    cv2.waitKey(0)

    dist_transform = cv2.distanceTransform(tmps, cv2.DIST_L2, 5)

    # cv2.imshow('dist_transform', dist_transform)
    # cv2.waitKey(0)
    # plt.imshow(dist_transform)
    # plt.show()
    tmps = np.uint8(tmps)
    ret, markers = cv2.connectedComponents(tmps)
    # plt.imshow(markers)
    # plt.show()

    markers += 1

    sure_fg = cv2.subtract(dist_transform, dist_transform)

    for i in range(2, ret+1):
        dist_copy = np.copy(dist_transform)

        # dist_copy[markers == i] = 1
        dist_copy[markers != i] = 0
        maxValue = np.max(dist_copy)
        dist_copy[dist_copy <= 0.7*maxValue] = 0
        dist_copy[dist_copy > 0.7 * maxValue] = 255
        sure_fg += dist_copy

    # cv2.imshow('markers', calque)
    # cv2.waitKey(0)

    sure_fg = np.uint8(sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    unknown = cv2.subtract(gray_mask, sure_fg)

    # plt.imshow(tmps)
    # plt.show()

    markers[unknown == 255] = 0

    # plt.imshow(markers)
    # plt.show()
    # markers = np.uint8(markers)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    segmented = np.copy(mask)
    truc = cv2.watershed(segmented, markers)
    segmented[truc == -1] = 0

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.imshow('segmented', segmented)
    cv2.waitKey(0)

    # plt.imshow(markers)
    # plt.show()


