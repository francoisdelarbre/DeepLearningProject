import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PATH = 'data/stage1_train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/processed_masks/'


def wattershed_transform(union_mask):
    """Wattershed on union mask. use distance map to find markers
        :param union_mask, a uint8 mask"""
    plt.figure()
    plt.imshow(union_mask)

    # compute distance map of mask
    tmps = union_mask.copy()
    dist_transform = cv2.distanceTransform(tmps, cv2.DIST_L2, 5)


    # find big blop in the images, we try to separate each blop with wattershed
    tmps = np.uint8(tmps)
    ret, blops = cv2.connectedComponents(tmps)
    blops += 1  # background is 1, 0 will be for the zone to flood by wattershed

    # matrix of the sure foreground, it will serve as marker
    sure_fg = cv2.subtract(dist_transform, dist_transform)

    # iterate over the blops
    for i in range(2, ret+1):
        dist_copy = np.copy(dist_transform)

        # mask with only the current blop
        dist_copy[blops != i] = 0

        # max distance within blob
        maxValue = np.max(dist_copy)

        # keep as marker the pixel with sufficient distance
        dist_copy[dist_copy <= 0.7*maxValue] = 0
        dist_copy[dist_copy > 0.7 * maxValue] = 255

        # add marker to markers map
        sure_fg += dist_copy

    # put different labels to the markers
    sure_fg = np.uint8(sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1

    # set zone to flood to 0
    not_sure = cv2.subtract(union_mask, sure_fg)
    markers[not_sure == 255] = 0

    # find broders thanks to wattershed
    segmented = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    result = cv2.watershed(segmented, markers)

    # remove borders from cells
    union_mask[result == -1] = 0
    cv2.imshow('resutl', union_mask)
    cv2.waitKey(0)
    return union_mask


def wattershed_transform_border(union_mask, border_mask):
    """Wattershed on union mask. use distance map to find markers
        :param union_mask, a uint8 mask
        :param border_mask, a uint8 mask"""

    # compute distance map of mask
    tmps = union_mask.copy()
    tmps = cv2.subtract(tmps, border_mask)
    dist_transform = cv2.distanceTransform(tmps, cv2.DIST_L2, 5)

    cv2.imshow('tmps', tmps)
    cv2.waitKey(0)

    # find big blop in the images, we try to separate each blop with wattershed
    tmps = np.uint8(tmps)
    ret, blops = cv2.connectedComponents(tmps)
    blops += 1 # background is 1, 0 will be for the zone to flood by wattershed

    # matrix of the sure foreground, it will serve as marker
    sure_fg = cv2.subtract(dist_transform, dist_transform)

    # iterate over the blops
    for i in range(2, ret+1):
        dist_copy = np.copy(dist_transform)

        # mask with only the current blop
        dist_copy[blops != i] = 0

        # max distance within blob
        maxValue = np.max(dist_copy)

        # keep as marker the pixel with sufficient distance
        dist_copy[dist_copy <= 0.7*maxValue] = 0
        dist_copy[dist_copy > 0.7 * maxValue] = 255

        # add marker to markers map
        sure_fg += dist_copy


    # put different labels to the markers
    sure_fg = np.uint8(sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1

    # set zone to flood to 0
    not_sure = cv2.subtract(union_mask, sure_fg)
    markers[not_sure == 255] = 0

    # find broders thanks to wattershed
    segmented = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    result = cv2.watershed(segmented, markers)

    # remove borders from cells
    union_mask[result == -1] = 0

    cv2.imshow('result', union_mask)
    cv2.waitKey(0)

    return union_mask




if __name__ == "__main__":
    mask = cv2.imread(PATH+'union_mask.png')
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    borders = cv2.imread(PATH+'border_mask.png')
    borders = cv2.cvtColor(borders, cv2.COLOR_BGR2GRAY)
    wattershed_transform(gray_mask.copy())
    wattershed_transform_border(gray_mask.copy(), borders.copy())



