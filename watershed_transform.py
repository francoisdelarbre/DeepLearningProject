import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

# PATH = 'data/extra_data/TCGA-18-5592-01Z-00-DX1/processed_masks/'
PATH = 'data/stage1_train/4a424e0cb845cf6fd4d9fe62875552c7b89a4e0276cf16ebf46babe4656a794e/processed_masks/'

def watershed_transform(union_mask):
    """Wattershed on union mask. use distance map to find markers
        :param union_mask, a uint8 mask"""

    # compute distance map of mask
    tmps = union_mask.copy()
    dist_transform = cv2.distanceTransform(tmps, cv2.DIST_L2, 5)


    # find big blop in the images, we try to separate each blop with watershed
    tmps = np.uint8(tmps)
    ret, blops = cv2.connectedComponents(tmps)
    blops += 1  # background is 1, 0 will be for the zone to flood by watershed

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

    ret, labels = cv2.connectedComponents(union_mask, connectivity=4)
    label_hue = np.uint8(179 * labels / np.max(labels))
    for i in range(1, ret):
        label_hue[labels==i] = random.randint(1, 254)

    # Map component labels to hue val
    # label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # plt.imshow(labeled_img)
    # plt.show()

    return union_mask


def watershed_transform_border(union_mask, border_mask):
    """Wattershed on union mask. use distance map to find markers
        :param union_mask, a uint8 mask
        :param border_mask, a uint8 mask"""

    # compute distance map of mask
    tmps = union_mask.copy()
    tmps = cv2.subtract(tmps, border_mask)
    tmps[tmps < 0] = 0

    dist_transform = cv2.distanceTransform(tmps, cv2.DIST_L2, 5)

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

    return union_mask


def watershed_transform_center(union_mask, center_mask):
    """Wattershed on union mask. use distance map to find markers
        :param union_mask, a uint8 mask
        :param border_mask, a uint8 mask"""

    # compute distance map of mask
    tmps = union_mask.copy()
    center_mask = np.bitwise_and(center_mask, union_mask) #put center that are in nackground to 0
    sure_fg = center_mask
    not_sure = cv2.subtract(union_mask, sure_fg)


    # find big blop in the images, we try to separate each blop with wattershed
    sure_fg = np.uint8(sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # background is 1, 0 will be for the zone to flood by wattershed

    markers[not_sure == 255] = 0

    segmented = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    result = cv2.watershed(segmented, markers)

    # remove borders from cells
    union_mask[result == -1] = 0

    return union_mask


if __name__ == "__main__":
    mask = cv2.imread(PATH+'union_mask.png')
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray_mask)
    plt.show()
    borders = cv2.imread(PATH+'border_mask.png')
    borders = cv2.cvtColor(borders, cv2.COLOR_BGR2GRAY)

    centers = cv2.imread(PATH+'center_mask.png')
    centers = cv2.cvtColor(centers, cv2.COLOR_BGR2GRAY)
    watershed_transform(gray_mask.copy())
    watershed_transform_border(gray_mask.copy(), borders.copy())
    watershed_transform_center(gray_mask.copy(), centers.copy())


