import numpy as np
from numba import njit
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import random

FLOAT_TYPE = np.float32


@njit(error_model='numpy', parallel=True, fastmath=True)
def get_run_length_enc(array):
    """given a 2D float32 image of 0's and 1's, returns the run-length encoding of that array
    for submission (a list of integers)"""
    flattened_array = np.transpose(array).flatten()
    run_length = []
    # handles the first element of the array
    if flattened_array[0] == 1.:
        run_length.append(1)
    # body of the array
    for i in range(1, flattened_array.shape[0]):
        if flattened_array[i] == 1. and flattened_array[i - 1] == 0.:
            run_length.append(i + 1)
        elif flattened_array[i] == 0. and flattened_array[i - 1] == 1.:
            run_length.append(i - run_length[-1] + 1)
    # last element
    if flattened_array[-1] == 1.:
        run_length.append(flattened_array.shape[0] - run_length[-1] + 1)

    return run_length


def from_run_length_enc(run_length, shape):
    reversed_shape = (shape[1], shape[0])
    image_array = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for one_index, length in zip(run_length[::2], run_length[1::2]):
        start = one_index - 1
        image_array[start:start + length] = 1.

    return image_array.reshape(reversed_shape).transpose()


def split_train_val(data_dir, train_prop, val_last=True):
    """splits the data into training and validation sets
    :param data_dir: the directory containing the data
    :param train_prop: the proportion of the dataset used for training
    :param val_last: whether the validation data is to be taken after (in the shuffled list of elements) the training
    data or the other way around
    :return ids_list_train, ids_list_val: the ids of the images in the training and validation set"""
    ids_list = [directory.name for directory in Path(data_dir).iterdir() if directory.is_dir()]

    random.Random(17).shuffle(ids_list)
    last_train_element = int(train_prop * len(ids_list))
    if val_last:
        return ids_list[:last_train_element], ids_list[last_train_element:]
    else:
        return ids_list[-last_train_element:], ids_list[:-last_train_element]


def get_resized_images(image, output_resolution=(128, 128), overlapping=(10, 10), default_value=0.0):
    """given a 2D float image, return cropped images from the input image
    :param image: image coded in float between 0 and 1
    :param output_resolution: tuple of int, the resolution of the output images (the two first dimensions)
    :param overlapping: tuple of int, how much pixel cropped images overlap each other
    :param default_value: default value if padding
    :return a list of the cropped images and a list the coordonate of the upper left corner in the  original image (
    from the upper left corner) of the
    coresponding cropped image in the first list"""
    image_shape = tuple(int(dim) for dim in image.shape)
    output_resolution = tuple(int(dim) for dim in output_resolution)
    number_hor_crops = image_shape[0] // (output_resolution[0] - overlapping[0]) + \
        (1 if image_shape[0] % (output_resolution[0] - overlapping[0]) else 0)
    number_ver_crops = image_shape[1] // (output_resolution[1] - overlapping[1]) + \
        (1 if image_shape[1] % (output_resolution[1] - overlapping[1]) else 0)

    # compute x coordonates of top left corner of the cropped images
    x_crops = []
    for i in range(number_hor_crops // 2):
        x_crops.append(i * (output_resolution[0] - overlapping[0]))
        x_crops.append(image_shape[0] - output_resolution[0] -
                       i * (output_resolution[0] - overlapping[0]))

    if number_hor_crops % 2:
        x_crops.append(image_shape[0] // 2 - output_resolution[0] // 2)
    x_crops.sort()

    # compute y coordonates of top left corner of the cropped images
    y_crops = []
    for i in range(number_ver_crops // 2):
        y_crops.append(i * (output_resolution[1] - overlapping[1]))
        y_crops.append(image_shape[1] - output_resolution[1] -
                       i * (output_resolution[1] - overlapping[1]))

    if number_ver_crops % 2:
        y_crops.append(image_shape[1] // 2 - output_resolution[1] // 2)
    y_crops.sort()

    cropped_images = []
    coordonates = []

    for x_crop in x_crops:
        for y_crop in y_crops:
            coordonates.append((x_crop, y_crop))

            current_image = np.full((*output_resolution[:2], *image_shape[2:]), default_value)

            x = - x_crop if x_crop < 0 else 0
            y = - y_crop if y_crop < 0 else 0

            x_crop = 0 if x_crop < 0 else x_crop
            y_crop = 0 if y_crop < 0 else y_crop

            width = (output_resolution[0] if output_resolution[0] < image_shape[0] else image_shape[0])
            height = (output_resolution[1] if output_resolution[1] < image_shape[1] else image_shape[1])

            current_image[x:x + width, y:y + height] = image[x_crop:x_crop + width, y_crop:y_crop + height]

            cropped_images.append(current_image)

    return cropped_images, coordonates


def get_full_resolution(cropped_images, coordinates, original_resolution, round=False, cropped_image_weight=None):
    """Compute the full resolution image from list of cropped image and the coordinates of their upper left corner
    :param cropped_images: list of cropped images
    :param coordinates: list of the coordinates  of the upper left corners of the cropped images
    :param original_resolution: resolution of the original image that has to be reconstructed
    :param round: if the output has to be round
    :param cropped_image_weight: optional weight of the pixel of the cropped images
    :return:
    """
    cropped_image_shape = cropped_images[0].shape
    image = np.zeros((*original_resolution, *cropped_image_shape[2:]), dtype=FLOAT_TYPE)
    image_load = np.zeros(original_resolution, dtype=FLOAT_TYPE)

    if cropped_image_weight is None:
        cropped_image_weight = np.ones(cropped_image_shape[:2], dtype=FLOAT_TYPE)

    cropped_image_ones = np.ones(cropped_image_shape, dtype=FLOAT_TYPE)

    if len(cropped_image_weight.shape) < len(cropped_image_shape):
        cropped_image_weight = cropped_image_weight[:, :, np.newaxis]

    for cropped_image, coordonate in zip(cropped_images, coordinates):
        x_crop, y_crop = coordonate
        cropped_image_shape = tuple(int(dim) for dim in cropped_image.shape)

        x = -x_crop if x_crop < 0 else 0
        y = -y_crop if x_crop < 0 else 0

        x_crop = 0 if x_crop < 0 else x_crop
        y_crop = 0 if y_crop < 0 else y_crop

        width = cropped_image_shape[0] if cropped_image_shape[0] < original_resolution[0] \
            else original_resolution[0]
        height = cropped_image_shape[1] if cropped_image_shape[1] < original_resolution[1] \
            else original_resolution[1]

        image[x_crop:x_crop + width, y_crop:y_crop + height] += \
            cropped_image[x:x + width, y:y + height] * cropped_image_weight[x:x + width, y:y + height]

        image_load[x_crop:x_crop + width, y_crop:y_crop + height] += cropped_image_ones[x:x + width, y:y + height]

    if len(image.shape) > len(image_load.shape):
        image = (image / image_load[:, :, np.newaxis])
    else:
        image = (image / image_load)

    if round:
        return image.round()

    return image


def predict_image(model, image, last_layer_res=None):
    """
    predicts the mask of the image by calling several times the model (model input res < image res)
    :param model: the model to perform predictions
    :param image: the image we want to segment sementically
    :param last_layer_res: if not None, the resolution of the output of the model, otherwise model.output.shape will
    be used
    :return: the full predicted image
    """
    out_res = model.output.shape[1:3] if last_layer_res is None else [last_layer_res, last_layer_res]
    images, coordinates = get_resized_images(image, out_res)
    images_array = np.array(images)

    predictions = model.predict(images_array)

    predictions_list = np.split(predictions, predictions.shape[0], axis=0)

    for i, prediction in enumerate(predictions_list):
        predictions_list[i] = prediction.squeeze()

    prediction_full = get_full_resolution(predictions_list, coordinates, image.shape[:2], round=True)

    return prediction_full


def test_crop_image():
    image = np.random.rand(100).reshape((10, 10))
    cropped_images, coordonates = get_resized_images(image, (13, 13), (1, 1))
    print(coordonates)
    image_rebuilded = get_full_resolution(cropped_images, coordonates, image.shape[:2], round=False)

    diff = image - image_rebuilded
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(image_rebuilded)
    plt.figure()
    plt.imshow(diff)
    print("number of errors: {}".format(np.sum(diff > 1e-7)))

    images_dir = Path("data/stage1_train")
    image_ids = ["e9b8ad127f2163438b6236c74938f43d7b4863aaf39a16367f4af59bfd96597b",
                 "f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3",
                 "29ea4f6eb4545f43868a9b40a60000426bf8dfd9d062546656a37bd2a2aaf9ec",
                 "29780b28e6a75fac7b96f164a1580666513199794f1b19a5df8587fe0cb59b67"]

    for image_id in image_ids:
        image_path = images_dir / image_id / "images" / (image_id + ".png")

        image = cv2.imread(str(image_path))
        print(image.shape)
        image = image / 255.

        cropped_images, coordonates = get_resized_images(image, (256, 256))

        image_rebuilded = get_full_resolution(cropped_images, coordonates, image.shape[:2], round=False)

        diff = image - image_rebuilded

        print("number of errors: {}".format(np.sum(diff > 1e-7)))

    for image_id in image_ids:
        image_path = images_dir / image_id / "images" / (image_id + ".png")

        image = cv2.imread(str(image_path))
        print(image.shape)
        image = image / 255.

        cropped_images, coordonates = get_resized_images(image, (128, 128))

        image_rebuilded = get_full_resolution(cropped_images, coordonates, image.shape[:2], round=False)

        diff = image - image_rebuilded

        print("number of errors: {}".format(np.sum(diff > 1e-7)))

    for image_id in image_ids:
        image_path = images_dir / image_id / "images" / (image_id + ".png")

        image = cv2.imread(str(image_path))
        print(image.shape)
        image = image / 255.

        cropped_images, coordonates = get_resized_images(image, (128, 128), (0, 0))

        image_rebuilded = get_full_resolution(cropped_images, coordonates, image.shape[:2], round=False)

        diff = image - image_rebuilded

        print("number of errors: {}".format(np.sum(diff > 1e-7)))


def test_run_length_enc_dec():
    np.random.seed(17)
    mask = np.round(np.random.random(100).reshape((10, 10)))
    print(mask)
    run_length = get_run_length_enc(mask)
    print(run_length)
    rebuilded_mask = from_run_length_enc(run_length, (10, 10))

    diff = mask - rebuilded_mask
    number_of_errors = np.sum(diff > 1e-7)

    if number_of_errors:
        print("failed : there was {} errors".format(number_of_errors))
    else:
        print("succes: {} error found".format(number_of_errors))


if __name__ == "__main__":
    test_run_length_enc_dec()
