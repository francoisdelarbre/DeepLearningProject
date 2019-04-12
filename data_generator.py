import tensorflow as tf
from tensorflow.keras.utils import Sequence

from matplotlib import pyplot as plt
from albumentations import Compose, RandomSizedCrop, OpticalDistortion, ElasticTransform, RandomRotate90, \
    VerticalFlip, OneOf, GridDistortion, Blur, MotionBlur, GaussNoise, RGBShift, ToGray, RandomBrightnessContrast, \
    RandomGamma, CLAHE, HueSaturationValue, ChannelShuffle, CenterCrop
from pathlib import Path
import numpy as np
import cv2


class DataGenerator(Sequence):
    def __init__(self, data_directory, output_masks, ids_list=None, batch_size=32,
                 resolution=128, n_channels_input=3, shuffle=True, performs_data_augmentation=True):
        self.data_directory = Path(data_directory)
        self.batch_size = batch_size
        self.n_channels = n_channels_input
        self.shuffle = shuffle
        self.resolution = resolution
        self.output_masks = output_masks

        # data augmentation pipeline:
        if performs_data_augmentation:
            small_color_augm = Compose([RandomBrightnessContrast(p=1), RandomGamma(p=1), CLAHE(p=1)], p=.25)
            medium_color_augm = Compose([CLAHE(p=1), HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50,
                                                                        val_shift_limit=50, p=1)], p=.25)
            huge_color_augm = Compose([ChannelShuffle(p=1)], p=.5)  # ChannelShuffle more likely than the 2 others
            self.preprocessing = Compose([RandomSizedCrop(min_max_height=(int(resolution*2/3), int(resolution*3/2)),
                                                          height=resolution, width=resolution,
                                                          interpolation=cv2.INTER_NEAREST, p=1.0),
                                          OneOf([OpticalDistortion(p=1.0, distort_limit=2, shift_limit=0.5),
                                                 GridDistortion(p=.5), ElasticTransform(
                                                  p=1.0, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)], p=.5),
                                          RandomRotate90(p=.5), VerticalFlip(p=.5),
                                          OneOf([small_color_augm, medium_color_augm, huge_color_augm], p=.8),
                                          OneOf([Blur(p=.3), GaussNoise(p=.3), MotionBlur(p=.3)], p=.5),
                                          OneOf([RGBShift(p=.5), ToGray(p=.5)], p=.3)
                                          ])  # TODO: add nuclei to cells to improve cell separation, use transforms.Lambda
        else:
            self.preprocessing = Compose([CenterCrop(resolution, resolution, p=1.0)])

        if ids_list is None:
            ids_list = [directory.name for directory in self.data_directory.iterdir() if directory.is_dir()]

        self.ids_list = np.array(ids_list)
        self.indexes = np.arange(len(self.ids_list))

    def on_epoch_end(self):
        """ called at the end of each epoch """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """ return the number of batchs used to compute the number of steps per epoch """
        return len(self.ids_list) // self.batch_size

    def __getitem__(self, item):
        """ return a batch """
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]

        ids_list_subset = self.ids_list[indexes]

        return self.__data_generation(ids_list_subset)

    def get_unique_item(self, item):
        """same as item but only gives back a single element"""
        id_item = self.ids_list[self.indexes[item]]
        return self.__data_generation([id_item])

    def __data_generation(self, ids_list_subset):
        """ take a list of ids of images and return the corresponding batch to feed the network"""
        # get the images/masks from the files
        images_paths = (self.data_directory / image_id / "images" / (image_id + ".png")
                        for image_id in ids_list_subset)

        images = (cv2.imread(str(image_path)) for image_path in images_paths)
        masks = (self.get_channels_masks(id_image=current_id)
                 for current_id in ids_list_subset)

        # data augmentation
        processed_images = []
        processed_masks = []
        for image, mask in zip(images, masks):
            augmented = self.preprocessing(image=image, mask=mask)
            processed_images.append(augmented['image'])
            processed_masks.append(augmented['mask'])

        # convert to numpy array and rescale
        processed_images = np.array(processed_images)
        processed_images = processed_images.astype(np.float16) / 255.
        processed_masks = np.array(processed_masks)
        processed_masks = processed_masks.astype(np.float16) / 255.

        return processed_images, processed_masks

    def get_channels_masks(self, id_image, processed_dir_name="processed_masks"):
        """ return the masks of an image. The image needs to have been processed and the compiled mask must be in the
        subdirectory named processed_dir_name"""

        masks_dir = self.data_directory / id_image / processed_dir_name

        if not masks_dir.is_dir():
            raise ValueError("image {} has not been processed yet")

        # read each image as a channel
        mask_channels = (cv2.imread(str(masks_dir / (mask_name + ".png")), 0) for mask_name in self.output_masks)

        # each channel must be of dimension 2: shape [size1, size2, 1] => [size1, size2]
        mask_channels = (np.squeeze(mask_channel) for mask_channel in mask_channels)

        # concatenate all the channels
        mask = np.array(list(mask_channels))

        # channels dimension must come last : shape [n_channels, size1, size2] => [size1, size2, n_channels]
        mask = np.transpose(mask, axes=[1, 2, 0])

        return mask


def visualize(image, mask, original_image=None, original_mask=None):
    """visualization function used for debugging, comes from
    https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb"""
    fontsize = 18

    def augment_channels(array):
        """transforms a one-channel array into a 3-channels RGB array"""
        new_array = np.zeros((array.shape[0], array.shape[1], 3), dtype=array.dtype)
        for i in range(3):
            new_array[:, :, i] = array[:, :, 0]
        return new_array

    if mask.shape[2] == 1:
        mask = augment_channels(mask)

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        if original_mask.shape[2] == 1:
            original_mask = augment_channels(original_mask)

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


# Some tests
# TODO: make more tests?
def test_data_generator(data_dir="data/stage1_train"):

    data_generator = DataGenerator(data_directory=data_dir, output_masks=('border_mask', 'union_mask'))

    x, y = data_generator[0]
    print(x.shape)
    print(y.shape)
