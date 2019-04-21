import tensorflow as tf
from tensorflow.keras.utils import Sequence

from matplotlib import pyplot as plt
from albumentations import Compose, RandomSizedCrop, OpticalDistortion, ElasticTransform, RandomRotate90, \
    VerticalFlip, OneOf, GridDistortion, Blur, MotionBlur, GaussNoise, RGBShift, ToGray, RandomBrightnessContrast, \
    RandomGamma, CLAHE, HueSaturationValue, ChannelShuffle, CenterCrop
from pathlib import Path
import numpy as np
import cv2


def get_5_crops_gen(resolution):
    """returns a get_5_crops function with the corresponding resolution"""

    def get_5_crops(images, masks, resolution=resolution):
        """returns the concatenation of the 5 crops (top-left, top-right, center, bottom-left, bottom-right)
        of the image and the masks"""
        processed_images = []
        processed_masks = []
        for image, mask in zip(images, masks):
            img_size = image.shape
            for crop_idx in [(0, resolution, 0, resolution),
                             (0, resolution, -resolution, img_size[1]),
                             ((img_size[0] - resolution) // 2, (img_size[0] + resolution) // 2,
                              (img_size[1] - resolution) // 2, (img_size[1] + resolution) // 2),
                             (-resolution, img_size[0], 0, resolution),
                             (-resolution, img_size[0], -resolution, img_size[1])]:
                crops_image = image[crop_idx[0]:crop_idx[1], crop_idx[2]:crop_idx[3], :]
                crops_mask = mask[crop_idx[0]:crop_idx[1], crop_idx[2]:crop_idx[3], :]
                processed_images.append(crops_image)
                processed_masks.append(crops_mask)

        return processed_images, processed_masks

    return get_5_crops


class DataGenerator(Sequence):
    def __init__(self, data_directories, output_masks, ids_list=None, batch_size=32, resolution=128, n_channels_input=3,
                 shuffle=True, performs_data_augmentation=True, non_border_cells_weights=30, sec_data_dir_factor=1):
        """:param data_directories: the directories containing the data as a tuple of 2 elements, the first one is the
        main dataset and the second one is another dataset to use (the second one is optionnal, i.e. (name1,) and
        (name1, name2) are accepted)
        :param output_masks: the name of the masks to output
        :param ids_list: the list of ids of the images to use a pair whose first elements corresponds to the first
        dataset and the second one to the other, each of those can be None to signify we want to use all of the images
        :param batch_size: the batch size to use
        :param resolution: the resolution of the images/masks
        :param n_channels_input: the number of channels in an input image
        :param shuffle: wether to shuffle the inputs
        :param performs_data_augmentation: wether to perform data augmentation or not
        :param non_border_cells_weights: if 'weight_mask' in output_masks: the weight to give to non_border_cells
        pixels (border cells pixels have a weight of 255)
        :param sec_data_dir_factor: the relative importance of the images from the secondary dataset w.r.t. the main one
        (ignore if there is no secondary dataset)"""
        self.main_data_directory = Path(data_directories[0])
        if len(data_directories) == 1:
            self.secondary_data_directoy = None
        else:
            self.secondary_data_directoy = Path(data_directories[1])

        self.batch_size = batch_size
        self.n_channels = n_channels_input
        self.shuffle = shuffle
        self.resolution = resolution
        self.output_masks = output_masks
        self.non_border_cells_weights = non_border_cells_weights
        self.performs_data_augmentation = performs_data_augmentation

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
            self.preprocessing = get_5_crops_gen(self.resolution)

        main_ids_list = ids_list[0] if ids_list[0] is not None else \
            [directory.name for directory in self.data_directory.iterdir() if directory.is_dir()]

        if self.secondary_data_directoy is None:
            secondary_ids_list = []
        else:
            secondary_ids_list = ids_list[1] if ids_list[1] is not None else \
                [directory.name for directory in self.secondary_data_directoy.iterdir() if directory.is_dir()]
            secondary_ids_list = secondary_ids_list * sec_data_dir_factor  # repeat the images several times to
            # compensate the fact they are bigger, makes sense since we always take random crops

        self.main_indexes_limit = len(main_ids_list)
        main_ids_list.extend(secondary_ids_list)  # main_ids_list now contains all of the samples
        self.ids_list = np.array(main_ids_list)
        self.indexes = np.arange(self.ids_list.shape[0])

    def on_epoch_end(self):
        """ called at the end of each epoch """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """ return the number of batches used to compute the number of steps per epoch """
        return len(self.ids_list) // self.batch_size

    def __getitem__(self, item):
        """ return a batch """
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]

        ids_list_subset = self.ids_list[indexes]

        return self.__data_generation(indexes, ids_list_subset)

    def get_some_items(self, items):
        """same as item but only gives back a single element of the first set (so )"""
        indexes = self.indexes[items]
        id_item = self.ids_list[indexes]
        return self.__data_generation(indexes, id_item)

    def __data_generation(self, indexes, ids_list_subset):
        """take a list of ids of images as well as their indexes and returns the corresponding batch to feed
        the network"""
        # get the images/masks from the files
        images_paths = ((self.main_data_directory / image_id / "images" / (image_id + ".png"))
                        if index < self.main_indexes_limit else
                        (self.secondary_data_directoy / image_id / "images" / (image_id + ".tif"))
                        for index, image_id in zip(indexes, ids_list_subset))

        images = (cv2.imread(str(image_path)) for image_path in images_paths)
        masks = (self.get_channels_masks(index=index, id_image=current_id)
                 for index, current_id in zip(indexes, ids_list_subset))

        # data augmentation
        if self.performs_data_augmentation:
            processed_images = []
            processed_masks = []
            for image, mask in zip(images, masks):
                augmented = self.preprocessing(image=image, mask=mask)
                processed_images.append(augmented['image'])
                processed_masks.append(augmented['mask'])
        else:
            processed_images, processed_masks = self.preprocessing(images, masks)

        # convert to numpy array and rescale
        processed_images = np.array(processed_images)
        processed_masks = np.array(processed_masks)
        processed_images = processed_images.astype(np.float16) / 255.
        processed_masks = processed_masks.astype(np.float16) / 255.

        return processed_images, processed_masks

    def get_channels_masks(self, index, id_image, processed_dir_name="processed_masks"):
        """ return the masks of an image. The image needs to have been processed and the compiled mask must be in the
        subdirectory named processed_dir_name"""

        masks_dir = (self.main_data_directory if index < self.main_indexes_limit
                     else self.secondary_data_directoy) / id_image / processed_dir_name

        if not masks_dir.is_dir():
            raise ValueError("image has not been processed yet")

        # read each image as a channel
        def process_mask(mask_name):
            mask = cv2.imread(str(masks_dir / (mask_name + ".png")), 0)
            if mask_name == "weight_mask":
                mask[mask == 0] = self.non_border_cells_weights  # so that the non-border cells are weakly weighted but
                # still taken into account
            return mask

        mask_channels = (process_mask(mask_name) for mask_name in self.output_masks)

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

    data_generator = DataGenerator(data_directories=(data_dir,), output_masks=('border_mask', 'union_mask'))

    x, y = data_generator[0]
    print(x.shape)
    print(y.shape)
