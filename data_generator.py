import tensorflow as tf
from tensorflow.keras.utils import Sequence

from pathlib import Path
import numpy as np
import cv2


class DataGenerator(Sequence):
    def __init__(self, data_directory, output_masks, ids_list=None, batch_size=32,
                 resolution=(128, 128), n_channels_input=3, shuffle=True):
        self.data_directory = Path(data_directory)
        self.batch_size = batch_size
        self.resolution = resolution
        self.n_channels = n_channels_input
        self.shuffle = shuffle
        self.output_masks = output_masks

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
        """ take a list of ids of images and return the corresponding batch to feed the network """
        # get the images from the files
        images_paths = (self.data_directory / image_id / "images" / (image_id + ".png")
                        for image_id in ids_list_subset)

        images = (cv2.imread(str(image_path)) for image_path in images_paths)
        images = (cv2.resize(image, self.resolution) for image in images)  # TODO: gérer ça avec des random_crop dans un second temps

        # images is a generator, convert it to array
        images = np.array(list(images))
        # convert array from uint [0, 255] to float16 [0, 1]
        images = images.astype(np.float16) / 255.
        
        masks = (self.get_channels_masks(id_image=current_id) for current_id in ids_list_subset)

        # mask is a generator, convert it to array
        masks = np.array(list(masks))
        # convert array from uint [0, 255] to float16 [0, 1]
        masks = masks.astype(np.float16) / 255.

        return images, masks

    def get_channels_masks(self, id_image, processed_dir_name="processed_masks"):
        """ return the mask of an image. The image needs to have been processed and the compiled mask must be in the
        subdirectory named processed_dir_name"""

        masks_dir = self.data_directory / id_image / processed_dir_name

        if not masks_dir.is_dir():
            raise ValueError("image {} has not been processed yet")

        # read each image as a channel
        mask_channels = (cv2.imread(str(masks_dir / (mask_name + ".png")), 0) for mask_name in self.output_masks)
        mask_channels = (cv2.resize(mask_channel, self.resolution) for mask_channel in mask_channels)  # TODO: gérer ça avec des random_crop dans un second temps

        # each channel must be of dimension 2: shape [size1, size2, 1] => [size1, size2]
        mask_channels = (np.squeeze(mask_channel) for mask_channel in mask_channels)

        # concatinate all the channels
        mask = np.array(list(mask_channels))

        # TODO: gérer ça avec des random_crop dans un second temps
        mask[mask != 0] = 255  # because cv2.resize makes interpolations so we have values between 0 and 255

        # channels dimension must come last : shape [n_channels, size1, size2] => [size1, size2, n_channels]
        dim_permutation = list(range(len(mask.shape)))
        dim_permutation[0] = dim_permutation[-1]
        dim_permutation[-1] = 0
        mask = np.transpose(mask, axes=dim_permutation)

        return mask


# Some tests
# TODO: make more tests?
def test_data_generator(data_dir="data/stage1_train"):

    data_generator = DataGenerator(data_directory=data_dir, output_masks=('border_mask', 'union_mask'))

    x, y = data_generator[0]
    print(x.shape)
    print(y.shape)
