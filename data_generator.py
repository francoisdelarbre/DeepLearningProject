import tensorflow as tf

from tensorflow.keras.utils import Sequence

from pathlib import Path

import cv2

import numpy as np


class DataGenerator(Sequence):
    def __init__(self, data_directory, ids_list=None, batch_size=32,
                 resolution=(128, 128), n_channels=3, shuffle=True):
        self.data_directory = Path(data_directory)
        self.batch_size = batch_size
        self.resolution = resolution
        self.n_channels = n_channels
        self.suffle = shuffle

        if ids_list is None:
            ids_list = [dir.name for dir in self.data_directory.iterdir()
                       if dir.is_dir()]

        self.ids_list = np.array(ids_list)
        self.indexes = np.arange(len(self.ids_list))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.ids_list) // self.batch_size

    def __getitem__(self, item):
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]

        ids_list_tmp = self.ids_list[indexes]

        return self.__data_generation(ids_list_tmp)

    def __data_generation(self, ids_list_tmp):
        images_paths = (self.data_directory / image_id / "images" / image_id
                       for image_id in ids_list_tmp)
        images_tensors = (tf.io.read_file(str(image_input_path),
                                   name=image_input_path.name) for
                          image_input_path in images_paths)

        images = (tf.image.decode_png(image_tensor, channels=self.n_channels)
                  for image_tensor in images_tensors)

        images = (tf.image.resize_images(image, self.resolution[:2])
                  for image in images)

        images_tensor = tf.convert_to_tensor(list(images))

        masks = (self.get_channels_masks(id_image=current_id)
                       for current_id in ids_list_tmp)

        # TODO: mask is a generator, convert it to tensor
        return images_tensor, masks


    def get_channels_masks(self, id_image):
        # TODO: make this function
        masks_dir = self.data_directory / id_image / "masks"
        mask = None
        for mask_path in masks_dir.iterdir():
            current_mask = cv2.imread(str(mask_path))

            # TODO: I know, hard to make it uglier
            if mask is None:
                mask = current_mask
            else:
                mask = mask + current_mask
        print(mask.dtype)
        return mask

def test_data_generator(data_dir="data/stage1_train"):

    data_generator = DataGenerator(data_directory=data_dir)

    X, y = data_generator[0]


