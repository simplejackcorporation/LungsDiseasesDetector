import numpy as np
import keras
import os
import glob

import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    # https://github.com/tensorflow/tensorflow/issues/38064

    def __init__(self,
                 base_path,
                 batch_size=1):
        print("DataGenerator init")
        self.batch_size = batch_size
        self.items_paths = glob.glob(os.path.join(base_path, '*/*')) # ID / (LEFT/RIGHT)


        self.indexes = np.arange(len(self.items_paths))
        self.read_function = cv2.imread

        print("base_path {} "
              "items paths len: {}: "
              "read_function: {}".format(
            base_path,
            len(self.items_paths),
            self.read_function))

        # PRIVATE (for now)
        self.WIDTH = 224
        self.HEIGHT = 224
        self.CHANNELS_N = 3

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.items_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_items_paths = [self.items_paths[k] for k in indexes]

        # Generate data
        X = self.__data_generation(batch_items_paths)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)


    @tf.autograph.experimental.do_not_convert
    def __data_generation(self, items_paths):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        images_list = []
        for i, item_path in enumerate(items_paths):
            image = self.read_function(item_path)
            image = self.preprocess_image(image)
            images_list.append(image)
            # Store class
            # y[i] = self.labels[ID]

        x = np.array(images_list)
        temp_y = np.zeros((self.batch_size, 1))
        return x, temp_y

    def preprocess_image(self, image):
        image = cv2.resize(image, (self.WIDTH, self.HEIGHT))

        return image


import cv2

if __name__ == '__main__':
    temp_base_path = r"C:\Users\m\Desktop\LUNGS\lungs-finder\dataset\train"

    data_generator = DataGenerator(base_path=temp_base_path)
    item = data_generator[0]
    cv2.imshow("cropped", item[0])
    cv2.waitKey(0)
