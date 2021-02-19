import numpy as np
import pydicom
import os
import keras
import lungs_finder as lf
import zipfile
import cv2
import pandas as pd


def scale(arr, out_range=(0, 255)):
    # y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    y = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
    return y


def crop_rect(image, rectangle):
    x, y, width, height = rectangle
    crop_img = image[y:y + height, x:x + width]
    return crop_img


def proportional_resize(image, max_side):
    if image.shape[0] > max_side or image.shape[1] > max_side:
        if image.shape[0] > image.shape[1]:
            height = max_side
            width = int(height / image.shape[0] * image.shape[1])
        else:
            width = max_side
            height = int(width / image.shape[1] * image.shape[0])
    else:
        height = image.shape[0]
        width = image.shape[1]

    return cv2.resize(image, (width, height))


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 base_path,
                 list_IDs,
                 batch_size=32,
                 sub_folder=None):
        self.sub_folder = sub_folder
        self.base_path = base_path
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        list_IDs_temp = self.list_IDs[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        X = []
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            if zipfile.is_zipfile(self.base_path):
                archive = zipfile.ZipFile(self.base_path, 'r')
                if self.sub_folder is not None:
                    path = self.sub_folder + os.sep + ID
                else:
                    path = ID

                item_file = archive.open(path)
            else:
                item_file = os.path.join(self.base_path, ID)

            image = pydicom.dcmread(item_file).pixel_array

            image = scale(image)

            image = proportional_resize(image, 512)

            right_lung_hog_rectangle = lf.find_right_lung_hog(image)

            right_lung_img = None
            if right_lung_hog_rectangle is not None:
                right_lung_img = crop_rect(image, right_lung_hog_rectangle)

            left_lung_img = None
            left_lung_hog_rectangle = lf.find_left_lung_hog(image)
            if left_lung_hog_rectangle is not None:
                left_lung_img = crop_rect(image, left_lung_hog_rectangle)

            X.append((left_lung_img, right_lung_img))
            # Store class
            # y[i] = self.labels[ID]

        return X


if __name__ == '__main__':
   data_generator = DataGenerator(base_path=r"C:\Users\m\Desktop\LUNGS\lungs-finder\Example",
                                  list_IDs=["0e40d75b6a0f50c6855b6f3203f421a2.dicom"])
   items = data_generator[0]
   left, right = items[0]
   cv2.imshow("Left", left)
   cv2.imshow("Right", left)
   cv2.waitKey(0)

