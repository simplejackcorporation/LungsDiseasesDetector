import numpy as np
import os
import glob
import keras
import cv2
from utils import Utils
import time
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 base_path,
                 batch_size=8,
                 is_val=False):

        self.is_val = is_val
        self.shuffle = True

        self.base_path = base_path
        self.batch_size = batch_size

        self.pandas_data_frame = pd.read_csv(os.path.join(self.base_path, "train.csv"))
        self.class_counts_dict = Utils.get_class_count_dict(self.pandas_data_frame)

        self.items_paths = []
        for key, value in self.class_counts_dict.items():
            for img_id_key in value.keys():
                print(img_id_key)
                img_name = "small_{}.png".format(img_id_key)
                path = os.path.join(base_path, img_name)
                self.items_paths.append((path, key))

        self.equalize_dataset(self.is_val)


        # print("self.items_paths", self.items_paths)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.items_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        temp_n = 5

        batch_items_paths = self.items_paths[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.empty((self.batch_size, 224, 224, 3))
        # rect_Y = np.zeros((self.batch_size, temp_n, 4))
        # class_Y = np.zeros((self.batch_size, temp_n))
        class_Y = np.zeros((self.batch_size, 1))

        ind = 0
        while ind < self.batch_size:
            item_path, class_id = batch_items_paths[ind]

            image = cv2.imread(item_path)
            # image = Utils.cropLungsAreaImage(image, item_path)


            try:
                image = cv2.resize(image, (224, 224))
            except Exception as e:
                print("VOVA HERE")
                print(e)
                print(item_path)
                ind += 1
                continue

            image = Utils.normalize(image)

            if class_id == 14:
                class_Y[ind] = 0
            else:
                class_Y[ind] = 1

            # print(Y)
            # print(id)
            X[ind] = image
             # = Y[0]
             # = Y[1]

            ind += 1

        # print("class_y shape", class_Y.shape)
        # print("rect_Y shape", rect_Y.shape)
        # return X, [rect_Y, class_Y]

        return X, class_Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.items_paths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def getYForID(self, id):
        for index, row in self.pandas_data_frame.iterrows():
            if row.image_id == id:
                class_id = row.class_id
                if class_id == 14: # no findings
                    return [[self.get_average_random_frame()], class_id]

                x, y, width, height = row.x_min, row.y_min, row.x_max - row.x_min, row.y_max - row.y_min
                return [x, y, width, height], class_id


    def equalize_dataset(self, is_val):
        lengths = []
        for key, value in self.class_counts_dict.items():
            lengths.append(len(list(value.keys())))

        min_count_value = min(lengths)
        delta = 3
        for key in self.class_counts_dict.keys():
            sliced_img_objs = self.class_counts_dict[key]

            buff_dict = sliced_img_objs.copy()
            for index, item in enumerate(sliced_img_objs):
                if index > min_count_value:
                    break

                if is_val is True:
                    if index > delta:
                        buff_dict.pop(item, None)
                else:
                    if index < delta:
                        buff_dict.pop(item, None)

            self.class_counts_dict[key] = buff_dict


DICOM_PATH = r"C:\Users\m\Desktop\datasets\dicom_train"
lungs_train_2000_PATH = r"C:\Users\m\Desktop\datasets\lungs_train_2000"

def test_show_image():
    data_generator = DataGenerator(base_path=os.path.join(lungs_train_2000_PATH, "train"),
                                   y_base_path=lungs_train_2000_PATH)

    start_time = time.time()
    items = data_generator[0]
    print(items[1][1].shape)
    print("Item generation time:", time.time() - start_time)

    # cv2.imshow("batch_image", batch_image)
    cv2.waitKey(0)

import collections

def test_class_dict():
    data_generator = DataGenerator(base_path=os.path.join(lungs_train_2000_PATH, "train"),
                                   y_base_path=lungs_train_2000_PATH)

    for key, value in data_generator.class_counts_dict.items():
        print("key {}, count {}".format(key, len(value)))



if __name__ == '__main__':
    # test_show_image()
    test_class_dict()

