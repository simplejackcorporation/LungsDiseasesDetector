import numpy as np
import os
import glob
import keras
import cv2

from dataset_tool import DatasetTool, TaskType

from path_config import PNG_TRAIN_DATASET, DICOM_TRAIN_DATASET
from utils import Utils
import time
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 base_path,
                 dataset_tool,
                 batch_size=8,
                 is_val=False):

        self.is_val = is_val
        self.shuffle = True
        self.dataset_tool = dataset_tool
        self.base_path = base_path
        self.batch_size = batch_size

        self.items_paths = []
        print("Keys in init", self.dataset_tool.class_counts_dict[14])
        for key, value in self.dataset_tool.class_counts_dict.items():
            for img_id_key in value.keys():
                img_name = "small_{}.png".format(img_id_key)
                path = os.path.join(base_path, img_name)
                self.items_paths.append((path, key))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.items_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_items_paths = self.items_paths[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.empty((self.batch_size, 224, 224, 3))
        class_Y = np.zeros((self.batch_size, self.dataset_tool.n_classes))
        # class_Y = np.zeros((self.batch_size, self.dataset_tool.n_classes))

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

            X[ind] = image

            # doesnt work correctly
            # self.dataset_tool.get_label(class_id)

            if self.dataset_tool.task_type == TaskType.BINARY_CLASSIFICATION:
                class_Y[ind] = int(class_id == 14)
            else:
                class_index = list(sorted(self.dataset_tool.class_counts_dict.keys())).index(class_id)
                zeros_arr = np.zeros(3)
                zeros_arr[class_index] = 1
                class_Y[ind] = zeros_arr#self.dataset_tool.get_label(class_id)

            ind += 1

        # print("class_y shape", class_Y.shape)
        # print("rect_Y shape", rect_Y.shape)
        # return X, [rect_Y, class_Y]

        return X, class_Y

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.items_paths)


    def getYForID(self, id):
        for index, row in self.pandas_data_frame.iterrows():
            if row.image_id == id:
                class_id = row.class_id
                if class_id == 14: # no findings
                    return [[self.get_average_random_frame()], class_id]

                x, y, width, height = row.x_min, row.y_min, row.x_max - row.x_min, row.y_max - row.y_min
                return [x, y, width, height], class_id

DICOM_PATH = DICOM_TRAIN_DATASET
lungs_train_2000_PATH = PNG_TRAIN_DATASET

def create_dg():
    is_val = False
    path = os.path.join(lungs_train_2000_PATH, "train")#os.path.join(BASE_PATH, r"dicom_train")

    task_type = TaskType.BINARY_CLASSIFICATION
    dataset_tool = DatasetTool(path, task_type, is_val)

    data_generator = DataGenerator(path, dataset_tool,  is_val=is_val)

    return data_generator

def test_show_image():
    data_generator = create_dg()

    start_time = time.time()
    items = data_generator[0]
    print(items[1].shape)
    print(items[1])
    print("Item generation time:", time.time() - start_time)

    cv2.imshow("batch_image", items[0][1])
    cv2.waitKey(0)

def test_class_dict():
    data_generator = create_dg()

    print("\n \n \n data generator len", len(data_generator))
    for key, value in data_generator.dataset_tool.class_counts_dict.items():
        print("key {}, count {}".format(key, len(value)))

if __name__ == '__main__':
    test_show_image()
    # test_class_dict()

