import numpy as np
import os
import glob
import keras
import cv2
from utils import Utils
import time

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 base_path,
                 batch_size=16):
        self.base_path = base_path
        self.batch_size = batch_size

        self.items_paths = glob.glob(os.path.join(self.base_path, "*"))
        print("self.base_path", self.base_path)
        print("self.items_paths", self.items_paths)
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.items_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_items_paths = self.items_paths[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.empty((self.batch_size, 224, 224, 3))

        ind = 0
        while ind < self.batch_size:
            item_path = batch_items_paths[ind]
            image = cv2.imread(item_path)
            image = Utils.cropLungsAreaImage(image, item_path)

            if image is None:
                continue

            image = cv2.resize(image, (224, 224))
            image = Utils.normalize(image)

            X[ind] = image

        temp_y = np.empty((self.batch_size, 1))
        return X, temp_y

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)


if __name__ == '__main__':
   data_generator = DataGenerator(base_path=r"C:\Users\m\Desktop\datasets\dicom_train")

   start_time = time.time()
   items = data_generator[0]
   print(items.shape)
   print("Item generation time:", time.time() - start_time)

   batch_image = items[0]
   print("batch_image.shape", batch_image.shape)
   cv2.imshow("batch_image", batch_image)
   cv2.waitKey(0)

