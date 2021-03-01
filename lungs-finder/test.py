import numpy as np
import os
import glob
import keras
import cv2

from dataset_tool import DatasetTool, TaskType

from path_config import PNG_TRAIN_DATASET, DICOM_TRAIN_DATASET
from utils import Utils
import time
from model_builder import ModelBuilder
from data_generator import DataGenerator

if __name__ == '__main__':
    is_val = False
    path = os.path.join(PNG_TRAIN_DATASET, "train")#os.path.join(BASE_PATH, r"dicom_train")

    task_type = TaskType.OBJECT_DETECTION
    dataset_tool = DatasetTool(path, task_type, is_val)

    data_generator = DataGenerator(path, dataset_tool,
                                   is_val=is_val)

    model_builder = ModelBuilder(task_type)
    model = model_builder.model()
    start_time = time.time()
    item = data_generator[0]
    predictions = model.predict(item)
    print(predictions.shape)
    print(item[1].shape)
