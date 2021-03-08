import numpy as np
import os
import glob
import keras
import cv2

from dataset_tool import DatasetTool, TaskType

from path_config import PNG_TRAIN_DATASET, DICOM_TRAIN_DATASET, BATCH_SIZE
from utils import Utils
import time
from model_builder import ModelBuilder
from data_generator import DataGenerator

if __name__ == '__main__':
    is_val = False
    path = os.path.join(PNG_TRAIN_DATASET, "train")#os.path.join(BASE_PATH, r"dicom_train")

    task_type = TaskType.OBJECT_DETECTION
    dataset_tool = DatasetTool(path, task_type, is_val)

    data_generator = DataGenerator(path,
                                       dataset_tool,
                                       is_val=False,
                                       batch_size=BATCH_SIZE)

    model_builder = ModelBuilder(task_type, n_classes=dataset_tool.n_classes)
    model = model_builder.yolo_like_model()
    # model.build(input_shape = (BATCH_SIZE, 224, 224, 3 ))
    # model.summary()
    # print("input shape", model.input.shape)
    # print("output shape", model.output.shape)

    start_time = time.time()
    item = data_generator[0]
    print("item shape", item[0].shape)

    predictions = model.predict(item[0])
    print(predictions.shape)
    print(item[1].shape)
