# your class labels
import keras
import numpy as np
import os
from model_builder import ModelBuilder
from path_config import TaskType, PNG_TRAIN_DATASET, BATCH_SIZE
from dataset_tool import DatasetTool
from data_generator import DataGenerator

class AccuracyCallback(keras.callbacks.Callback):

    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.task_type = self.data_generator.dataset_tool.task_type
        self.n_classses = self.data_generator.dataset_tool.n_classes

    def on_epoch_end(self, epoch, logs=None):
        test_data = []

        for item in self.data_generator:
            test_data.append(item)


        result_dict = {}

        for item in test_data:
            x_result = self.model.predict(item[0], verbose=0)
            labels = item[1]
            for index, pred in enumerate(x_result):
                if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                    unwraped_pred = int(np.argmax(pred))
                    # print(unwraped_pred)
                    label = int(np.argmax(labels[index]))

                # elif self.task_type == TaskType.OBJECT_DETECTION:
                #     print("PREDSS")
                #     print(pred)
                #     unwraped_pred = int(np.argmax(pred))
                #     # print(unwraped_pred)
                #     label = int(np.argmax(labels[index]))

                if label not in result_dict:
                    result_dict[label] = {"correct": 0,
                                          "incorrect": 0,
                                          "confuse_class": []}

                if unwraped_pred == label:
                    result_dict[label]["correct"] += 1
                else:
                    result_dict[label]["incorrect"] += 1
                    result_dict[label]["confuse_class"].append(unwraped_pred)

        for key, value in result_dict.items():
            correct_count = value["correct"]
            incorrect_count = value["incorrect"]

            acc = value["correct"] / (value["incorrect"] + value["correct"])
            print("Accuracy {}, correct count {}, incorrect count {}, of class {}".format(acc,
                                                                                          correct_count,
                                                                                          incorrect_count,
                                                                                          key))

if __name__ == '__main__':
    task_type = TaskType.OBJECT_DETECTION
    path = os.path.join(PNG_TRAIN_DATASET, "train")

    val_dataset_tool = DatasetTool(path, task_type, True)

    n_classes = val_dataset_tool.n_classes
    validation_generator = DataGenerator(path, val_dataset_tool, batch_size=BATCH_SIZE, is_val=True)

    model_builder = ModelBuilder(task_type,
                                 n_classes)

    callback = AccuracyCallback(validation_generator)
    callback.model = model_builder.model()
    callback.on_epoch_end(epoch=2)
