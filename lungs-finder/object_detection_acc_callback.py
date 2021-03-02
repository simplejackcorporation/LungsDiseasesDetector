
import keras
import os
from model_builder import ModelBuilder
from path_config import TaskType, PNG_TRAIN_DATASET, BATCH_SIZE
from dataset_tool import DatasetTool
from data_generator import DataGenerator
from utils import Utils

class ObjectDetectionAccCallback(keras.callbacks.Callback):

    def __init__(self, data_generator):
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs=None):
        test_data = []

        for item in self.data_generator:
            test_data.append(item)

        result_dict = {}

        for item in test_data:
            x_result = self.model.predict(item[0], verbose=0)
            labels = item[1]
            for index, pred in enumerate(x_result):
                rects_preds = pred[:, :4]
                is_background_preds = pred[:, 4:5]
                class_preds = pred[:, 5:]

                current_labels = labels[index]
                rects_labels = current_labels[:, :4]
                is_background_labels = current_labels[:, 4:5]
                class_labels = current_labels[:, 5:]

                ious = []
                for index, rect_pred in enumerate(rects_preds):
                    rect_label = rects_labels[index]
                    iou = Utils.iou(rect_label, rect_pred)
                    ious.append(iou)
        ious_mean = sum(ious) / len(ious)
        print("IOU's mean : {} at epoch {}".format(ious_mean, epoch))




if __name__ == '__main__':
    task_type = TaskType.OBJECT_DETECTION
    path = os.path.join(PNG_TRAIN_DATASET, "train")

    val_dataset_tool = DatasetTool(path, task_type, True)

    n_classes = val_dataset_tool.n_classes
    validation_generator = DataGenerator(path, val_dataset_tool, batch_size=BATCH_SIZE, is_val=True)

    model_builder = ModelBuilder(task_type,
                                 n_classes)

    callback = ObjectDetectionAccCallback(validation_generator)
    callback.model = model_builder.model()
    callback.on_epoch_end(epoch=2)
