import numpy as np
from path_config import PNG_TRAIN_DATASET, DICOM_TRAIN_DATASET, TaskType

class DatasetTool:
    def __init__(self,
                 path,
                 task_type,
                 is_val):

        pandas_data_frame = pd.read_csv(os.path.join(path, "train.csv"))
        pandas_data_frame = pandas_data_frame.drop_duplicates(subset=['image_id'])

        self.class_counts_dict = self.get_class_count_dict(pandas_data_frame)
        self.task_type = task_type

        if task_type == TaskType.BINARY_CLASSIFICATION:
            print("\n \n \n DatasetTool binary")
            self.equalize_binary_dataset()

            ### TEMPORARY (SOURCE OF FUTURE BUGS)
            self.n_classes = 1
            ### !

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.equalize_multiclass_dataset()
            self.n_classes = len(self.class_counts_dict.keys())
            self.sorted_keys_list = list(sorted(self.class_counts_dict.keys()))

        self.divide_train_or_val(is_val)


    def get_class_count_dict(self, p_data_frame):
        class_count_dict = {}
        for ind, row in p_data_frame.iterrows():
            class_id = row.class_id
            image_id = row.image_id

            rect = row.x_min, row.y_min, row.x_max - row.x_min, row.y_max - row.y_min

            if class_id not in class_count_dict:
                class_count_dict[class_id] = {image_id: [rect]}

            else:
                if image_id not in class_count_dict[class_id]:
                    class_count_dict[class_id][image_id] = [rect]
                else:
                    rects_array = class_count_dict[class_id][image_id]

                    rects_array.append(rect)
                    class_count_dict[class_id][image_id] = rects_array

        return class_count_dict


    def equalize_binary_dataset(self):
        negative_class_id = 14

        number_positive_img_ids = 0
        for key, item in self.class_counts_dict.items():
            if key != negative_class_id:
                number_positive_img_ids += len(item)

        neg_class_img_keys = list(sorted(self.class_counts_dict[negative_class_id]))

        sliced_keys = neg_class_img_keys[0:number_positive_img_ids]
        self.copy_slice_dict(sliced_keys, self.class_counts_dict, negative_class_id)

    def equalize_multiclass_dataset(self):
        MIN_LIMIT = 100
        min_value = 0
        for class_key, item in self.class_counts_dict.items():
            len_item = len(item)

            if len_item < MIN_LIMIT:
                continue

            if min_value == 0:
                min_value = len_item

            if len_item < min_value:
                min_value = len_item

        buff_dict = self.class_counts_dict.copy()
        for class_key, item in self.class_counts_dict.items():
            len_item = len(item)

            if len_item < MIN_LIMIT:
                buff_dict.pop(class_key, None)
                continue

            class_img_keys = list(sorted(buff_dict[class_key]))
            sliced_keys = class_img_keys[0:min_value]
            self.copy_slice_dict(sliced_keys, buff_dict, class_key)

        self.class_counts_dict = buff_dict

    def divide_train_or_val(self, is_val, ratio=0.7):
        for key, item in self.class_counts_dict.items():
            img_ids_keys = list(sorted(item.keys()))

            start_val_index = int(len(img_ids_keys) * ratio)

            if is_val:
                sliced_keys = img_ids_keys[start_val_index:len(img_ids_keys)]
            else:
                sliced_keys = img_ids_keys[0:start_val_index]

            self.copy_slice_dict(sliced_keys, self.class_counts_dict, key)

    def copy_slice_dict(self, sliced_keys, class_counts_dict, class_id):
        buff_dict = {}  # sliced_img_objs.copy()

        for img_id_key in sliced_keys:
            buff_dict[img_id_key] = class_counts_dict[class_id][img_id_key]

        class_counts_dict[class_id] = buff_dict

    ### LABEL

    def create_label_placeholder(self, batch_size):
        if self.task_type == TaskType.BINARY_CLASSIFICATION or self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return np.zeros((batch_size, self.n_classes))

    def get_label(self, class_id):
        if self.task_type == TaskType.BINARY_CLASSIFICATION:

            ### temporary
            neg_class = 14
            ###

            return int(class_id == neg_class)

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            class_index = self.sorted_keys_list.index(class_id)
            zeros_arr = np.zeros(self.n_classes)
            zeros_arr[class_index] = 1
            return zeros_arr

        def getYForID(self, id):
            for index, row in self.pandas_data_frame.iterrows():
                if row.image_id == id:
                    class_id = row.class_id
                    if class_id == 14:  # no findings
                        return [[self.get_average_random_frame()], class_id]

                    x, y, width, height = row.x_min, row.y_min, row.x_max - row.x_min, row.y_max - row.y_min
                    return [x, y, width, height], class_id

import os
import pandas as pd

if __name__ == '__main__':
    is_val = False

    path = os.path.join(PNG_TRAIN_DATASET, "train")#os.path.join(BASE_PATH, r"dicom_train")
    task_type = TaskType.BINARY_CLASSIFICATION
    dataset_tool = DatasetTool(path, task_type, is_val)
    count_sum = 0
    for key, value in dataset_tool.class_counts_dict.items():
        count_sum += len(value)
        print("key {}, count {}".format(key, len(value)))

    print(count_sum)

    # for key, value in class_counts_dict.items():
    #     print("key {}, count {}".format(key, len(value)))