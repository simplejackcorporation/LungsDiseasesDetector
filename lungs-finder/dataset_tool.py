import numpy as np
from path_config import PNG_TRAIN_DATASET, DICOM_TRAIN_DATASET, TaskType, N_PROPOSALS
from utils import Utils

EMPTY_CLASS_ID = 14

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
            self.equalize_binary_dataset()

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.equalize_multiclass_dataset()

        elif task_type == TaskType.OBJECT_DETECTION:
            self.equalize_multiclass_dataset()

        self.sorted_keys_list = list(sorted(self.class_counts_dict.keys()))
        print("keys", self.sorted_keys_list)
        self.n_classes = len(self.class_counts_dict.keys())
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
        number_positive_img_ids = 0
        for key, item in self.class_counts_dict.items():
            if key != EMPTY_CLASS_ID:
                number_positive_img_ids += len(item)

        neg_class_img_keys = list(sorted(self.class_counts_dict[EMPTY_CLASS_ID]))

        sliced_keys = neg_class_img_keys[0:number_positive_img_ids]
        self.copy_slice_dict(sliced_keys, self.class_counts_dict, EMPTY_CLASS_ID)

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
        # https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
        buff_dict = {}  # sliced_img_objs.copy()

        for img_id_key in sliced_keys:
            buff_dict[img_id_key] = class_counts_dict[class_id][img_id_key]

        class_counts_dict[class_id] = buff_dict

    ### LABEL

    def create_label_placeholder(self, batch_size):
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return np.zeros((batch_size, 1))

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return np.zeros((batch_size, self.n_classes))

        elif self.task_type == TaskType.OBJECT_DETECTION:
            return np.zeros((batch_size, N_PROPOSALS, 4 + 1 + self.n_classes)) # x y w h is_back one_hot length

    # img_id used JUST for object detection (rects retrieval)
    def get_label(self, class_id , img_id=None):
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            return int(class_id == EMPTY_CLASS_ID)

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return self.one_hot_representation(class_id, self.sorted_keys_list)

        elif self.task_type == TaskType.OBJECT_DETECTION:
            zerr_arr = self.one_hot_representation(class_id, self.sorted_keys_list)
            zerr_arr = zerr_arr.tolist()
            rect = self.get_rect(class_id, img_id)
            is_background = class_id == EMPTY_CLASS_ID
            label = rect + [int(is_background)] + zerr_arr
            return label

    def one_hot_representation(self, class_id, sorted_keys_list):
        class_index = sorted_keys_list.index(class_id)
        zerr_arr = np.zeros(self.n_classes)
        zerr_arr[class_index] = 1
        return zerr_arr

    #img id can have multiple class id
    def get_rect(self, class_id, img_id):
        if class_id == EMPTY_CLASS_ID:
            class_id = Utils.choose_random_item(self.sorted_keys_list, EMPTY_CLASS_ID)
            img_id = np.random.choice(list(self.class_counts_dict[class_id].keys()))

        rects = self.class_counts_dict[class_id][img_id]
        choosen_rect = list(rects[0])
        rect = [int(x) for x in choosen_rect]
        return rect




import os
import pandas as pd

if __name__ == '__main__':
    is_val = False

    path = os.path.join(PNG_TRAIN_DATASET, "train")#os.path.join(BASE_PATH, r"dicom_train")
    task_type = TaskType.OBJECT_DETECTION
    dataset_tool = DatasetTool(path, task_type, is_val)
    print("keys", dataset_tool.sorted_keys_list)

    count_sum = 0
    for key, value in dataset_tool.class_counts_dict.items():
        print("KEY", key)
        count_sum += len(value)
        for img_k in list(value.keys()):
            print(dataset_tool.get_label(key, img_k))
        print("key {}, count {}".format(key, len(value)))

    print(count_sum)

    # for key, value in class_counts_dict.items():
    #     print("key {}, count {}".format(key, len(value)))