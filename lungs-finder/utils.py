import cv2
import pydicom
import lungs_opencv_tools as lf
import numpy as np

from path_config import DICOM_TRAIN_DATASET


class Utils:

    @staticmethod
    def normalize(arr, out_range=(0, 255)):
        # y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        y = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        return y

    @staticmethod
    def crop_rect(image, rectangle):
        x, y, width, height = rectangle
        crop_img = image[y:y + height, x:x + width]
        return crop_img

    @staticmethod
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

    @staticmethod
    def convert_dicom_to_img(path):
        image = pydicom.dcmread(path).pixel_array
        return image

    @staticmethod
    def get_left_rignt_hog_predictions(image):
        right_lung_hog_rectangle = lf.find_right_lung_hog(image)
        left_lung_hog_rectangle = lf.find_left_lung_hog(image)

        if right_lung_hog_rectangle is None and left_lung_hog_rectangle is None:
            return [None, None]

        ### !!!
        ### [ Right lung Left lung ]
        return right_lung_hog_rectangle, left_lung_hog_rectangle


    @staticmethod
    # DOESN"T WORK FOR DATA GENERATOR
    def cropSepareteLungsImages(image):
        right_rect, left_rect = Utils.get_left_rignt_hog_predictions(image)

        right_lung_img = None
        if right_rect is not None:
            right_lung_img = Utils.crop_rect(image, right_rect)

        left_lung_img = None
        if left_rect is not None:
            left_lung_img = Utils.crop_rect(image, left_rect)

        return [right_lung_img, left_lung_img]

    @staticmethod
    # DOESN"T WORK FOR DATA GENERATOR
    def applyDeltas(rect, x_delta, y_delta):
        x, y, width, height = rect

        if x - x_delta < 0:
            x = 0
        else:
            x -= x_delta
            height += y_delta

        if y - y_delta < 0:
            y = 0
        else:
            y -= y_delta
            width += x_delta

        return x, y, width, height

    @staticmethod
    # DOESN"T WORK FOR DATA GENERATOR
    def cropLungsAreaImage(image, item_path=None):
        x_delta = 20

        y_delta = 20

        right_rect, left_rect = Utils.get_left_rignt_hog_predictions(image)

        if right_rect is not None:
            r_rect = Utils.applyDeltas(right_rect, x_delta, y_delta)

            if left_rect is None:
                return Utils.crop_rect(image, r_rect)

        if left_rect is not None:
            l_rect = Utils.applyDeltas(left_rect, x_delta, y_delta)

            if right_rect is None:
                return Utils.crop_rect(image, l_rect)

        r_x, r_y, _, r_height = r_rect
        l_x, l_y, _, l_height = l_rect

        crop_height = max(l_height, r_height)
        crop_y = min(l_y, r_y)
        crop_width = l_x - r_x
        crop_x = l_x

        crop_rect = Utils.applyDeltas((crop_x, crop_y, crop_width, crop_height), x_delta, y_delta)
        crop_img = Utils.crop_rect(image, crop_rect)

        return crop_img

    @staticmethod
    def isNaN(num):
        return num != num

    @staticmethod
    def get_class_count_dict(p_data_frame):
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

    @staticmethod
    def choose_random_item(arr, except_item):
        choosen_item = except_item

        while choosen_item == except_item:
            choosen_item = np.random.choice(arr)

        return choosen_item

    @staticmethod
    def get_intersection_area(t_x, t_y, t_w, t_h,
                              p_x, p_y, p_w, p_h):
        inter_width = min(t_x + t_w, p_x + p_w) - max(t_x, p_x)
        inter_height = min(t_y + t_h, p_y + p_h) - max(t_y, p_y)
        return max(0, inter_height) * max(0, inter_width)

    @staticmethod
    def iou(true_box, predicted_box):
        """
        Receives boxes in format (x,y, width, height) and return IoU
        """

        intersection_area = Utils.get_intersection_area(*true_box,
                                                        *predicted_box)

        def area(box): return box[2] * box[3]

        return intersection_area / (area(true_box) + area(predicted_box) - intersection_area)


def IOU_test():
    box1 = [20, 25, 35, 10]
    box2 = [20, 25, 30, 10]

    iou = Utils.iou(box1, box2)
    print(iou)


def test_CropSepareteLungsImages():
    name = "small_23f29659e174d2c4651857bf304a5d75.dicom.png"
    path = DICOM_TRAIN_DATASET + "\\" + name
    image = cv2.imread(path)
    image = Utils.normalize(image)
    # image = Utils.cropLungsAreaImage(image)
    left, right = Utils.cropSepareteLungsImages(image)

    if left is not None:
        cv2.imshow("test", left)

    if right is not None:
        cv2.imshow("test", right)

    cv2.waitKey(0)


if __name__ == '__main__':
    IOU_test()
