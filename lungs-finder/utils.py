import cv2
import glob
import os
import pydicom
import lungs_finder as lf

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
    # DON"T WORK FOR DATA GENERATOR
    def cropSepareteLungsImages(image):
        right_lung_hog_rectangle = lf.find_right_lung_hog(image)
        left_lung_hog_rectangle = lf.find_left_lung_hog(image)

        right_lung_img = Utils.crop_rect(image, right_lung_hog_rectangle)
        left_lung_img = Utils.crop_rect(image, left_lung_hog_rectangle)

        return [left_lung_img, right_lung_img]

    @staticmethod
    # DON"T WORK FOR DATA GENERATOR
    def cropLungsAreaImage(image):
        r_x, r_y, r_width, r_height = lf.find_left_lung_hog(image)
        l_x, l_y, l_width, l_height = lf.find_right_lung_hog(image)

        crop_height = max(l_height, r_height)
        crop_y = min(l_y, r_y)
        crop_width = r_x - l_x
        crop_x = l_x

        y_delta = 20
        x_delta = 20

        if crop_x - x_delta < 0:
            crop_x = 0
        else:
            crop_x = crop_x - x_delta

        if crop_y - y_delta < 0:
            crop_y = 0
        else:
            crop_y = crop_y - y_delta

        crop_rect = crop_x, crop_y, crop_height + y_delta, crop_width + x_delta

        # print("r_x {} l_x {}".format(r_x, l_x))
        # print("crop_rect", crop_rect)
        crop_img = Utils.crop_rect(image, crop_rect)

        return crop_img

if __name__ == '__main__':
    name = "small_5c23e16f590703222743a8895b41e898.dicom.png"
    path = r"C:\Users\m\Desktop\datasets\dicom_train\{}".format(name)
    image = cv2.imread(path)
    image = Utils.normalize(image)
    image = Utils.cropLungsAreaImage(image)

    cv2.imshow("test", image)
    cv2.waitKey(0)
