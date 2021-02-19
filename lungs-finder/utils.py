import cv2
import glob
import os
import pydicom
import lungs_finder as lf

class Utils:

    @staticmethod
    def scale(arr, out_range=(0, 255)):
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
    # DON"T WORK FOR DATA GENERATOR
    def getIMAGE_DICOM(dicom_item_path):
        image = pydicom.dcmread(dicom_item_path).pixel_array

        image = Utils.scale(image)

        image = Utils.proportional_resize(image, 512)

        right_lung_hog_rectangle = lf.find_right_lung_hog(image)
        right_lung_img = Utils.crop_rect(image, right_lung_hog_rectangle)

        left_lung_hog_rectangle = lf.find_left_lung_hog(image)
        left_lung_img = Utils.crop_rect(image, left_lung_hog_rectangle)

        return [left_lung_img, right_lung_img]