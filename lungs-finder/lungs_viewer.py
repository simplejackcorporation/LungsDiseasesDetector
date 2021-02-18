import os
import sys
import numpy as np
import cv2
import lungs_finder as lf

from skimage import exposure
from sklearn import preprocessing
from PIL import Image

import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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

import sys

def scale(arr, out_range=(0, 255)):
    # y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    y = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')

    print(y)
    return y

def scan(argv):
    np.set_printoptions(threshold=sys.maxsize)

    use_labels = True
    #
    health_image_id = r"C:\Users\m\Desktop\LUNGS\lungs-finder\Example\0e40d75b6a0f50c6855b6f3203f421a2.dicom"
    image = pydicom.dcmread(f"{health_image_id}").pixel_array
    # print(type(image))
    # print(image.shape)
    # image = image.copy()

    # image = np.load(health_image_id, allow_pickle=True)
    # image = exposure.equalize_adapthist(image)
    # image = Image.fromarray(image.astype('uint8'), 'GRAY')
    # image = preprocessing.normalize(image)

    # image = cv2.resize(image, (783, 974))
    # image = image.astype(np.uint8)
    # image = cv2.imread(r"C:\Users\m\Desktop\LUNGS\lungs-finder\Example\Picture1.png", 0)
    # image = preprocessing.normalize(image)
    image = scale(image)

    print(image[20: 400])

    # cv2.imshow("Found lungs", image)
    # #
    # cv2.waitKey(0)

    # image = cv2.equalizeHist(image)
    # image = cv2.medianBlur(image, 3)

    scaled_image = proportional_resize(image, 512)
    right_lung_hog_rectangle = lf.find_right_lung_hog(scaled_image)
    left_lung_hog_rectangle = lf.find_left_lung_hog(scaled_image)
    right_lung_lbp_rectangle = lf.find_right_lung_lbp(scaled_image)
    left_lung_lbp_rectangle = lf.find_left_lung_lbp(scaled_image)
    right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
    left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)
    color_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

    if right_lung_hog_rectangle is not None:
        x, y, width, height = right_lung_hog_rectangle
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (59, 254, 211), 2)
        if use_labels:
            cv2.putText(color_image, "HOG Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (59, 254, 211), 1)

    if left_lung_hog_rectangle is not None:
        x, y, width, height = left_lung_hog_rectangle
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (59, 254, 211), 2)
        if use_labels:
            cv2.putText(color_image, "HOG Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (59, 254, 211), 1)

    if right_lung_lbp_rectangle is not None:
        x, y, width, height = right_lung_lbp_rectangle
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (130, 199, 0), 2)
        if use_labels:
            cv2.putText(color_image, "LBP Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (130, 199, 0), 1)

    if left_lung_lbp_rectangle is not None:
        x, y, width, height = left_lung_lbp_rectangle
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (130, 199, 0), 2)
        if use_labels:
            cv2.putText(color_image, "LBP Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (130, 199, 0), 1)

    if right_lung_haar_rectangle is not None:
        x, y, width, height = right_lung_haar_rectangle
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (245, 199, 75), 2)
        if use_labels:
            cv2.putText(color_image, "HAAR Right", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (245, 199, 75), 1)

    if left_lung_haar_rectangle is not None:
        x, y, width, height = left_lung_haar_rectangle
        cv2.rectangle(color_image, (x, y), (x + width, y + height), (245, 199, 75), 2)
        if use_labels:
            cv2.putText(color_image, "HAAR Left", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (245, 199, 75), 1)

    cv2.imshow("lungs-finder", color_image)
    found_lungs = lf.get_lungs(scaled_image)

    if found_lungs is not None:
        cv2.imshow("Found lungs", found_lungs)

    cv2.waitKey(0)


if __name__ == "__main__":
    sys.exit(scan(sys.argv))
