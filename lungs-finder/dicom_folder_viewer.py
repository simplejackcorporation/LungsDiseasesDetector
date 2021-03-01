
import zipfile
import cv2
import pandas as pd

from data_generator import DataGenerator

left_window_name = "Left Lungs"
right_window_name = "Right Lungs"


def show_in_place(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)


def draw_left_and_right(left_lung, right_lung):
    if right_lung is not None:
        show_in_place(right_window_name, right_lung, 40, 30)
    if left_lung is not None:
        show_in_place(left_window_name, left_lung, 40 + (len(right_lung) if right_lung is not None else 200), 30)


path_to_zip = 'E:/Любомир/Datasets/X-Rays/vinbigdata-chest-xray-abnormalities-detection.zip'

if __name__ == "__main__":
    archive = zipfile.ZipFile(path_to_zip, 'r')
    df = pd.read_csv(archive.open(f'train.csv'))

    list_id = [image_id + ".dicom" for image_id in df.image_id]

    data_generator = DataGenerator(
        base_path=path_to_zip,
        list_IDs=list_id[:50],
        is_zip_file=True)

    items = data_generator[0]

    i = 0
    draw_left_and_right(*items[i])

    while True:
        k = cv2.waitKeyEx(0)
        if k == 2424832:  # Left arrow
            i -= 1
            i = abs(i % len(items))
            cv2.destroyAllWindows()
            draw_left_and_right(*items[i])
        elif k == 2555904:  # Right arrow
            i += 1
            i = abs(i % len(items))
            cv2.destroyAllWindows()
            draw_left_and_right(*items[i])
        else:
            cv2.destroyAllWindows()
