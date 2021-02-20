from data_generator import DataGenerator
from model import Model
import time

if __name__ == '__main__':
    model = Model.model()
    model.summary()

    data_generator = DataGenerator(base_path=r"C:\Users\m\Desktop\datasets\dicom_train")

    start_time = time.time()
    item = data_generator[0]
    predictions = model.predict(item)
    print(predictions)
