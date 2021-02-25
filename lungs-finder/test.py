import path_config
from data_generator import DataGenerator
from model import Model
import time

if __name__ == '__main__':
    model = Model.model()
    model.summary()

    data_generator = DataGenerator(base_path=path_config.DICOM_TRAIN_DATASET)

    start_time = time.time()
    item = data_generator[0]
    predictions = model.predict(item)
    print(predictions)
