import os
import keras

from data_generator import DataGenerator
from model import Model
from utils import Utils

def custom_loss(y_true , y_pred):
    # y_true,
    # y_pred x, y, w, h
    # loss = keras.losses.MSE

    print("\n\n y_true", y_true)

    print("\ny_pred", y_pred)



def train():
    #DATASET
    BASE_PATH = r"C:\Users\m\Desktop\datasets"
    EPOCHS = 10

    train_path = os.path.join(BASE_PATH, r"dicom_train")
    # validation_path = os.path.join(BASE_PATH, r"dataset\validation")

    training_generator = DataGenerator(train_path)
    # validation_generator = DataGenerator(validation_path)

    #MODEL
    model = Model.model()

    for index, layer in enumerate(model.layers):
        percentage_of_retrain = 0.2 # max is 1 (all layers) last layer is 0
        if index < int(len(model.layers) * (1 - percentage_of_retrain)):
            layer.trainable = False
        else:
            layer.trainable = True

    model.summary()
    print("len(model.layers) :", len(model.layers))

    #COMPILE
    optimizer = keras.optimizers.RMSprop()
    loss = keras.losses.MSE # custom_loss

    model.compile(
        optimizer=optimizer,
        loss=loss,
        # metrics=None,
        # loss_weights=None,
        # weighted_metrics=None,
        # run_eagerly=None,
        # steps_per_execution=None,
    # **kwargs	Arguments supported for backwards compatibility only.
    )

    #TRAIN
    model.fit_generator(generator=training_generator,
                        # validation_data=validation_generator,
                        epochs=EPOCHS,
                        use_multiprocessing=True)

if __name__ == '__main__':
    train()