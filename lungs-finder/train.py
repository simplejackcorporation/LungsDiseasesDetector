import os
import keras

from data_generator import DataGenerator
from model import Model
from utils import Utils


def train():
    #DATASET
    BASE_PATH = r"C:\Users\m\Desktop\LUNGS\lungs-finder"
    EPOCHS = 200

    train_path = os.path.join(BASE_PATH, r"dataset\train")
    validation_path = os.path.join(BASE_PATH, r"dataset\validation")

    training_generator = DataGenerator(train_path)
    validation_generator = DataGenerator(validation_path)

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
    loss = keras.losses.MSE

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
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        use_multiprocessing=True)


if __name__ == '__main__':
    train()