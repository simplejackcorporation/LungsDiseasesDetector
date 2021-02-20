import os
import keras
import keras.backend as K

from data_generator import DataGenerator
from model import Model


def custom_loss(y_true, y_pred):
    print("custom loss")
    print("y_pred ", y_pred)

    y_pred = K.print_tensor(y_pred)
    return y_true - y_pred


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
    losses = {
        "rect_output": "mse",
        "class_output": "categorical_crossentropy",
    }

    optimizer = keras.optimizers.RMSprop()
    loss = losses # custom_loss

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
                        epochs=1,
                        steps_per_epoch=100,
                        use_multiprocessing=True)

if __name__ == '__main__':
    train()