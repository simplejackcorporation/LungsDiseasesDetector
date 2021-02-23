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
    EPOCHS = 10
    lungs_train_2000_PATH = r"C:\Users\m\Desktop\datasets\lungs_train_2000"

    train_path = os.path.join(lungs_train_2000_PATH, "train")#os.path.join(BASE_PATH, r"dicom_train")
    training_generator = DataGenerator(base_path=train_path, is_val=False)
    validation_generator = DataGenerator(train_path, is_val=True)

    #MODEL
    model = Model.model()

    for index, layer in enumerate(model.layers):
        last_train_layer = int(len(model.layers) - 5)

        if index < last_train_layer:
            layer.trainable = False
        else:
            layer.trainable = True

    model.summary()
    print("len(model.layers) :", len(model.layers))

    #COMPILE
    losses = {
        "rect_output": "mse",
        "class_output": "sparse_categorical_crossentropy",
    }

    optimizer = keras.optimizers.RMSprop()
    loss = keras.losses.binary_crossentropy  # losses # custom_loss #
    # loss = custom_loss

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],

        # loss_weights=None,
        # weighted_metrics=None,
        # run_eagerly=None,
        # steps_per_execution=None,
    # **kwargs	Arguments supported for backwards compatibility only.
    )

    checkpoint_filepath = r'C:\Users\m\Desktop\LUNGS\lungs-finder\weights'

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    #TRAIN
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=10,
                        callbacks=[model_checkpoint_callback],
                        use_multiprocessing=True)

if __name__ == '__main__':
    train()