import os
import keras

from data_generator import DataGenerator
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
    base_model = keras.applications.MobileNet(
        # input_shape=(224, 224, 3),
        # alpha=1.0,
        include_top=False,
        weights="imagenet",
        # input_tensor=None,
        # pooling=None,
        # classes=1000,
        # classifier_activation="softmax",
        #**kwargs: For backwards compatibility only.
    )


    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x = keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
    x = keras.layers.Dense(512,activation='relu')(x) #dense layer 3
    preds = keras.layers.Dense(1, activation='sigmoid')(x) #final layer with softmax activation
    model = keras.Model(inputs=base_model.input,outputs=preds)

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