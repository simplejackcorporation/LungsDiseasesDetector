import keras
from keras.layers import GlobalAveragePooling2D, Dense

class Model:
    @staticmethod
    def model():
        # MODEL
        base_model = keras.applications.MobileNetV2(
            # input_shape=(224, 224, 3),
            # alpha=1.0,
            include_top=False,
            weights="imagenet",
            # input_tensor=None,
            # pooling=None,
            # classes=1000,
            # classifier_activation="softmax",
            # **kwargs: For backwards compatibility only.
        )

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        preds = Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=base_model.input, outputs=preds)
        return  model
