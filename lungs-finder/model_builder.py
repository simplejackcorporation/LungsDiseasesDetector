import keras
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape

from path_config import TaskType

class ModelBuilder:
    def __init__(self, task_type, n_classes=None):

        if task_type == TaskType.BINARY_CLASSIFICATION:
            self.model_head = Dense(1, activation='sigmoid')
            self.loss = keras.losses.binary_crossentropy

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.model_head = Dense(n_classes, activation='softmax')
            self.loss = keras.losses.categorical_crossentropy


    def model(self):
        # MODEL
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
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

        # class_x = Reshape((5,  4))(class_x)
        #
        # rect_x = Dense((5 * 4), activation='relu')(x)
        # rect_x = Reshape((5,  4))(rect_x)

        # rects_predictions = Dense(4, activation='linear', name="rect_output")(rect_x)
        class_predicions = self.model_head(x)

        model = keras.Model(inputs=base_model.input, outputs=class_predicions)
        #         model = keras.Model(inputs=base_model.input, outputs=[rects_predictions, class_predicions])
        return model

    def custom_loss(y_true, y_pred):
        print("custom loss")
        print("y_pred ", y_pred)

        y_pred = K.print_tensor(y_pred)
        return y_true - y_pred

if __name__ == '__main__':
    n_class = 3
    model_builder = ModelBuilder(TaskType.MULTICLASS_CLASSIFICATION, n_class)
    model = model_builder.model()
    model.summary()
    print(model.output.shape)
