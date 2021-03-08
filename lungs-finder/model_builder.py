import keras

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape, Activation, Input
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
from path_config import TaskType, N_PROPOSALS, BATCH_SIZE, CELL_SIDE, CELLS_COUNT
from yolo_like_model import YoloLikeModel, get_rects_and_class_tensors, get_rects_and_class_tensors2

CELL_AREA = CELL_SIDE ** 2

def mseloss(y_true, y_pred):
    print("\n MSE LOSS")
    print("y_true", y_true)
    print("y_pred", y_pred)

    true_rects_res, _, _ = get_rects_and_class_tensors2(y_true)
    pred_rects_res, _, _ = get_rects_and_class_tensors2(y_pred)
    print("true_rects_res", true_rects_res.shape)
    print("pred_rects_res", pred_rects_res.shape)

    coordinates = tf.keras.activations.tanh(pred_rects_res[:, :, 0:2])

    coordinates = tf.math.add(coordinates, CELL_SIDE)
    sides = tf.math.scalar_mul(CELL_SIDE, tf.exp(pred_rects_res[:, :, 2:4]))
    pred_concated_res = tf.concat([coordinates,
                                  sides], axis=2)

    return mse(true_rects_res, pred_concated_res)


def backgroundloss(y_true, y_pred):
    print("backgroundloss")
    _, true_is_background_res, _ = get_rects_and_class_tensors(y_true)
    _, pred_is_background_res, _ = get_rects_and_class_tensors(y_pred)

    return binary_crossentropy(true_is_background_res, pred_is_background_res)


def classloss(y_true, y_pred):
    print("classloss")

    _, _, true_one_hot_class_res = get_rects_and_class_tensors(y_true)
    _, _, pred_one_hot_class_res = get_rects_and_class_tensors(y_pred)

    return categorical_crossentropy(true_one_hot_class_res, pred_one_hot_class_res)

class ModelBuilder:

    def __init__(self,
                 task_type,
                 n_classes):

        self.task_type = task_type
        self.n_classes = n_classes

        if task_type == TaskType.BINARY_CLASSIFICATION:
            self.loss = binary_crossentropy

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.loss = categorical_crossentropy

        # elif task_type == TaskType.OBJECT_DETECTION:
        #     self.loss = total_object_detection_loss()


    def model_head(self, input):
        model_head = None

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            model_head = Dense(1, activation='sigmoid')(input)

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            model_head = Dense(self.n_classes, activation='softmax')(input)

        elif self.task_type == TaskType.OBJECT_DETECTION:
            shape = 5 + self.n_classes + 1
            print("\n \n shape", shape)
            print("\n \n self.n_classes", self.n_classes)

            model_head = Reshape((N_PROPOSALS, shape), name="RESHAPEEEE")(input)

            model_head = Dense(shape,
                               activation=Activation(object_detection_activation))(model_head)

        return model_head

    def yolo_like_model(self):
        return YoloLikeModel()

    def model(self):
        # MODEL
        base_model = keras.applications.MobileNetV2(
            # input_shape=(224, 224, 3),
            # alpha=1.0,
            include_top=False,
            weights="imagenet",
            # input_tensor=input_tensor,
            # pooling=None,
            # classes=1000,
            # classifier_activation="softmax",
            # **kwargs: For backwards compatibility only.
        )

        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        x = Dense((N_PROPOSALS * (5 + self.n_classes + 1)), activation='relu')(x)

        class_predicions = self.model_head(x)

        model = keras.Model(inputs=base_model.input,
                            outputs=class_predicions)

        return model

    def model2(self):
        # MODEL
        base_model = keras.applications.MobileNetV2(
            # input_shape=(224, 224, 3),
            # alpha=1.0,
            include_top=False,
            weights="imagenet",
            # input_tensor=input_tensor,
            # pooling=None,
            # classes=1000,
            # classifier_activation="softmax",
            # **kwargs: For backwards compatibility only.
        )


        # model = keras.Model(inputs=base_model.input,
        #                     outputs=x)

        return base_model

if __name__ == '__main__':
    n_class = 3
    model_builder = ModelBuilder(TaskType.OBJECT_DETECTION, n_classes=n_class)
    model = model_builder.yolo_like_model()
    model.summary()
    print(model.output.shape)
