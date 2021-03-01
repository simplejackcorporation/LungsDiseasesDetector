import keras
import keras.backend as K

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape, Activation
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
from path_config import TaskType, N_PROPOSALS
from keras.utils.generic_utils import get_custom_objects

def get_rects_and_class_tensors(x):
    n_proposals = N_PROPOSALS
    batch_size = tf.shape(x)[0] # needed for slice
    class_count = 3 # :( :( :( :(
    rects_res = tf.slice(x, [0, 0, 0], [batch_size, n_proposals, 4])
    class_res = tf.slice(x, [0, 0, 4], [batch_size, n_proposals, class_count])
    return rects_res, class_res


def object_detection_activation(x):
    rects_res, class_res = get_rects_and_class_tensors(x)

    activated_rects_res = keras.activations.linear(rects_res)
    activated_class_res = keras.activations.softmax(class_res)

    act_concated_res = tf.concat([activated_rects_res, activated_class_res], axis=2)
    return act_concated_res

def object_detection_loss(y_true, y_pred):
    true_rects_res, true_class_res = get_rects_and_class_tensors(y_true)
    pred_rects_res, pred_class_res = get_rects_and_class_tensors(y_pred)

    MSE = mse(true_rects_res, pred_rects_res)

    c_crossentropy = categorical_crossentropy(true_class_res, pred_class_res)
    #
    total_loss = MSE + c_crossentropy

    return total_loss


class ModelBuilder:

    def __init__(self,
                 task_type,
                 n_classes = 3):

        print("\n \n ", task_type)
        self.task_type = task_type
        self.n_classes = n_classes

        if task_type == TaskType.BINARY_CLASSIFICATION:
            self.loss = binary_crossentropy

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.loss = categorical_crossentropy

        elif task_type == TaskType.OBJECT_DETECTION:
            self.loss = object_detection_loss


    def model_head(self, input):
        model_head = None

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            model_head = Dense(1, activation='sigmoid')(input)

        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            model_head = Dense(self.n_classes, activation='softmax')(input)

        elif self.task_type == TaskType.OBJECT_DETECTION:
            print("here")
            model_head = Reshape((N_PROPOSALS, 5 + self.n_classes), name="RESHAPEEEE")(input)
            model_head = Dense(5 + self.n_classes, activation=Activation(object_detection_activation))(model_head)

        return model_head

    def model(self):
        # MODEL
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
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

        # x = Dense(25, activation='relu')(x)

        x = Dense(40, activation='relu')(x)

        # class_x = Reshape((5,  4))(class_x)
        #
        # rect_x = Dense((5 * 4), activation='relu')(x)
        # rect_x = Reshape((5,  4))(rect_x)

        # rects_predictions = Dense(4, activation='linear', name="rect_output")(rect_x)
        class_predicions = self.model_head(x)

        model = keras.Model(inputs=base_model.input,
                            outputs=class_predicions)
        #         model = keras.Model(inputs=base_model.input, outputs=[rects_predictions, class_predicions])
        model.summary()

        return model

if __name__ == '__main__':
    n_class = 3
    model_builder = ModelBuilder(TaskType.OBJECT_DETECTION)
    model = model_builder.model()
    # model.summary()
    print(model.output.shape)
