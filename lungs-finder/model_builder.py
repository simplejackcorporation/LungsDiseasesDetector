import keras
import keras.backend as K

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape, Activation
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
from path_config import TaskType, N_PROPOSALS
from keras.utils.generic_utils import get_custom_objects

@tf.function
def get_rects_and_class_tensors(x):
    n_proposals = N_PROPOSALS
    batch_size = tf.shape(x)[0] # needed for slice
    class_count = 3 # :( :( :( :(
    rects_res = tf.slice(x, [0, 0, 0], [batch_size, n_proposals, 4])

    is_background_res = tf.slice(x, [0, 0, 4], [batch_size, n_proposals, 1])

    # last_ind = class_count + 1 # + is background (emptry class can be removed in future)
    one_hot_class_res = tf.slice(x, [0, 0, 5], [batch_size, n_proposals, class_count])

    return rects_res, is_background_res, one_hot_class_res

# def object_detection_loss(y_true, y_pred):
#     true_rects_res, true_is_background_res, true_one_hot_class_res = get_rects_and_class_tensors(y_true)
#     pred_rects_res, pred_is_background_res, pred_one_hot_class_res = get_rects_and_class_tensors(y_pred)
#
#     MSE = mse(true_rects_res, pred_rects_res)
#     background_binary_entropy = binary_crossentropy(true_is_background_res, pred_is_background_res)
#     c_crossentropy = categorical_crossentropy(true_one_hot_class_res, pred_one_hot_class_res)
#
#     #
#     backround_coeff = 1.1 # a bit more imporatant to detect background/non-background
#     total_loss = MSE + c_crossentropy + backround_coeff * background_binary_entropy
#
#     return total_loss


@tf.function
def object_detection_activation(x):
    rects_res, is_background_res, one_hot_class_res = get_rects_and_class_tensors(x)

    activated_rects_res = keras.activations.relu(rects_res)
    activated_is_background_res = keras.activations.sigmoid(is_background_res)
    activated_one_hot_class_res = keras.activations.softmax(one_hot_class_res)

    act_concated_res = tf.concat([activated_rects_res,
                                  activated_is_background_res,
                                  activated_one_hot_class_res], axis=2)
    return act_concated_res


def mseloss(y_true, y_pred):
    true_rects_res, _, _ = get_rects_and_class_tensors(y_true)
    pred_rects_res, _, _ = get_rects_and_class_tensors(y_pred)

    return mse(true_rects_res, pred_rects_res)

def backgroundloss(y_true, y_pred):
    _, true_is_background_res, _ = get_rects_and_class_tensors(y_true)
    _, pred_is_background_res, _ = get_rects_and_class_tensors(y_pred)

    return binary_crossentropy(true_is_background_res, pred_is_background_res)

def classloss(y_true, y_pred):
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
        #     self.loss = object_detection_loss


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

            model_head = Dense(shape,activation=Activation(object_detection_activation))(model_head)

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

        x = Dense((N_PROPOSALS * (5 + self.n_classes + 1)), activation='relu')(x)

        class_predicions = self.model_head(x)

        model = keras.Model(inputs=base_model.input,
                            outputs=class_predicions)

        return model

if __name__ == '__main__':
    n_class = 3
    model_builder = ModelBuilder(TaskType.OBJECT_DETECTION, n_classes=n_class)
    model = model_builder.model()
    # model.summary()
    print(model.output.shape)
