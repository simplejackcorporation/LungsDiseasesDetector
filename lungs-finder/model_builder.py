import keras

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape, Activation, Input
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
from path_config import TaskType, N_PROPOSALS, BATCH_SIZE, CELL_SIDE, CELLS_COUNT

CELL_AREA = CELL_SIDE ** 2

@tf.function
def get_rects_and_class_tensors(x):
    n_proposals = N_PROPOSALS
    batch_size = tf.shape(x)[0] # needed for slice
    class_count = 3 # :( :( :( :(
    rects_res = tf.slice(x, [0, 0, 0], [batch_size, n_proposals, 4])

    is_background_res = tf.slice(x, [0, 0, 4], [batch_size, n_proposals, 1])

    # last_ind = class_count + 1 # + is background (empt ry class can be removed in future)
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
    print("\n MSE LOSS")

    true_rects_res, _, _ = get_rects_and_class_tensors(y_true)
    pred_rects_res, _, _ = get_rects_and_class_tensors(y_pred)
    print("true_rects_res", true_rects_res.shape)
    print("pred_rects_res", pred_rects_res.shape)

    coordinates = tf.keras.activations.tanh(pred_rects_res[:, :, 0:2])

    coordinates = tf.math.add(coordinates, CELL_SIDE)
    sides = tf.math.scalar_mul(CELL_SIDE, tf.exp(pred_rects_res[:, :, 2:4]))
    pred_concated_res = tf.concat([coordinates,
                                  sides], axis=2)

    return mse(true_rects_res, pred_concated_res)


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

            model_head = Dense(shape,
                               activation=Activation(object_detection_activation))(model_head)

        return model_head

    def yolo_like_model(self):
        newInput = Input(batch_shape=(BATCH_SIZE, 224, 224, 3))  # let us say this new InputLayer
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
        base_model.summary()
        model = base_model(newInput)
        newModel = keras.Model(newInput, model)

        x = newModel.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)

        output_len = (CELL_SIDE ** 2) * (CELLS_COUNT ** 2)
        x = Dense(output_len, activation="relu")(x)

        #x = Conv2D(1, (64, 64))(x)
        # x = YOLOLayer(x)(x)
        denses = []
        for item_ind in range(0, output_len):
            if item_ind % (CELL_SIDE ** 2) == 0:
                start_ind = item_ind + (CELL_SIDE ** 2)
                slice_x = tf.slice(x, [0, start_ind], [BATCH_SIZE, start_ind + CELL_AREA])
                n_classes = 3
                shape = 5 + n_classes + 1

                dense_l = Dense((N_PROPOSALS * shape))(slice_x)
                dense_l = Reshape((N_PROPOSALS, shape))(dense_l)
                dense_l = Dense(shape, activation=Activation(object_detection_activation),
                                name="Dense_{}".format(item_ind))(dense_l)

                denses.append(dense_l)

        print("len(denses)", len(denses))
        conc_l = keras.layers.Concatenate()(denses)
        newModel = keras.Model(newInput, conc_l)

        return newModel

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
    # model.summary()
    print(model.output.shape)
