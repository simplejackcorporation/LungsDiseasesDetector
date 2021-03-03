import tensorflow as tf
import keras
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape, Activation, Input
from path_config import TaskType, N_PROPOSALS, BATCH_SIZE, CELL_SIDE, CELLS_COUNT
from keras.losses import categorical_crossentropy, binary_crossentropy, mse


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

def get_rects_and_class_tensors2(x):
    print("get_rects_and_class_tensors2")
    print("x shape", x.shape)

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

class YoloLikeModel():

    def __init__(self):
        newInput = Input(batch_shape=(BATCH_SIZE, 224, 224, 3))  # let us say this new InputLayer
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            # alpha=1.0,
            include_top=False,
            weights="imagenet",
        )
        # base_model.summary()
        model = base_model(newInput)
        newModel = keras.Model(newInput, model)

        x = newModel.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)

        output_len = (CELL_SIDE ** 2) * (CELLS_COUNT ** 2)
        x = Dense(output_len, activation="relu")(x)

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
        model = keras.Model(newInput, conc_l)
        print("VOVA")
        self.model = model