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
    print("\n \n activated_rects_res shape", activated_rects_res.shape)
    activated_is_background_res = keras.activations.sigmoid(is_background_res)
    activated_one_hot_class_res = keras.activations.softmax(one_hot_class_res)

    act_concated_res = tf.concat([activated_rects_res,
                                  activated_is_background_res,
                                  activated_one_hot_class_res], axis=2)


    print("\n \n act_concated_res", activated_rects_res.shape)

    return act_concated_res


output_len = (CELL_SIDE ** 2) * (CELLS_COUNT ** 2)

class ImageGridOneCellLayer(keras.layers.Layer):
    def __init__(self, shape):
        super(ImageGridOneCellLayer, self).__init__()
        self.dense1 = Dense((N_PROPOSALS * shape))
        self.dense2 = Reshape((N_PROPOSALS, shape))

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

class ImageGridLayer(keras.layers.Layer):
    @staticmethod
    def slice(input):
        slices = []
        for item_ind in range(0, output_len):
            if item_ind % CELL_AREA == 0:
                slice_x = tf.slice(input, [0, item_ind], [BATCH_SIZE, CELL_AREA])
                slices.append(slice_x)

                # print(item_ind)
                # print(item_ind + CELL_AREA)
                # print("slice x shape", slice_x.shape)

        return slices

    def __init__(self):
        super(ImageGridLayer, self).__init__()
        input_l = tf.Variable(tf.zeros((BATCH_SIZE, output_len)))
        print("input_l shape", input_l.shape)
        slices = ImageGridLayer.slice(input_l)

        self.cells = []
        for index, slice in enumerate(slices):
            n_classes = 3
            shape = 5 + n_classes + 1
            one_cell_l = ImageGridOneCellLayer(shape=shape)
            self.cells.append(one_cell_l)

    def call(self, inputs):
        # input_l = tf.Variable(tf.zeros(BATCH_SIZE, output_len * (CELL_SIDE ** 2)))
        # print("Variable" , input_l.shape)

        slices = ImageGridLayer.slice(inputs)
        print("slices" , slices)

        preds = []
        for index, slice in enumerate(slices):
            pred = self.cells[index](slice)
            # print("PRED", pred)
            preds.append(pred)

        # print("preds", len(preds))
        # print("preds[0].shape", preds[0].shape)
        preds = tf.reshape(preds, (10, 64, 5, 9), name=None)
        # print("reshaped", reshaped)


        return preds

class YoloLikeModel(keras.Model):
    def __init__(self):
        super(YoloLikeModel, self).__init__()

        # self.input_l = Input(batch_shape=(BATCH_SIZE, 224, 224, 3))  # let us say this new InputLayer
        self.base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            # alpha=1.0,
            include_top=False,
            weights="imagenet",
        )
        # base_model.summary()
        # model = base_model(self.input_l)
        # newModel = keras.Model(base_model.input, model)

        self.pooling_l = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense = Dense(1024, activation="relu")

        output_len = (CELL_SIDE ** 2) * (CELLS_COUNT ** 2)
        self.dense2 = Dense(output_len, activation="relu")
        self.image_grid_layer = ImageGridLayer()

        # self.output_l = keras.layers.Concatenate()
        # self.output_l = conc_l
        # self.model = keras.Model(base_model.input, conc_l)
        print("VOVA")
        # self.model = model

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pooling_l(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense2(x)
        x = self.image_grid_layer(x)
        return x#keras.layers.Concatenate()(x)


