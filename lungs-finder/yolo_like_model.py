import tensorflow as tf
import keras
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Reshape, Activation, Input
from path_config import TaskType, N_PROPOSALS, BATCH_SIZE, CELL_SIDE, CELLS_COUNT
from keras.losses import categorical_crossentropy, binary_crossentropy, mse
import numpy as np

CELL_AREA = CELL_SIDE ** 2
output_len = (CELL_SIDE ** 2) * (CELLS_COUNT ** 2)


@tf.function
# x should be 10 batch size on 5 x 8 (batch size x n_proposals * x y w h class_one_hot is_back
def get_rects_and_class_tensors(x):
    print("get_rects_and_class_tensors ")
    print(x)

    n_proposals = N_PROPOSALS
    batch_size = tf.shape(x)[0] # needed for slice
    class_count = 3 # :( :( :( :(
    rects_res = tf.slice(x, [0, 0, 0], [batch_size, n_proposals, 4])

    is_background_res = tf.slice(x, [0, 0, 4], [batch_size, n_proposals, 1])

    # last_ind = class_count + 1 # + is background (empt ry class can be removed in future)
    one_hot_class_res = tf.slice(x, [0, 0, 5], [batch_size, n_proposals, class_count])

    return rects_res, is_background_res, one_hot_class_res

def object_detection_loss(y_true, y_pred):
    true_rects_res, true_is_background_res, true_one_hot_class_res = get_rects_and_class_tensors(y_true)
    pred_rects_res, pred_is_background_res, pred_one_hot_class_res = get_rects_and_class_tensors(y_pred)

    MSE = mse(true_rects_res, pred_rects_res)
    background_binary_entropy = binary_crossentropy(true_is_background_res, pred_is_background_res)
    c_crossentropy = categorical_crossentropy(true_one_hot_class_res, pred_one_hot_class_res)

    #
    backround_coeff = 1.1 # a bit more imporatant to detect background/non-background
    total_loss = MSE + c_crossentropy + backround_coeff * background_binary_entropy

    return total_loss

@tf.function
def object_detection_activation(x):
    print("object_detection_activation", x.shape)
    rects_res, is_background_res, one_hot_class_res = get_rects_and_class_tensors(x)

    activated_rects_res = keras.activations.relu(rects_res)

    activated_is_background_res = keras.activations.sigmoid(is_background_res)

    activated_one_hot_class_res = keras.activations.softmax(one_hot_class_res)

    act_concated_res = tf.concat([activated_rects_res,
                                  activated_is_background_res,
                                  activated_one_hot_class_res], axis=2)

    # print("activated_rects_res shape", activated_rects_res.shape)
    # tf.print(activated_rects_res)

    # print("activated_is_background_res shape", activated_is_background_res.shape)
    # tf.print(activated_is_background_res)

    # print("activated_one_hot_class_res shape", activated_one_hot_class_res.shape)
    # tf.print(activated_one_hot_class_res)

    return act_concated_res


class ImageGridOneCellLayer(keras.layers.Layer):
    def __init__(self, shape):
        super(ImageGridOneCellLayer, self).__init__()

        self.dense1 = Dense(N_PROPOSALS * shape)
        self.dense2 = Reshape((N_PROPOSALS, shape))
        self.activation = Activation(object_detection_activation)

    def call(self, inputs):
        # print("inputs", inputs)

        # x = tf.reshape(inputs, [64,])
        # x = self.input_l(inputs)
        x = self.dense1(inputs)
        # print("CELL dense1 X", x)

        x = self.dense2(x)
        x = self.activation(x)
        return x

class ImageGridLayer(keras.layers.Layer):
    def __init__(self):
        super(ImageGridLayer, self).__init__()

        n_classes = 3
        self.one_cell_pred_shape = 4 + n_classes + 1 # xywh n_c + is back
        self.cells = []
        for r_index in range(0, CELLS_COUNT * CELLS_COUNT):
                one_cell_l = ImageGridOneCellLayer(shape=self.one_cell_pred_shape)
                self.cells.append(one_cell_l)

        # tensor_cells = tf.convert_to_tensor(cells) # imposible too do !

    def call(self, inputs):
        slices = tf.reshape(inputs, (BATCH_SIZE, CELLS_COUNT, CELLS_COUNT, CELL_SIDE, CELL_SIDE))
        preds = []# np.empty()
        for r_index in range(0, CELLS_COUNT):
            for c_index in range(0, CELLS_COUNT):
                sliced_cell = tf.slice(slices, [0, r_index, c_index, 0, 0], [BATCH_SIZE, 1, 1, 8, 8])

                batch_current_pos_cells = tf.reshape(sliced_cell, (BATCH_SIZE, CELL_SIDE * CELL_SIDE))
                preds_item = self.cells[r_index + c_index](batch_current_pos_cells)

                preds.append(preds_item)

        # preds = tf.reshape(preds, (BATCH_SIZE, 8, 5, 8))
        # print("ImageGridLayer", inputs)
        # print("slices", slices)
        # print("cells len", len(self.cells))
        # print("cells[0] shape", self.cells[0].shape)

        return tf.transpose(preds, [1, 0, 2, 3])

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

        self.dense2 = Dense(output_len, activation="relu")
        self.image_grid_layer = ImageGridLayer()

        # self.output_l = keras.layers.Concatenate()
        # self.output_l = conc_l
        # self.model = keras.Model(base_model.input, conc_l)
        # self.model = model

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pooling_l(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense2(x)
        x = self.image_grid_layer(x)

        ## ConcateLayer
        return x#keras.layers.Concatenate()(x)

    def anchor_boxes(self):
        print("anchor_boxes")
        img_w = 224
        img_h = 224
        for index, cell in enumerate(self.image_grid_layer.cells):
            print(index)
            # print(cell.shape)

    @staticmethod
    def mse_loss(y_true, y_pred):
        print("y_pred.shape", y_pred.shape)
        true_rects_res, _, _ = get_rects_and_class_tensors(y_true)
        pred_rects_res, _, _ = get_rects_and_class_tensors(y_pred)

        return mse(true_rects_res, pred_rects_res)

    @staticmethod
    def background_loss(y_true, y_pred):
        _, true_is_background_res, _ = get_rects_and_class_tensors(y_true)
        _, pred_is_background_res, _ = get_rects_and_class_tensors(y_pred)

        return binary_crossentropy(true_is_background_res, pred_is_background_res)

    @staticmethod
    def class_loss(y_true, y_pred):
        _, _, true_one_hot_class_res = get_rects_and_class_tensors(y_true)
        _, _, pred_one_hot_class_res = get_rects_and_class_tensors(y_pred)

        return categorical_crossentropy(true_one_hot_class_res, pred_one_hot_class_res)

if __name__ == '__main__':
    model = YoloLikeModel()
    print(model.anchor_boxes())


