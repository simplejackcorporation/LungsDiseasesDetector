import os
import keras
import tensorflow as tf

import keras.backend as K

from accuracy_callback import AccuracyCallback
from data_generator import DataGenerator
from model_builder import ModelBuilder
from path_config import PNG_TRAIN_DATASET, TENSORBOARD_PATH, TaskType, BATCH_SIZE
from dataset_tool import DatasetTool, TaskType

def train():
    print("GPU:", tf.test.is_gpu_available())
    task_type = TaskType.OBJECT_DETECTION

    path = os.path.join(PNG_TRAIN_DATASET, "train")

    train_dataset_tool = DatasetTool(path, task_type, False)
    val_dataset_tool = DatasetTool(path, task_type, True)

    training_generator = DataGenerator(path, train_dataset_tool,  is_val=False, batch_size=BATCH_SIZE)
    validation_generator = DataGenerator(path, val_dataset_tool, is_val=True, batch_size=BATCH_SIZE)

    #MODEL
    model_builder = ModelBuilder(task_type, n_classes = train_dataset_tool.n_classes)
    model = model_builder.model()

    for index, layer in enumerate(model.layers):
        last_train_layer_index = int(len(model.layers) - 5)
        layer.trainable = index > last_train_layer_index

    model.summary()
    print("len(model.layers) :", len(model.layers))

    optimizer = keras.optimizers.RMSprop()
    loss = model_builder.loss

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],

        # loss_weights=None,
        # weighted_metrics=None,
        # run_eagerly=None,
        # steps_per_execution=None,
    # **kwargs	Arguments supported for backwards compatibility only.
    )

    checkpoint_filepath = r'C:\Users\m\Desktop\LUNGS\lungs-finder\weights'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')
    callbacks = [tensorboard_callback]

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        callback = AccuracyCallback(validation_generator)
        callbacks.append(callback)
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True,
    #     callbacks=[tensorboard_callback])

    #TRAIN
    model.fit(training_generator,
              validation_data=None if task_type == TaskType.MULTICLASS_CLASSIFICATION else validation_generator,
              callbacks=callbacks,
              epochs=10,
              use_multiprocessing=True)

    # model.fit_generator(generator=training_generator,
    #                     # validation_data=,
    #                     epochs=15,
    #                     callbacks=callbacks,
    #                     workers=6,
    #                     use_multiprocessing=True)

if __name__ == '__main__':
    train()