import os
import time
import keras
import tensorflow as tf

from accuracy_callback import AccuracyCallback
from object_detection_acc_callback import ObjectDetectionAccCallback

from data_generator import DataGenerator
from model_builder import ModelBuilder, mse_loss, background_loss, class_loss
from path_config import PNG_TRAIN_DATASET, TENSORBOARD_PATH, TaskType, BATCH_SIZE
from dataset_tool import DatasetTool, TaskType

def train():
    print("GPU:", tf.test.is_gpu_available())
    task_type = TaskType.OBJECT_DETECTION

    path = os.path.join(PNG_TRAIN_DATASET, "train")

    train_dataset_tool = DatasetTool(path, task_type, False)
    val_dataset_tool = DatasetTool(path, task_type, True)

    training_generator = DataGenerator(path,
                                       train_dataset_tool,
                                       is_val=False,
                                       batch_size=BATCH_SIZE)

    validation_generator = DataGenerator(path,
                                         val_dataset_tool,
                                         is_val=True,
                                         batch_size=BATCH_SIZE)

    #MODEL
    model_builder = ModelBuilder(task_type,
                                 n_classes=train_dataset_tool.n_classes)

    model = model_builder.yolo_like_model()

    for index, layer in enumerate(model.layers):
        # last_train_layer_index = int(len(model.layers) - 15)
        layer.trainable = True #index > last_train_layer_index

    # model.summary()
    print("len(model.layers) :", len(model.layers))

    optimizer = keras.optimizers.RMSprop()

    model.compile(
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')
    callbacks = [tensorboard_callback]

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        callback = AccuracyCallback(validation_generator)
        callbacks.append(callback)
    elif task_type == TaskType.OBJECT_DETECTION:
        callback = ObjectDetectionAccCallback(validation_generator)
        callbacks.append(callback)

    #TRAIN
    epochs = 25

    mse_means = []
    previous_mse_value = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        start_time = time.time()
        start_epoch_time = time.time()
        total_reading_time_per_epoch = 0

        for step, (x_batch_train, y_batch_train) in enumerate(training_generator):
            data_gen_batch_reading_time = time.time() - start_time
            total_reading_time_per_epoch += data_gen_batch_reading_time

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                if task_type == TaskType.OBJECT_DETECTION:
                    mse_loss_value = mse_loss(y_batch_train, logits)
                    background_loss_value = background_loss(y_batch_train, logits)
                    class_loss_value = class_loss(y_batch_train, logits)
                    total_loss = mse_loss_value + background_loss_value + class_loss_value
                else:
                    total_loss = model_builder.loss(y_batch_train, logits)


            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # Log every 200 batches.
            if step % 10 == 0:
                total_loss_mean = tf.math.reduce_mean(total_loss).numpy()
                print("\nTraining total loss (for one batch) at step %d: %.4f" % (step, float(total_loss_mean)))
                print("data_gen_batch_reading_time of previous step %.4f:" % data_gen_batch_reading_time)

                if task_type == TaskType.OBJECT_DETECTION:
                    mse_loss_mean = tf.math.reduce_mean(mse_loss_value).numpy()
                    if previous_mse_value == 0:
                        previous_mse_value = mse_loss_mean

                    mse_means.append(mse_loss_mean)
                    background_loss_mean = tf.math.reduce_mean(background_loss_value).numpy()
                    class_loss_mean = tf.math.reduce_mean(class_loss_value).numpy()

                    print(
                        "mse_loss_mean %.4f, background loss %.4f, class loss %.4f"
                        % (float(mse_loss_mean), float(background_loss_mean),  float(class_loss_mean))
                    )

            start_time = time.time()

        start_callback_time = time.time()
        for p_callback in callbacks:
            p_callback.model = model
            p_callback.on_epoch_end(epoch)

        av_mse_mean = sum(mse_means) / len(mse_means)
        if av_mse_mean < previous_mse_value:
            print("SAVE previous_mse_value {}, av_mse_mean {}".format(previous_mse_value, av_mse_mean))

            previous_mse_value = av_mse_mean
            checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"

            model.save_weights(checkpoint_path.format(epoch=epoch))

        callback_time = time.time() - start_callback_time
        epoch_time = time.time() - start_epoch_time

        print("\n !!!EPOCH FINISHED!!!\n epoch_time %.4f, data reading time %.4f, callback time %.4f" % (epoch_time,
                                                                               total_reading_time_per_epoch,
                                                                               callback_time))


if __name__ == '__main__':
    train()
