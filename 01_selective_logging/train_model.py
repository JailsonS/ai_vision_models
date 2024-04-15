
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import backend as  K
from tensorflow.keras.callbacks import CSVLogger

from utils.metrics import *
from utils.augmentation import *
from models.UnetDefault import Unet


'''
    Config Info
'''

# {'train': 369, 'val': 123, 'test': 124}
# {'train':222, 'val':74, 'test':74}
REPLACE_ZEROS = True
USE_TOTAL_CHANNELS = False
USE_FACTOR_BRIGHT = False
BANDS = [
    'red_t0','green_t0', 'blue_t0', 'nir_t0', 'swir1_t0',
    'red_t1','green_t1', 'blue_t1', 'nir_t1', 'swir1_t1'
    # 'ndfi_t0','ndfi_t1'
]

TARGET_BANDS = [
    0, 1, 2,
    5, 6, 7
]

KERNEL_SIZE = 512

NUM_CLASSES = 1

TRAIN_DATASET = '01_selective_logging/data/train_dataset_3.tfrecord'
VAL_DATASET = '01_selective_logging/data/val_dataset_3.tfrecord'

# config train variables
PATH_CHECK_POINTS = '01_selective_logging/model/ckpt4/checkpoint.x'

HYPER_PARAMS = {
    'optimizer': tf.keras.optimizers.Nadam(learning_rate=0.001),
    #'optimizer': tf.keras.optimizers.legacy.SGD(learning_rate=0.0002),
    'loss': 'MeanSquaredError',
    'metrics': ['RootMeanSquaredError']
}

TRAIN_SIZE = 222
#VAL_SIZE = 123
VAL_SIZE = 74

SAVE_CPKT = True
EPOCHS = 10
BATCH_SIZE = 9

TRAIN_STEPS = int(TRAIN_SIZE / BATCH_SIZE)
VAL_STEPS = int(VAL_SIZE / BATCH_SIZE)

MODEL_NAME = 'unet_default_logging_4_2'
MODEL_OUTPUT = f'01_selective_logging/model/{MODEL_NAME}'

'''

    Auxiliar Functions

'''

def read_example(serialized: bytes) -> tuple[tf.Tensor, tf.Tensor]:
    features_dict = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized, features_dict)


    inputs = tf.io.parse_tensor(example["inputs"], tf.float32)
    labels = tf.io.parse_tensor(example["labels"], tf.int64)

    # TensorFlow can't infer the shapes, so we set them explicitly.
    inputs.set_shape([None, None, len(BANDS)])
    labels.set_shape([None, None, 1])

    # Classifications are measured against one-hot encoded vectors.
    one_hot_labels = tf.one_hot(labels[:, :, 0], NUM_CLASSES)
    return (inputs, one_hot_labels)



def replace_nan(data, label):

    if REPLACE_ZEROS:
        label_temp = tf.add(label, 1)
        label = tf.where(tf.equal(label_temp, 2.0), 0.0, label_temp)

    data = tf.where(tf.math.is_nan(data), 0., data)
    label = tf.where(tf.math.is_nan(label), 0., label)

    data_list = []
    if not USE_TOTAL_CHANNELS:
        data = tf.stack([data[:,:,x] for x in TARGET_BANDS], axis=2)

   

    return data, label


def normalize_channels(data, label):
    
    unstacked = tf.unstack(data, axis=2)

    data_norm = []

    for i in unstacked:
        min_arr = tf.reduce_min(i)
        max_arr = tf.reduce_max(i)

        tensor = tf.divide(
            tf.subtract(i, min_arr),
            tf.subtract(max_arr, min_arr)
        )

        data_norm.append(tensor)

    data_normalized = tf.stack(data_norm, axis=2)
    data_normalized = tf.clip_by_value(data_normalized * 1.5, 0, 1) if USE_FACTOR_BRIGHT else data_normalized

    return data_normalized, label




'''

    Input Data

'''




dataset_train = tf.data.TFRecordDataset([TRAIN_DATASET])\
    .map(read_example)\
    .map(replace_nan)\
    .map(normalize_channels)

dataset_val = tf.data.TFRecordDataset([VAL_DATASET])\
    .map(read_example)\
    .map(replace_nan)\
    .map(normalize_channels)

# dataset_train = dataset_train.shuffle(buffer_size=TRAIN_SIZE).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dataset_train = apply_augmentation(dataset_train).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dataset_val = dataset_val.batch(1).repeat()



'''

    Train model

'''

import datetime, os
from glob import glob

HYPER_PARAMS['loss'] = soft_dice_loss

listckpt = glob(PATH_CHECK_POINTS + '*')

model = Unet(
    TARGET_BANDS, 
    optimizer=HYPER_PARAMS['optimizer'], 
    loss=soft_dice_loss, 
    # metrics=[
    #     tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]),
    #     tf.keras.metrics.Precision(),
    #     tf.keras.metrics.Recall(),
    #     tf.keras.metrics.F1Score()
    # ]
    metrics=[
        running_recall, 
        running_f1, 
        running_precision, 
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)


model = model.getModel(n_classes=NUM_CLASSES)

if len(listckpt) > 0:
    print('loaded ckpt')
    model.load_weights(PATH_CHECK_POINTS)

if SAVE_CPKT:

    checkpoint_path = os.path.dirname(PATH_CHECK_POINTS)
    log_dir = f"01_selective_logging/model/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    csv_logger = CSVLogger(f'01_selective_logging/model/{MODEL_NAME}.csv', append=True, separator=';')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=PATH_CHECK_POINTS,
        save_best_only=True,
        #save_weights_only=True,
        verbose=1
    )

    earlystopper_callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0,
        patience = 10,
        verbose = 1,
        restore_best_weights = True
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    model.fit(
        x=dataset_train,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=dataset_val,
        validation_steps=VAL_STEPS,
        callbacks=[cp_callback, tensorboard_callback, earlystopper_callback, csv_logger])

else:
    model.fit(
        x=dataset_train,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=dataset_val,
        validation_steps=VAL_STEPS)

'''
    Save Model
'''

model.save('01_selective_logging/model/model_v4.keras')