
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('./'))



import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import backend as  K


from utils.metrics import *
from utils.augmentation import *
from ..models import UnetDefault


'''
    Config Info
'''

# {'train': 369, 'val': 123, 'test': 124}

BANDS = ['ndfi_t0', 'ndfi_t1']

KERNEL_SIZE = 512

NUM_CLASSES = 1

TRAIN_DATASET = '01_selective_logging/data'
VAL_DATASET = '01_selective_logging/data'

# config train variables
PATH_CHECK_POINTS = 'src/logging/pipeline_a/model/ckpt1/'

HYPER_PARAMS = {
    'optimizer': tf.keras.optimizers.Nadam(learning_rate=0.001),
    #'optimizer': tf.keras.optimizers.legacy.SGD(learning_rate=0.0002),
    'loss': 'MeanSquaredError',
    'metrics': ['RootMeanSquaredError']
}

TRAIN_SIZE = 369
VAL_SIZE = 123

SAVE_CPKT = True
EPOCHS = 25
BATCH_SIZE = 9

TRAIN_STEPS = int(TRAIN_SIZE / BATCH_SIZE)
VAL_STEPS = int(VAL_SIZE / BATCH_SIZE)

MODEL_NAME = 'unet_default_logging_1'
MODEL_OUTPUT = f'models/{MODEL_NAME}'

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
    data = tf.where(tf.math.is_nan(data), 0., data)
    label = tf.where(tf.math.is_nan(label), 0., label)
    return data, label


'''

    Input Data

'''




dataset_train = tf.data.TFRecordDataset([TRAIN_DATASET])\
    .map(read_example)\
    .map(replace_nan)

dataset_val = tf.data.TFRecordDataset([VAL_DATASET])\
    .map(read_example)\
    .map(replace_nan)


dataset_train = dataset_train.shuffle(buffer_size=TRAIN_SIZE).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dataset_train = apply_augmentation(dataset_train).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dataset_val = dataset_val.batch(1).repeat()




'''

    Train model

'''

import datetime, os
from glob import glob

HYPER_PARAMS['loss'] = soft_dice_loss

listckpt = glob(PATH_CHECK_POINTS + '*')

model = UnetDefault(BANDS, optimizer=HYPER_PARAMS['optimizer'], loss=soft_dice_loss, metrics=[running_recall, running_f1, running_precision])

model = model.getModel(n_classes=NUM_CLASSES)

if len(listckpt) > 0:
    print('loaded ckpt')
    model.load_weights(PATH_CHECK_POINTS)

if SAVE_CPKT:

    checkpoint_path = os.path.dirname(PATH_CHECK_POINTS)
    log_dir = "src/logging/pipeline_a/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=PATH_CHECK_POINTS,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    earlystopper_callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0,
        patience = 5,
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
        callbacks=[cp_callback, tensorboard_callback, earlystopper_callback])

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

model.save(f'01_selective_logging/model/{MODEL_NAME}')