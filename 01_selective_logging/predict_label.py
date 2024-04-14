
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import tensorflow.keras.preprocessing as prep
import numpy as np
import tensorflow as tf
import pandas as pd
import keras

from tensorflow.keras import backend as  K
from tensorflow.keras.callbacks import CSVLogger

from utils.metrics import *
from utils.augmentation import *
from models.UnetDefault import Unet

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

'''
    Config Info
'''

# {'train': 369, 'val': 123, 'test': 124}
# {'train':222, 'val':74, 'test':74}

USE_TOTAL_CHANNELS = True
USE_FACTOR_BRIGHT = False
BANDS = [
    'red_t0','green_t0', 'blue_t0', 'nir_t0', 'swir1_t0',
    'red_t1','green_t1', 'blue_t1', 'nir_t1', 'swir1_t1'
]

TARGET_BANDS = [
    0, 1, 2,
    5, 6, 7
]

KERNEL_SIZE = 512

NUM_CLASSES = 1

TEST_DATASET = '01_selective_logging/data/test_dataset_3.tfrecord'
TRAIN_DATASET = '01_selective_logging/data/train_dataset_3.tfrecord'
VAL_DATASET = '01_selective_logging/data/val_dataset_3.tfrecord'


BATCH_SIZE = 9

MODEL_NAME = 'model_v4.keras'
MODEL_OUTPUT = f'01_selective_logging/model/{MODEL_NAME}'



FILENAME_TRAIN = '01_selective_logging/data/train_dataset_4.tfrecord'
FILENAME_TEST = '01_selective_logging/data/test_dataset_4.tfrecord'
FILENAME_VAL = '01_selective_logging/data/val_dataset_4.tfrecord'

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

    if not USE_TOTAL_CHANNELS:
        data = tf.stack([data[:,:,x] for x in TARGET_BANDS], axis=2)

   

    return data, label


def normalize_channels(data, label=None):
    
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


def serialize(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        )
        for name, data in {"inputs": inputs, "labels": labels}.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


'''

    Input Data

'''




dataset_test = tf.data.TFRecordDataset([TEST_DATASET])\
    .map(read_example)\
    .map(replace_nan)\
    #.map(normalize_channels)

dataset_train = tf.data.TFRecordDataset([TRAIN_DATASET])\
    .map(read_example)\
    .map(replace_nan)\
    #.map(normalize_channels)

dataset_val = tf.data.TFRecordDataset([VAL_DATASET])\
    .map(read_example)\
    .map(replace_nan)\
    #.map(normalize_channels)


'''

    Train model

'''

model = keras.saving.load_model(MODEL_OUTPUT, compile=False)



model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), 
    loss=soft_dice_loss, 
    metrics=[
        running_recall, 
        running_f1, 
        running_precision, 
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

# model.evaluate(dataset_test.batch(9))

font = {
    'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 10,
}

'''
# optionally compress files.
writer = tf.io.TFRecordWriter(FILENAME_TRAIN)

for data, label in dataset_train:

    data_for = tf.stack([data[:,:,x] for x in TARGET_BANDS], axis=2)
    data_for, _ = normalize_channels(data_for)

    input_data = tf.expand_dims(data_for, axis=0)

    
    probabilities = model.predict(input_data)
    probabilities[probabilities < 0.01] = 0
    probabilities[probabilities >= 0.01] = 1
    output = probabilities[0]

    output = tf.cast(output, tf.int64)


    serialized = serialize(data, output)

    writer.write(serialized)
    writer.flush()


writer.close()

'''

# optionally compress files.
writer = tf.io.TFRecordWriter(FILENAME_TEST)

for data, label in dataset_test:

    data_for = tf.stack([data[:,:,x] for x in TARGET_BANDS], axis=2)
    data_for, _ = normalize_channels(data_for)

    input_data = tf.expand_dims(data_for, axis=0)
    
    probabilities = model.predict(input_data)
    probabilities[probabilities < 0.01] = 0
    probabilities[probabilities >= 0.01] = 1
    output = probabilities[0]

    output = tf.cast(output, tf.int64)

    serialized = serialize(data, output)

    writer.write(serialized)
    writer.flush()

writer.close()








# optionally compress files.
writer = tf.io.TFRecordWriter(FILENAME_VAL)

for data, label in dataset_val:

    data_for = tf.stack([data[:,:,x] for x in TARGET_BANDS], axis=2)
    data_for, _ = normalize_channels(data_for)

    input_data = tf.expand_dims(data_for, axis=0)
    
    probabilities = model.predict(input_data)
    probabilities[probabilities < 0.01] = 0
    probabilities[probabilities >= 0.01] = 1
    output = probabilities[0]

    output = tf.cast(output, tf.int64)

    serialized = serialize(data, output)

    writer.write(serialized)
    writer.flush()

writer.close()
