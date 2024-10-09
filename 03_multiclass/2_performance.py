
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


config = {

    'base_path': '03_multiclass',

    'channels': {
        'gv':0, 
        'npv':1, 
        'soil':2, 
        'cloud':3,
        'gvs':4,
        'ndfi':5, 
        #'csfi':6
    },

    'chip_size': 256,

    'train_dataset': {
        'path': '03_multiclass/data/train.tfrecord',
        'size': 1800 * 4
    },

    'val_dataset': {
        'path': '03_multiclass/data/val.tfrecord',
        'size': 600
    },

    'test_dataset': {
        'path': '03_multiclass/data/test.tfrecord',
        'size': 600
    },

    'number_output_classes': 8,

    'model_params': {
        'model_name':'multiclass_s2_v1',
        'loss': soft_dice_loss,
        'metrics':[
            running_recall, 
            running_f1, 
            running_precision, 
            tf.keras.metrics.OneHotIoU(
                num_classes=8,
                target_class_ids=[0,1,2,3,4,5,6,7],
            )
        ],
        'batch_size':15,
        'epochs': 50,
        'output_model': '03_multiclass/model',
        'output_ckpt':'03_multiclass/model/ckpt',
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)
    }
}

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
    labels = tf.io.parse_tensor(example["labels"], tf.int8)

    # tensorFlow can't infer the shapes, so we set them explicitly.
    inputs.set_shape([None, None, len(list(config['channels'].values()))])
    labels.set_shape([None, None, 1])

    # classifications are measured against one-hot encoded vectors.
    one_hot_labels = tf.one_hot(labels[:, :, 0], config['number_output_classes'])
    return (inputs, one_hot_labels)

def normalize_channels(data, label):

    feature_index = list(config['channels'].values())

    data_filtered = tf.gather(data, tf.constant(feature_index), axis=-1)

    unstacked = tf.unstack(data_filtered, axis=2)

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

    return data_normalized, label

def replace_nan(data, label):

    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)

    label = tf.where(tf.math.is_nan(label), tf.zeros_like(label), label)
    label = tf.where(tf.math.is_inf(label), tf.zeros_like(label), label)

    return data, label



'''

    Input Data

'''


dataset_test = tf.data.TFRecordDataset([config['test_dataset']['path']])\
    .map(read_example)\
    .map(normalize_channels)\
    .map(replace_nan)\
    .batch(config['model_params']['batch_size'])


'''

    Train model

'''

model = keras.saving.load_model(config['model_params']['output_model'] + '/lulc_v1.keras', compile=False)


model.compile(
    optimizer=config['model_params']['optimizer'], 
    loss=config['model_params']['loss'], 
    metrics=config['model_params']['metrics']
)

model.evaluate(dataset_test)

# loss: 0.0405 - running_recall: 0.9468 - running_f1: 0.9494 - running_precision: 0.9520 - one_hot_io_u: 0.7181