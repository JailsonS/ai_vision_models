
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import tensorflow as tf
import pandas as pd
import datetime, os

from tensorflow.keras import backend as  K
from tensorflow.keras.callbacks import CSVLogger
from glob import glob
from utils.metrics import *
from utils.augmentation import *
from models.UnetDefault import Unet





'''
    Config Info
'''

config = {

    'base_path': '',

    'channels': {
        'red': 0,
        'green': 1,
        'blue': 2,
        'swir1': 3,
        'nir': 4
    },

    'chip_size': 256,

    'train_dataset': {
        'path': '',
        'size': None
    },
    'val_dataset': {
        'path': '',
        'size': None
    },

    'number_output_classes': 5,

    'model_params': {
        'model_name':'',
        'lr': 0.001,
        'loss':None,
        'metrics':[],
        'save_ckpt': True,
        'batch_size':None,
        'epochs': 50,
        'output_model': '',
        'output_ckpt':'',
        'optimizer': None
    }
}

'''
    Base functions
'''


def read_example(serialized: bytes) -> tuple[tf.Tensor, tf.Tensor]:

    features_dict = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized, features_dict)


    inputs = tf.io.parse_tensor(example["inputs"], tf.float32)
    labels = tf.io.parse_tensor(example["labels"], tf.int64)

    # tensorFlow can't infer the shapes, so we set them explicitly.
    inputs.set_shape([None, None, len(config['channels'])])
    labels.set_shape([None, None, 1])

    # classifications are measured against one-hot encoded vectors.
    one_hot_labels = tf.one_hot(labels[:, :, 0], config['number_output_classes'])
    return (inputs, one_hot_labels)

def replace_nan(data, label):
    label_temp = tf.add(label, 1)
    label = tf.where(tf.equal(label_temp, 2.0), 0.0, label_temp)

    data = tf.where(tf.math.is_nan(data), 0., data)
    label = tf.where(tf.math.is_nan(label), 0., label)

    return data, label


'''

    Input Data

'''


dataset_train = tf.data.TFRecordDataset([config['train_dataset']['path']])\
    .map(read_example)

dataset_val = tf.data.TFRecordDataset([config['val_dataset']['path']])\
    .map(read_example)



'''

    Apply Data Augmentation

'''


dataset_train = apply_augmentation(dataset_train)\
    .repeat()\
    .batch(config['model_params']['batch_size'])\
    .prefetch(tf.data.AUTOTUNE)


dataset_val = dataset_val.batch(1).repeat()




'''

    Prepare Model

'''

model = Unet(
    list(dict(config['channels']).values()), 
    optimizer=config['model_params']['optimizer'], 
    loss=config['model_params']['loss'], 
    metrics=config['model_params']['metrics']
)

model = model.getModel(n_classes=config['number_output_classes'])



'''

    Run Model

'''

path_ckpt = os.path.dirname(config['model_params']['output_ckpt'])

path_log_dir = f"{config['base_path']}/model/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

csv_logger = CSVLogger(f'{config['base_path']}/model/{config["model_params"]['model_name']}.csv', append=True, separator=';')


# check if it has ckpt points
if len(glob(config['model_params']['output_ckpt'] + '/*')) > 0:
    print('loaded ckpt')
    model.load_weights(config['model_params']['output_ckpt'])





cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path_ckpt,
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

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_log_dir, histogram_freq=1)




model.fit(
    x=dataset_train,
    epochs=config['model_params']['epochs'],
    steps_per_epoch=int(),
    validation_data=dataset_val,
    validation_steps=int(),
    callbacks=[cp_callback, tensorboard_callback, earlystopper_callback, csv_logger])