
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
        'size': 600 * 4
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
        'save_ckpt': True,
        'batch_size':20,
        'epochs': 50,
        'output_model': '03_multiclass/model',
        'output_ckpt':'03_multiclass/model/ckpt',
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)
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
    labels = tf.io.parse_tensor(example["labels"], tf.int8)

    # tensorFlow can't infer the shapes, so we set them explicitly.
    inputs.set_shape([None, None, len(list(config['channels'].values()))])
    labels.set_shape([None, None, 1])

    # classifications are measured against one-hot encoded vectors.
    one_hot_labels = tf.one_hot(labels[:, :, 0], config['number_output_classes'])
    return (inputs, one_hot_labels)

def replace_nan(data, label):

    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)

    label = tf.where(tf.math.is_nan(label), tf.zeros_like(label), label)
    label = tf.where(tf.math.is_inf(label), tf.zeros_like(label), label)

    return data, label

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

def filter_inconsistent_shapes(image, mask):

    expected_image_shape = len(list(config['channels']))  # Forma esperada para a imagem
    expected_mask_shape = config['number_output_classes']

    current_image_shape = image.shape[-1]
    current_mask_shape = mask.shape[-1]

    return (expected_image_shape == current_image_shape) & (expected_mask_shape == current_mask_shape)

def count_samples(dataset):
    count = 0
    for _ in dataset: count += 1
    return count

'''

    Input Data

'''


dataset_train = tf.data.TFRecordDataset([config['train_dataset']['path']])\
    .map(read_example)\
    .map(normalize_channels)


dataset_val = tf.data.TFRecordDataset([config['val_dataset']['path']])\
    .map(read_example)\
    .map(normalize_channels)


'''
    Compute total dataset size
'''

config['train_dataset']['size'] = count_samples(dataset_train)
config['val_dataset']['size'] = count_samples(dataset_val)

'''

    Apply Data Augmentation

'''


dataset_train = apply_augmentation(dataset_train)\
    .map(replace_nan)\
    .repeat()\
    .batch(config['model_params']['batch_size'], drop_remainder=True)\
    .filter(lambda image, mask: filter_inconsistent_shapes(image, mask))\
    .prefetch(tf.data.AUTOTUNE)

dataset_val = dataset_val\
    .map(replace_nan)\
    .batch(1, drop_remainder=True).repeat()


dataset_val = dataset_val.filter(lambda image, mask: filter_inconsistent_shapes(image, mask))




# for inputs, labels in dataset_train.take(15):
#     tf.debugging.check_numerics(inputs, 'Input contains NaN or Inf')
#     tf.debugging.check_numerics(labels, 'Labels contain NaN or Inf')

'''

    Prepare Model

'''

model = Unet(
    list(dict(config['channels']).values()), 
    optimizer=config['model_params']['optimizer'], 
    loss=config['model_params']['loss'], 
    metrics=config['model_params']['metrics'],
    multiclass=True
)

model = model.getModel(n_classes=config['number_output_classes'])



'''

    Run Model

'''

path_ckpt = os.path.dirname(config['model_params']['output_ckpt'])

path_log_dir = f"{config['base_path']}/model/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

csv_logger = CSVLogger(f"{config['base_path']}/model/{config['model_params']['model_name']}.csv", append=True, separator=';')


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
    steps_per_epoch=int(config['train_dataset']['size'] / config['model_params']['batch_size']),
    validation_data=dataset_val,
    validation_steps=int(config['val_dataset']['size'] / config['model_params']['batch_size']),
    callbacks=[cp_callback, tensorboard_callback, earlystopper_callback, csv_logger])



model.save(config['base_path'] + '/model/lulc_v1.keras')