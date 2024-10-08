
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

REPLACE_ZEROS = True
USE_TOTAL_CHANNELS = False
USE_FACTOR_BRIGHT = False

# BANDS = [
#     'gv_t0',
#     'npv_t0',
#     'soil_t0',
#     'shade_t0',
#     'cloud_t0',
# 
#     'gv_t1',
#     'npv_t1',
#     'soil_t1',
#     'shade_t1',
#     'cloud_t1',
# 
#     'ndfi_t0',
#     'ndfi_t1',
# ]
# 
# TARGET_BANDS = [
#     0,1,2,3,4,
#     5,6,7,8,9,
#     10,11
# ]

BANDS = [
    'red_t0',
    'green_t0',
    'blue_t0',
    'nir_t0',
    'swir1_t0',

    'red_t1',
    'green_t1',
    'blue_t1',
    'nir_t1',
    'swir1_t1',
]

TARGET_BANDS = [
    0,1,2,
    5,6,7
]

KERNEL_SIZE = 512

NUM_CLASSES = 1

TEST_DATASET = '01_selective_logging/data/test_dataset_8.tfrecord'


BATCH_SIZE = 9

MODEL_NAME = 'model_v5.keras'
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




dataset_test = tf.data.TFRecordDataset([TEST_DATASET])\
    .map(read_example)\
    .map(replace_nan)\
    .map(normalize_channels)


'''

    Train model

'''

model = keras.saving.load_model(MODEL_OUTPUT, compile=False)



model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), 
    loss=soft_dice_loss, 
    metrics=[
        'accuracy',
        running_recall, 
        running_f1, 
        running_precision, 
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

model.evaluate(dataset_test.batch(9))

# v5  loss: 0.1580 - accuracy: 0.9970 - running_recall: 0.9031 - running_f1: 0.8097 - running_precision: 0.7370 - io_u: 0.4177
# v5_ loss: 0.2288 - accuracy: 0.9820 - running_recall: 0.8621 - running_f1: 0.8223 - running_precision: 0.9084 - io_u: 0.0646
# v7  loss: 0.4664 - accuracy: 0.9950 - running_recall: 0.3468 - running_f1: 0.4328 - running_precision: 0.6665 - io_u: 0.1732
# v10 loss: 0.5193 - accuracy: 0.9587 - running_recall: 0.2942 - running_f1: 0.3915 - running_precision: 0.9294. - io_u: 0.0123

exit()

font = {
    'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 10,
}

for i in range(1, 70):

    for data, label in dataset_test.skip(i).take(1):

        print(data.shape)

        fig = plt.figure(figsize=(15,15), frameon=False)
        gs = GridSpec(2, 4, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        #ax4 = fig.add_subplot(gs[1, 1])

        # (tensor - )
        # r, g, b = data[:, :, 10], data[:, :, 11], data[:, :, 11]
        r1, g1, b1 = data[:, :, 3], data[:, :, 4], data[:, :, 5]


        #rgb = np.stack([r,g,b], 2)
        rgb1 = np.stack([r1,g1,b1], 2)

        #rgb_t1 = tf.clip_by_value(rgb1 * 1.5, 0, 1)
        #rgb_t0 = tf.clip_by_value(rgb * 1.5, 0, 1)

        ax1.imshow(rgb1)
        ax2.imshow(label.numpy())
        #ax3.imshow(label.numpy(), cmap='gray')

        ax1.set_title("", fontdict=font)
        ax2.set_title("", fontdict=font)
        #ax3.set_title("(c) - label", fontdict=font)


        

    probabilities = model.predict(dataset_test.skip(i).take(1).batch(1))


    probabilities[probabilities < 0.5] = 0
    probabilities[probabilities >= 0.5] = 1
    probabilities = probabilities[0]

    # probabilities = probabilities[0].argmax(axis=-1).astype(np.uint8)
    # probabilities = np.expand_dims(probabilities, axis=2)

    ax3.imshow(prep.image.array_to_img(probabilities))
    ax3.set_title("", fontdict=font)



    plt.savefig(f'01_selective_logging/predictions/{i}.png', bbox_inches='tight', pad_inches=0)
    