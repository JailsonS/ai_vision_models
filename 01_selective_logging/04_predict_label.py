
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import numpy as np
import rasterio, concurrent

from tensorflow.keras import backend as  K

from utils.metrics import *
from utils.augmentation import *
from models.UnetDefault import Unet

from glob import glob
from retry import retry


'''
    Config Info
'''

# {'train': 369, 'val': 123, 'test': 124}
# {'train':222, 'val':74, 'test':74}

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

USE_TOTAL_CHANNELS = True
USE_FACTOR_BRIGHT = False
USE_CLOUD_DATASET = True

BANDS = [
    'red_t0',
    'green_t0',
    'blue_t0',
    'nir_t0',
    'swir1_t0',
    
    'gv_t0',
    'npv_t0',
    'soil_t0',
    'shade_t0',
    'cloud_t0',
    
    'red_t1',
    'green_t1',
    'blue_t1',
    'nir_t1',
    'swir1_t1',
    
    'gv_t1',
    'npv_t1',
    'soil_t1',
    'shade_t1',
    'cloud_t1',
    
    'ndfi_t0',
    'ndfi_t1',
    #'pre_label'
]

TARGET_BANDS = [
    0, 1, 2,
    10, 11, 12
]

KERNEL_SIZE = 512

NUM_CLASSES = 1

# root increment version
PATH_DATASET = '01_selective_logging/data/logging_v1'
OUTPUT_DATASET = '01_selective_logging/data/logging_v2/{}'

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



@retry()
def get_patch(path):

    """Get a patch centered on the coordinates, as a numpy array."""

    id = path.split('/')[-1]

    np_image = rasterio.open(path)


    # [c, w, h]
    image = np_image.read()


    # [w, h, c]
    image = tf.transpose(image, [1,2,0])


    # resize image
    image = tf.image.resize(image, size=(KERNEL_SIZE, KERNEL_SIZE))

    input = image[:, :, :-1]
    label = image[:, :, -1:]



    # replace zeros and ones
    # label_temp = tf.add(label, 1)
    # label = tf.where(tf.equal(label_temp, 2.0), 0.0, label_temp)

    # convert to int
    label = tf.cast(label, tf.int64)

    return input, label, id, np_image.meta



def write_dataset(enum_items):
    """"Write patches at the sample points into a TFRecord file."""

    future_to_point = {EXECUTOR.submit(get_patch, path): path for path in enum_items}

    for future in concurrent.futures.as_completed(future_to_point):
        input, _, id, metadata = future.result()

        input_data = np.stack([input[:, :, x] for x in TARGET_BANDS], 2)
        input_data, _ = normalize_channels(input_data)
        input_data = tf.expand_dims(input_data, axis=0)

        metadata['height'] = KERNEL_SIZE
        metadata['width'] = KERNEL_SIZE

        probabilities = model.predict(input_data)
        probabilities[probabilities < 0.5] = 0
        probabilities[probabilities >= 0.5] = 1

        output = probabilities[0]

        new_sample = tf.unstack(input, axis=2)
        new_sample.append(output[:,:,0])
        new_sample = np.stack(new_sample, axis=2)

        new_sample = np.transpose(new_sample, (2,0,1))

        with rasterio.open(
            OUTPUT_DATASET.format(id),
            'w',
            driver = 'COG',
            count = 23,
            height = KERNEL_SIZE,
            width  = KERNEL_SIZE,
            dtype  = metadata['dtype'],
            crs    = rasterio.crs.CRS.from_epsg(4326),
            transform=metadata['transform']
        ) as output:
            output.write(new_sample)

        os.remove(f'{PATH_DATASET}/{id}')

        





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


'''

    Pred Data

'''

list_path = list(glob(PATH_DATASET + '/*'))

# list_loaded = list(glob(OUTPUT_DATASET.format('*')))

write_dataset(list_path)