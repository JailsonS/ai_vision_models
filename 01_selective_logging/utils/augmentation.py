import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import backend as  K



def flip_up_down(images, label):
    unstacked = tf.unstack(images, axis=2)
    unstacked.append(label[:,:,0])
    stacked = tf.stack(unstacked, axis=2)


    images = tf.image.flip_up_down(stacked)

    x, y = images[:, :, :-1], images[:, :, -1:]

    return x, y

def flip_left_right(images, label):
    unstacked = tf.unstack(images, axis=2)
    unstacked.append(label[:,:,0])
    stacked = tf.stack(unstacked, axis=2)

    images = tf.image.flip_left_right(stacked)
    x, y = images[:, :, :-1], images[:, :, -1:]

    return x, y

def rotate(images, label):
    unstacked = tf.unstack(images, axis=2)
    unstacked.append(label[:,:,0])
    stacked = tf.stack(unstacked, axis=2)

    images = tf.image.rot90(stacked)
    x, y = images[:, :, :-1], images[:, :, -1:]

    return x, y

def apply_augmentation(dataset):
    rotated = dataset.map(rotate)
    flipped_l_r = dataset.map(flip_left_right)
    flipped_u_d = dataset.map(flip_up_down)

    res = dataset.concatenate(rotated).concatenate(flipped_l_r).concatenate(flipped_u_d)

    return res.flat_map(lambda x, y: tf.data.Dataset.from_tensors((x, y)))
