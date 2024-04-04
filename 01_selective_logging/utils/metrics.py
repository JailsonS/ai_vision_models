
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import backend as  K


def running_recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + K.epsilon())
    return recall

def running_precision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + K.epsilon())
    return precision

def running_f1(y_true, y_pred):
    precision = running_precision(y_true, y_pred)
    recall = running_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def adaptive_maxpool_loss(y_true, y_pred, alpha=0.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    positive = -y_true * K.log(y_pred) * alpha
    negative = -(1. - y_true) * K.log(1. - y_pred) * (1-alpha)
    pointwise_loss = positive + negative
    max_loss = tf.keras.layers.MaxPool2D(pool_size=8, strides=1, padding='same')(pointwise_loss)
    x = pointwise_loss * max_loss
    x = K.mean(x, axis=-1)
    return x

def soft_dice_loss(y_pred, y_true, smooth = 1):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    dice = K.abs(2. * intersection + smooth) / (K.abs(K.sum(K.square(y_true_f))) + K.abs(K.sum(K.square(y_pred_f))) + smooth)

    return 1-K.mean(dice)








