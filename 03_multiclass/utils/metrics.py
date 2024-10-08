
import tensorflow as tf
from tensorflow.keras import backend as  K

def running_recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + K.epsilon())
    return recall


def running_recall_multi(y_true, y_pred):

    # Converte as previsões para a classe com maior probabilidade
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    
    # Calcula o número de verdadeiros positivos (TP) e TP + FN
    TP = K.sum(K.cast(K.equal(y_true, y_pred), dtype='float32'))
    TP_FN = K.sum(K.cast(K.greater_equal(y_true, 0), dtype='float32'))
    
    # Calcula recall
    recall = TP / (TP_FN + K.epsilon())
    return recall





def running_precision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + K.epsilon())
    return precision


def running_precision_multi(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    
    TP = K.sum(K.cast(K.equal(y_true, y_pred), dtype='float32'))
    TP_FP = K.sum(K.cast(K.greater_equal(y_pred, 0), dtype='float32'))
    
    precision = TP / (TP_FP + K.epsilon())
    return precision



def running_f1(y_true, y_pred):
    precision = running_precision(y_true, y_pred)
    recall = running_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def running_f1_multi(y_true, y_pred):
    precision = running_precision_multi(y_true, y_pred)
    recall = running_recall_multi(y_true, y_pred)
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

def soft_dice_loss_multi(y_pred, y_true, smooth=1):
    """
    Função de perda Soft Dice Loss adaptada para múltiplas classes.
    
    Parâmetros:
    y_true: tensor de verdadeiros (one-hot encoded), shape (batch_size, height, width, num_classes)
    y_pred: tensor de predições, shape (batch_size, height, width, num_classes)
    smooth: valor pequeno para evitar divisão por zero (padrão = 1)
    
    Retorno:
    loss: valor médio do Dice Loss entre as classes
    """
    # Converte y_true e y_pred para float32, caso não estejam nesse formato

   


    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    y_pred_f = K.clip(y_pred_f, K.epsilon(), 1. - K.epsilon())


    # Achata as dimensões espaciais, mantendo a dimensão das classes
    y_true_f = K.flatten(y_true_f)
    y_pred_f = K.flatten(y_pred_f)

    # Calcula a interseção e a soma de todos os valores por classe
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)

    # Calcula o Dice por classe
    dice_per_class = (2. * intersection + smooth) / (union + smooth)

    # Retorna o Dice Loss médio entre as classes (1 - Dice médio)
    dice_loss = 1 - K.mean(dice_per_class)

    return dice_loss