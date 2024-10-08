
import tensorflow as tf
from tensorflow.keras import backend as  K

def running_recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + K.epsilon())
    return recall


def running_recall_multi(y_true, y_pred):
    """
    Calcula o recall para tarefas de classificação multi-classe usando rótulos one-hot.
    
    Parâmetros:
    - y_true: tensor de rótulos verdadeiros, codificados em one-hot
    - y_pred: tensor de previsões do modelo, em probabilidade ou logit

    Retorna:
    - recall: recall global (ou média ponderada) entre todas as classes
    """
    # Converter previsões para binário (0 ou 1)
    y_pred_binary = K.round(K.clip(y_pred, 0, 1))  # Caso o y_pred seja probabilístico

    # Verdadeiros Positivos (TP): predições corretas da classe positiva
    TP = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)), axis=0)
    
    # Falsos Negativos (FN): predições incorretas onde era positivo, mas previu negativo
    FN = K.sum(K.round(K.clip(y_true * (1 - y_pred_binary), 0, 1)), axis=0)
    
    # Recall para cada classe
    recall_per_class = TP / (TP + FN + K.epsilon())  # Adicionar epsilon para evitar divisão por 0
    
    # Média de recall ponderada (por classe)
    weighted_recall = K.mean(recall_per_class)
    
    return weighted_recall




def running_precision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + K.epsilon())
    return precision


def running_precision_multi(y_true, y_pred):
    # y_pred = K.argmax(y_pred, axis=-1)
    # y_true = K.argmax(y_true, axis=-1)
    # 
    # TP = K.sum(K.cast(K.equal(y_true, y_pred), dtype='float32'))
    # TP_FP = K.sum(K.cast(K.greater_equal(y_pred, 0), dtype='float32'))
    # 
    # precision = TP / (TP_FP + K.epsilon())
    # return precision
    """
    Calcula a precisão para tarefas de classificação multi-classe usando rótulos one-hot.
    
    Parâmetros:
    - y_true: tensor de rótulos verdadeiros, codificados em one-hot
    - y_pred: tensor de previsões do modelo, em probabilidade ou logit

    Retorna:
    - precision: precisão global (ou média ponderada) entre todas as classes
    """
    # Converter previsões para binário (0 ou 1) - opção 1
    y_pred_binary = K.round(K.clip(y_pred, 0, 1))  # Caso o y_pred seja probabilístico
    
    # Verdadeiros Positivos (TP): predições corretas da classe positiva
    TP = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)), axis=0)
    
    # Falsos Positivos (FP): predições incorretas onde previu positivo, mas era negativo
    FP = K.sum(K.round(K.clip((1 - y_true) * y_pred_binary, 0, 1)), axis=0)
    
    # Precisão para cada classe
    precision_per_class = TP / (TP + FP + K.epsilon())  # Adicionar epsilon para evitar divisão por 0
    
    # Média de precisão ponderada (por classe)
    weighted_precision = K.mean(precision_per_class)
    
    return weighted_precision


def running_f1(y_true, y_pred):
    precision = running_precision(y_true, y_pred)
    recall = running_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def running_f1_multi(y_true, y_pred):
    """
    Calcula a métrica F1 para tarefas de classificação multi-classe usando rótulos one-hot.

    Parâmetros:
    - y_true: tensor de rótulos verdadeiros, codificados em one-hot
    - y_pred: tensor de previsões do modelo, em probabilidade ou logit

    Retorna:
    - f1: F1-score global (ou média ponderada) entre todas as classes
    """
    # Converter previsões para binário (0 ou 1)
    y_pred_binary = K.round(K.clip(y_pred, 0, 1))  # Caso o y_pred seja probabilístico

    # Verdadeiros Positivos (TP)
    TP = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)), axis=0)

    # Falsos Positivos (FP)
    FP = K.sum(K.round(K.clip((1 - y_true) * y_pred_binary, 0, 1)), axis=0)

    # Falsos Negativos (FN)
    FN = K.sum(K.round(K.clip(y_true * (1 - y_pred_binary), 0, 1)), axis=0)

    # Precisão para cada classe
    precision_per_class = TP / (TP + FP + K.epsilon())

    # Recall para cada classe
    recall_per_class = TP / (TP + FN + K.epsilon())

    # F1 para cada classe
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + K.epsilon())

    # Média de F1 ponderada (por classe)
    weighted_f1 = K.mean(f1_per_class)

    return weighted_f1











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
    Calcula a perda Soft Dice para tarefas de segmentação multi-classe.

    Parâmetros:
    - y_true: tensor de rótulos verdadeiros, codificados em one-hot
    - y_pred: tensor de previsões do modelo, em probabilidade (softmax)
    - smooth: valor pequeno adicionado para evitar divisão por zero

    Retorna:
    - perda: Soft Dice loss média entre todas as classes
    """
    # Aplicar a função softmax nas previsões para obter probabilidades
    y_pred = K.softmax(y_pred)

    # Flatten os tensores para calcular a Dice
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Cálculo da interseção
    intersection = K.sum(y_true_f * y_pred_f)

    # Cálculo do coeficiente de Dice
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Retornar a perda como 1 - Dice
    return 1 - dice