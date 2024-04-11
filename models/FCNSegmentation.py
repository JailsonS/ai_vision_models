import tensorflow as tf

def model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder (VGG-like)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    # Bottleneck
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(x)
    
    # Upsampling
    x = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
    
    model = tf.keras.Model(inputs, x)
    
    return model


import numpy as np
from collections import defaultdict
import random

chips = [
    {'classes': ['classe1', 'classe2'], 'areas': [5, 4]}, 
    {'classes': ['classe1', 'classe2', 'classe3'], 'areas': [5, 4, 3]}, 
]

# Agrupando chips por combinações de classes
chip_groups = defaultdict(list)

for i, chip in enumerate(chips):
    key = tuple(sorted(chip['classes']))  # chave para identificar a combinação de classes
    chip_groups[key].append(i)  # adicionando o índice do chip ao grupo correspondente

# Realizando a amostragem estratificada dentro de cada grupo
proporcao_amostras = 0.2  # proporção de amostras desejada

amostras_selecionadas = []

for group_key, group_indices in chip_groups.items():
    num_amostras_grupo = int(len(group_indices) * proporcao_amostras)
    amostras_grupo = random.sample(group_indices, num_amostras_grupo)
    amostras_selecionadas.extend(amostras_grupo)

# 'amostras_selecionadas' agora contém os índices dos chips selecionados para amostragem

# Você pode então usar esses índices para acessar os chips específicos na sua lista 'chips'
# e realizar outras operações, como a extração das imagens correspondentes aos chips selecionados.


# 60% F
# 10% -> F P 
# 5% -> F P A
# 2% -> F P A W
# 2% -> A F P