import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend as K

class Unet():

    def __init__(self, bands, optimizer=None, loss='MeanSquaredError', metrics=[], multiclass=False) -> None:
        self.bands = bands
        self.optimizer = optimizer if optimizer != None else 'SGD'
        self.metrics = metrics,
        self.loss = loss
        self.multiclass = multiclass

    def _convBlock(self, input_tensor, n_filters):
        x = tf.keras.layers.Conv2D(n_filters, (3,3), padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('leaky_relu')(x)

        x = tf.keras.layers.Conv2D(n_filters, (3,3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('leaky_relu')(x)

        return x

    def encoderBlock(self, input_tensor, num_filters):
        x_encoded = self._convBlock(input_tensor, num_filters)
        x_polling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_encoded)
        return x_encoded, x_polling

    def decoderBlock(self, input_tensor, concat_tensor, num_filters):
        decoder = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('leaky_relu')(decoder)

        decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('leaky_relu')(decoder)

        decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('leaky_relu')(decoder)

        return decoder

    def getModel(self, n_classes):
        # Entrada para duas escalas: 512x512 e 256x256
        input_512 = tf.keras.layers.Input(shape=[512, 512, len(self.bands)], name="input_512")
        input_256 = tf.keras.layers.Input(shape=[256, 256, len(self.bands)], name="input_256")

        # Encoder para o input 512x512
        encoder0_512, encoder0_pool_512 = self.encoderBlock(input_512, 32)  # 256
        encoder1_512, encoder1_pool_512 = self.encoderBlock(encoder0_pool_512, 64)  # 128
        encoder2_512, encoder2_pool_512 = self.encoderBlock(encoder1_pool_512, 128)  # 64
        encoder3_512, encoder3_pool_512 = self.encoderBlock(encoder2_pool_512, 256)  # 32
        encoder4_512, encoder4_pool_512 = self.encoderBlock(encoder3_pool_512, 512)  # 16

        # Encoder para o input 256x256 (processamento em escala menor)
        encoder0_256, encoder0_pool_256 = self.encoderBlock(input_256, 32)  # 128
        encoder1_256, encoder1_pool_256 = self.encoderBlock(encoder0_pool_256, 64)  # 64
        encoder2_256, encoder2_pool_256 = self.encoderBlock(encoder1_pool_256, 128)  # 32
        encoder3_256, encoder3_pool_256 = self.encoderBlock(encoder2_pool_256, 256)  # 16

        # Combinar as saídas dos encoders de diferentes escalas (concatenar as features)
        combined_encoding = tf.keras.layers.concatenate([encoder4_512, encoder3_256], axis=-1)

        # Bloco central (parte mais baixa da U-Net)
        center = self._convBlock(combined_encoding, 1024)

        # Decoder
        decoder4 = self.decoderBlock(center, encoder4_512, 512)  # 32
        decoder3 = self.decoderBlock(decoder4, encoder3_512, 256)  # 64
        decoder2 = self.decoderBlock(decoder3, encoder2_512, 128)  # 128
        decoder1 = self.decoderBlock(decoder2, encoder1_512, 64)  # 256
        decoder0 = self.decoderBlock(decoder1, encoder0_512, 32)  # 512

        # Saída
        if self.multiclass:
            outputs = tf.keras.layers.Conv2D(n_classes, kernel_size=(1, 1), activation='softmax')(decoder0)
        else:
            outputs = tf.keras.layers.Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid')(decoder0)

        model = tf.keras.models.Model(inputs=[input_512, input_256], outputs=[outputs])

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics[0])

        return model
