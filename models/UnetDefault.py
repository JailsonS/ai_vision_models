
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import backend as  K

class Unet():

    def __init__(self, bands, optimizer=None, loss='MeanSquaredError', metrics=['RootMeanSquaredError']) -> None:
        self.bands = bands
        self.optimizer = optimizer if optimizer != None else 'SGD'
        self.metrics = metrics,
        self.loss = loss

    def _convBlock(self, input_tensor, n_filters):

        x = tf.keras.layers.Conv2D(n_filters,(3,3),padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(n_filters,(3,3),padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def encoderBlock(self, input_tensor, num_filters):
        # encoding
        x_encoded = self._convBlock(input_tensor, num_filters)

        x_polling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_encoded)

        return x_encoded, x_polling

    def decoderBlock(self, input_tensor, concat_tensor, num_filters):

        decoder = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=(2,2), strides=(2,2), padding='same')(input_tensor)
        decoder = tf.keras.layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)

        decoder = tf.keras.layers.Conv2D(num_filters, (3,3),padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)

        decoder = tf.keras.layers.Conv2D(num_filters, (3,3), padding='same')(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation('relu')(decoder)


        return decoder

    def getModel(self, n_classes):

        inputs = tf.keras.layers.Input(shape=[None, None, len(self.bands)]) # 256 size

        encoder0, encoder0_pool  = self.encoderBlock(inputs, 32) # 128
        encoder1, encoder1_pool  = self.encoderBlock(encoder0_pool, 64) # 64
        encoder2, encoder2_pool = self.encoderBlock(encoder1_pool, 128) # 32
        encoder3, encoder3_pool = self.encoderBlock(encoder2_pool, 256) # 16
        encoder4, encoder4_pool = self.encoderBlock(encoder3_pool, 512) # 8

        center = self._convBlock(encoder4_pool, 1024) # center

        decoder4 = self.decoderBlock(center, encoder4, 512) # 16
        decoder3 = self.decoderBlock(decoder4, encoder3, 256) # 32
        decoder2 = self.decoderBlock(decoder3, encoder2, 128) # 64
        decoder1 = self.decoderBlock(decoder2, encoder1, 64) # 128
        decoder0 = self.decoderBlock(decoder1, encoder0, 32) # 256

        outputs = tf.keras.layers.Conv2D(n_classes, kernel_size=(1, 1),activation='sigmoid')(decoder0)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
            #metrics=[tf.keras.metrics.get(metric) for metric in self.metrics]
        )

        return model