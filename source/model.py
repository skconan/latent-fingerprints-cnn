
"""
    File name: model.py
    Author: skconan
    Date created: 2019/10/13
    Python Version: 3.7
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from utilities import *

from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.callbacks import ModelCheckpoint
from mycallback import MyCallback
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras import backend as K
from fft import *

def custom_loss(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
    # np.count_nonzero(y_true)

    # return (y_true * y_pred) / (K.abs(y_true) *K.abs(y_pred))
    # return K.abs(y_true - y_pred)

class Autoencoder:
    def __init__(self, model_dir, pred_dir, img_rows=64, img_cols=64, channels=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels

        self.model_dir = model_dir
        self.pred_dir = pred_dir

        self.encoding_dim = 64

        adm = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=True)
        # sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
        #sgd = SGD(lr=0.01, decay=1e-7, momentum=0.1, nesterov=True)
        #sgd = SGD(lr=0.01, decay=1e-3, momentum=0.8, nesterov=True)
        rms = RMSprop(lr=0.001, rho=0.9)
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(
            # loss='mse', optimizer=rms, metrics=['acc'])
            # loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['acc'])
            loss='mae', optimizer=rms, metrics=['acc'])
            # loss=custom_loss, optimizer=adm, metrics=['acc'])
        # loss='mse', optimizer=sgd, metrics=['acc'])

        self.autoencoder_model.summary()

    def downsampling(self, input_layer, number_of_filter, dropout=False, maxpooling=True):
        x = Conv2D(number_of_filter, (3, 3), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(0.5)(x)
        if maxpooling:
            x = MaxPooling2D((2, 2))(x)

        return x

    def upsampling(self, input_layer, number_of_filter,dropout=False):
        x = Conv2D(number_of_filter, (3, 3), activation='relu',
                   padding='same')(input_layer)
        if dropout:
            x = Dropout(0.5)(x)
        x = UpSampling2D((2, 2))(x)
        return x

    def build_model(self):
        input_layer = Input(
            shape=(self.img_rows, self.img_cols, self.channels))

        x = self.downsampling(input_layer, self.encoding_dim, dropout=False)
        # x = self.downsampling(x, 128)
        x = self.downsampling(x, self.encoding_dim * 2, dropout=False)
        # x = self.downsampling(x, self.encoding_dim * 4, dropout=False)

        x = self.downsampling(x, self.encoding_dim * 4, dropout=False, maxpooling=False)

        # x = self.upsampling(x, self.encoding_dim * 4, dropout=False)
        x = self.upsampling(x, self.encoding_dim * 2, dropout=False)
        # x = self.upsampling(x, 128)
        x = self.upsampling(x, self.encoding_dim, dropout=False)

        output_layer = Conv2D(
            self.channels, (3, 3), activation='sigmoid', padding='same')(x)

        return Model(input_layer, output_layer)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')

        filepath = self.model_dir + "/model-{epoch:03d}-{val_loss:.4f}.hdf5"
        print(self.model_dir)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='min')
        # print(checkpoint)
        my_callback = MyCallback(x_val, y_val, self.model_dir, self.pred_dir)
        callbacks_list = [
            checkpoint,
            my_callback
        ]
        # print("x_train",x_train)
        # print("y_train",y_train)
        # print("x_val",x_val)
        # print("y_val",y_val)
        # print("batch_size",batch_size)
        # print("epochs",epochs)
        # end_step = np.ceil(1.0 * len(x_train) / batch_size).astype(np.int32) * epochs
        # print("End step:",end_step)
        # pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        #     initial_sparsity=0.0, final_sparsity=0.5,
        #     begin_step=0, end_step=end_step, frequency=100
        # )

        # self.autoencoder_model = tfmot.sparsity.keras.prune_low_magnitude(self.autoencoder_model, pruning_schedule=pruning_schedule)

        # history = self.autoencoder_model.fit(x_train, y_train,
        
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=callbacks_list
                                             )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig(self.pred_dir + "/graph.jpg")
        plt.close()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        # preds = freq2spatial(preds)
        return preds

    def save(self, path):
        self.autoencoder_model.save(path)
