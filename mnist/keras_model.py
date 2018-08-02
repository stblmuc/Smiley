# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.keras.api.keras

import numpy as np
from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop
# from tensorflow.contrib.keras.api.keras.utils import np_utils
from tensorflow.contrib.keras.api.keras import backend as K

import h5py
import image


class Keras:
    def __init__(self, pathToH5py, numCats=10):

        img_width, img_height = 28, 28

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 1)

        model_m = Sequential()
        model_m.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model_m.add(Activation('relu'))
        model_m.add(MaxPooling2D(pool_size=(2, 2)))

        model_m.add(Conv2D(32, (3, 3)))
        model_m.add(Activation('relu'))
        model_m.add(MaxPooling2D(pool_size=(2, 2)))

        model_m.add(Conv2D(64, (3, 3)))
        model_m.add(Activation('relu'))
        model_m.add(MaxPooling2D(pool_size=(2, 2)))

        model_m.add(Flatten())
        model_m.add(Dense(64))
        model_m.add(Activation('relu'))
        model_m.add(Dropout(0.5))
        model_m.add(Dense(numCats))
        model_m.add(Activation('sigmoid'))

        model_m.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])

        model_m.load_weights(pathToH5py)

        self.model = model_m

    # Create model of CNN with slim api
    def predict(self, inputs):

        # print(inputs.tolist())
        preds = self.model.predict(np.array(inputs))
        preds = preds * 100.
        # print(preds)
        return preds
