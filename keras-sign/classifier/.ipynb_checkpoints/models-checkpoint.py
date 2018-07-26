import os, sys, io

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, LeakyReLU
from keras.layers.normalization import BatchNormalization

class Layers:

    model = None

    def __init__(self, **kwargs):
        self.name = ""
        self._numOut = kwargs.pop("numOut", 25)
        self._inShape = [28, 28, 1]

    def compile(self, lr=1e-4):
        assert (self.model is not None), "model is not initialized"
        opt = keras.optimizers.Adam(lr=lr)
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=opt,
            metrics=['accuracy']
        )
    
    def __str__(self):
        assert (self.model is not None), "model is not initialized"
        self.model.summary()
        return self.name

    @property
    def numOut(self):
        return self._numOut

    @property
    def inShape(self):
        return self._inShape
    
    @property
    def training(self):
        return self._training

    @numOut.setter
    def numOut(self, x):
        assert isinstance(x, int) and (x > 0), "numOut needs to be a positive integer"
        self._numOut = x

    @inShape.setter
    def inShape(self, x):
        assert isinstance(x, list), "inShape must be a list, not {0}".format(type(x))
        self._inShape = x

    @training.setter
    def training(self, state):
        assert isinstance(state, bool), "training needs to be a bool, not {0}".format(type(state))
        self._training = state


class SunGod(Layers):
    def __init__(self, **kwargs):
        Layers.__init__(self, **kwargs)
        self.name = "SunGod"
        
        self.__build_model()
        self.compile()
        
        self.model.summary()

    def __build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=tuple(self.inShape)))
        self.model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(alpha=0.15))
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1)))
        self.model.add(Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(alpha=0.15))
        self.model.add(Dropout(0.1))
        
        self.model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1)))
        self.model.add(Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(alpha=0.15))
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(3,3)))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.numOut, activation='softmax'))