__author__ = 'moonkey'

from keras import models, layers
import logging
import numpy as np
# from src.data_util.synth_prepare import SynthGen

import keras.backend as K
import tensorflow as tf

def squeeze_dim(x, axis=-1):
    return K.squeeze(x, axis=axis)


def squeeze_dim_shape(input_shape, axis=-1):
    if axis == -1:
        return input_shape[:axis]
    else:
        return input_shape[:axis] + input_shape[axis:]


class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()

    """

    def __init__(self, input_tensor):
        self._build_network(input_tensor)

    def _build_network(self, input_tensor):
        """
        https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
        :return:
        """
        model = models.Sequential()

        # width is max of all: 785(test)
        input_layer = layers.InputLayer(input_shape=(1, 32, None))

        # if input_tensor is not None:
        input_layer.set_input(input_tensor=input_tensor)

        model.add(input_layer)

        model.add(layers.Lambda(lambda x: (x - 128.0) / 128.0))

        model.add(layers.Convolution2D(64, 3, 3, subsample=(1, 1),
                                       border_mode='same',
                                       dim_ordering='th'))
        model.add(layers.Activation('relu'))

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                        dim_ordering='th'))
        # print 'pool1', model.output_shape

        model.add(layers.Convolution2D(128, 3, 3, subsample=(1, 1),
                                        border_mode='same',
                                        dim_ordering='th'))
        model.add(layers.Activation('relu'))

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                        dim_ordering='th'))
        # print 'pool2', model.output_shape

        model.add(layers.Convolution2D(256, 3, 3, subsample=(1, 1),
                                       border_mode='same',
                                       dim_ordering='th'))
        # https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.Activation('relu'))

        model.add(layers.Convolution2D(256, 3, 3, subsample=(1, 1),
                                       border_mode='same',
                                       dim_ordering='th'))
        model.add(layers.Activation('relu'))

        model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1),
                                        dim_ordering='th'))
        # print 'pool3', model.output_shape

        model.add(layers.Convolution2D(512, 3, 3, subsample=(1, 1),
                                       border_mode='same',
                                       dim_ordering='th'))
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.Activation('relu'))

        model.add(layers.Convolution2D(512, 3, 3, subsample=(1, 1),
                                       border_mode='same',
                                       dim_ordering='th'))
        model.add(layers.Activation('relu'))

        model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1),
                                        dim_ordering='th'))
        # print 'pool4', model.output_shape

        model.add(layers.Convolution2D(512, 2, 2, subsample=(1, 1),
                                       border_mode='valid',
                                       dim_ordering='th'))
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.Activation('relu'))

        # ch, r, c -> c, r, ch
        model.add(layers.Permute((3, 1, 2)))

        # model.add(layers.Reshape((-1, 512)))
        # model.add(layers.wrappers.TimeDistributed(layers.Flatten()))
        model.add(layers.Lambda(squeeze_dim, output_shape=squeeze_dim_shape))

        self.model = model

    def tf_output(self):
        # if self.input_tensor is not None:
        return self.model.output

    def __call__(self, input_tensor):
        return self.model(input_tensor)

    def save(self):
        pass


