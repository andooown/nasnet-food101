# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.regularizers import l2
import keras.applications.nasnet


def NASNetMobile(input_shape, num_classes):
    nasnet = keras.applications.nasnet.NASNetMobile(input_shape=input_shape, include_top=False, weights='imagenet')

    x = nasnet.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(rate=0.5)(x)
    output = Dense(num_classes, activation='softmax',
                   kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005))(x)

    model = Model(inputs=nasnet.input, outputs=output)

    return model
