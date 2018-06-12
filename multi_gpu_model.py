# -*- coding: utf-8 -*-

from keras import Model
from keras.utils import multi_gpu_model


class MultiGPUModel(Model):

    def __init__(self, model, gpus):
        parallel_model = multi_gpu_model(model, gpus=gpus)
        self.__dict__.update(parallel_model.__dict__)
        self.model = model

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.model, attrname)

        return super(MultiGPUModel, self).__getattribute__(attrname)
