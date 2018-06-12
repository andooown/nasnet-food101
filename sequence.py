# -*- coding: utf-8 -*-

from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import numpy as np
import math
from utils import preprocess


class ImageSequence(Sequence):

    def __init__(self, pairs, target_size, num_class, batch_size=1, transform=True):
        files, labels = map(list, zip(*pairs))
        self.x = np.array(files)
        self.y = to_categorical(labels, num_classes=num_class)

        self._shuffle_data()

        self.target_size = target_size
        self.batch_size = batch_size
        self.transform = transform

    def __getitem__(self, item):
        # バッチサイズ分だけデータを取り出す
        x_batch = self.x[item * self.batch_size:(item + 1) * self.batch_size]
        y_batch = self.y[item * self.batch_size:(item + 1) * self.batch_size]

        x_batch = [preprocess(file, self.target_size, self.transform) for file in x_batch]

        return np.array(x_batch), np.array(y_batch)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        self._shuffle_data()

    def _shuffle_data(self):
        index = np.random.permutation(self.x.shape[0])
        self.x = self.x[index]
        self.y = self.y[index]


class ImageOnlySequence(Sequence):

    def __init__(self, files, target_size, batch_size=1, transform=True):
        self.x = files

        self._shuffle_data()

        self.target_size = target_size
        self.batch_size = batch_size
        self.transform = transform

    def __getitem__(self, item):
        # バッチサイズ分だけデータを取り出す
        x_batch = self.x[item * self.batch_size:(item + 1) * self.batch_size]
        # 前処理
        x_batch = [preprocess(file, self.target_size, self.transform) for file in x_batch]

        return np.array(x_batch)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        self._shuffle_data()

    def _shuffle_data(self):
        index = np.random.permutation(self.x.shape[0])
        self.x = self.x[index]
