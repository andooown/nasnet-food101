import os
import subprocess

# 利用可能な GPU の数を取得
res = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], stdout=subprocess.PIPE)
GPU_COUNT = len(res.stdout.splitlines())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(GPU_COUNT)))

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pickle
import glob

from model import NASNetMobile
from multi_gpu_model import MultiGPUModel
from sequence import ImageSequence
from utils import check_directory

DATA_DIR = 'data'       # データセットがあるディレクトリ
TFLOG_DIR = 'tflog'     # TensorFlow のログの保存先ディレクトリ
RESULT_DIR = 'result'   # 結果の保存先ディレクトリ

INPUT_SIZE = (224, 224, 3)  # 入力画像のサイズ (height x width x channel)
CLASS_COUNT = 101           # クラス数
EPOCH_COUNT = 1000          # エポック数
BATCH_SIZE = 50 * GPU_COUNT # バッチサイズ


def load_data():
    x, y = [], []
    for i, class_dir in enumerate(sorted(glob.glob(os.path.join(DATA_DIR, '*')))):
        for file_path in glob.glob(os.path.join(class_dir, '*.jpg')):
            x.append(file_path)
            y.append(i)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
    train_gen = ImageSequence(map(tuple, zip(x_train, y_train)),
                              target_size=INPUT_SIZE[1::-1],
                              num_class=CLASS_COUNT,
                              batch_size=BATCH_SIZE)
    valid_gen = ImageSequence(map(tuple, zip(x_valid, y_valid)),
                              target_size=INPUT_SIZE[1::-1],
                              num_class=CLASS_COUNT,
                              batch_size=BATCH_SIZE,
                              transform=False)

    return train_gen, valid_gen


def build_model():
    if GPU_COUNT > 1:
        with tf.device("/cpu:0"):
            model = NASNetMobile(INPUT_SIZE, CLASS_COUNT)
        model = MultiGPUModel(model, gpus=GPU_COUNT)
    else:
        model = NASNetMobile(INPUT_SIZE, CLASS_COUNT)

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return model


def schedule(epoch):
    if epoch < 12:
        return .01
    elif epoch < 24:
        return .002
    else:
        return .0004


def main():
    np.random.seed(0)

    check_directory(RESULT_DIR)
    check_directory(TFLOG_DIR)

    print('Build model.....', end='')
    model = build_model()
    print('Done!')

    print('Load file list in dataset.....', end='')
    train_gen, valid_gen = load_data()
    print('Done!')

    fpath = os.path.join(RESULT_DIR,
                         'model.{epoch:04d}-{loss:.5f}-{acc:.3f}-{val_loss:.5f}-{val_acc:.3f}.h5')
    checkpoint = ModelCheckpoint(fpath, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    lr_scheduler = LearningRateScheduler(schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    tensor_board = TensorBoard(log_dir=TFLOG_DIR, histogram_freq=0, batch_size=BATCH_SIZE,
                              write_graph=False, write_grads=True, write_images=True)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(train_gen),
                                  epochs=EPOCH_COUNT,
                                  validation_data=valid_gen,
                                  validation_steps=len(valid_gen),
                                  callbacks=[checkpoint, lr_scheduler, early_stopping, tensor_board],
                                  workers=20,
                                  max_queue_size=500,
                                  use_multiprocessing=True,
                                  verbose=1)

    print('Save training result.....', end='')
    with open(os.path.join(RESULT_DIR, 'history.pickle'), 'wb') as f:
        pickle.dump(history.history, f)
    model.save(os.path.join(RESULT_DIR, 'model.h5'))
    print('Done!')


if __name__ == '__main__':
    main()
