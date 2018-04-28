# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 22:21
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : train.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os
import drxan
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from keras.models import load_model


EMBEDDING_SIZE = 64
MAX_LEN = 50
BATCH_SIZE = 128

current_dir = os.getcwd()

data_dirs = dict()
data_dirs['train_data'] = os.path.join(current_dir, 'data/train.csv')
data_dirs['test_data'] = os.path.join(current_dir, 'data/test.csv')
data_dirs['pred_result'] = os.path.join(current_dir, 'data/preds.csv')
data_dirs['model_path'] = os.path.join(current_dir, 'data/model_data/model.h5')
# data_dirs['model_path'] = os.path.join(current_dir, 'data/model_data/model-ep{epoch:03d}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5')
data_dirs['logs_path'] = os.path.join(current_dir, 'data/logs/')


def train_model():
    print('[1] Extracting features from train data...')
    txt_seq, target, word_dict = drxan.data_helper.extract_features(data_dirs['train_data'],
                                                                    target='deal_probability',
                                                                    word_dict=None,
                                                                    min_freq=10)
    x_train = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[2] Creating DNN model...')
    model = drxan.models.create_cnn(seq_length=MAX_LEN, word_num=len(word_dict), embedding_dim=EMBEDDING_SIZE)
    print(model.summary())
    model.compile(optimizer='adam', loss=drxan.loss.rmse)

    print('[3] Training model,find the best model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    check_point = ModelCheckpoint(data_dirs['model_path'], 'val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1 / math.e, verbose=1, patience=10, min_lr=0.0001)
    tensor_board = TensorBoard(log_dir=data_dirs['logs_path'], histogram_freq=0, write_graph=True, write_images=True)
    train_hist = model.fit(x_train, target, batch_size=BATCH_SIZE, epochs=1000, validation_split=0.25,
                           callbacks=[early_stop, check_point, reduce_lr, tensor_board])
    return word_dict


def predict(word_dict):
    print('[5] Extracting features from test data...')
    txt_seq, pred_items = drxan.data_helper.extract_features(data_dirs['test_data'], target=None, word_dict=word_dict)
    x_test = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[6] Predicting the test data...')
    model = load_model(data_dirs['model_path'], custom_objects={'root_mean_squared_error': drxan.loss.rmse})
    preds = model.predict(x_test)
    pred_items['deal_probability'] = preds.reshape(-1)
    pred_items.to_csv(data_dirs['pred_result'], index=False)


if __name__ == '__main__':
    word_dict = train_model()
    predict(word_dict)