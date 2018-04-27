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
from keras.callbacks import EarlyStopping


EMBEDDING_SIZE = 64
MAX_LEN = 50
BATCH_SIZE = 128

current_dir = os.getcwd()

data_dirs = dict()
data_dirs['train_data'] = os.path.join(current_dir, 'data/train.csv')
data_dirs['test_data'] = os.path.join(current_dir, 'data/test.csv')
data_dirs['pred_result'] = os.path.join(current_dir, 'data/preds.csv')


def train_model():
    print('[1] Extracting features from train data...')
    txt_seq, target, word_dict = drxan.data_helper.extract_features(data_dirs['train_data'], target='deal_probability', word_dict=None)
    x_train = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[2] Creating DNN model...')
    model = drxan.models.create_cnn(seq_length=MAX_LEN, word_num=len(word_dict), embedding_dim=EMBEDDING_SIZE)
    print(model.summary())
    model.compile(optimizer='adam', loss=drxan.loss.rmse)

    print('[3] Training model,find the best epoch...')
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    train_hist = model.fit(x_train, target, batch_size=BATCH_SIZE, epochs=1000, validation_split=0.25, callbacks=[early_stop])
    best_epoch = len(train_hist.epoch)

    print('[4] Training the final model...')
    train_hist = model.fit(x_train, target, batch_size=BATCH_SIZE, epochs=best_epoch)

    return word_dict, model


def predict(word_dict, model):
    print('[5] Extracting features from test data...')
    txt_seq, pred_items = drxan.data_helper.extract_features(data_dirs['train_data'], target=None, word_dict=word_dict)
    x_test = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[6] Predicting the test data...')
    preds = model.predict(x_test)
    pred_items['deal_probability'] = preds.reshape(-1)
    pred_items.to_csv(data_dirs['pred_result'], index=False)


if __name__ == '__main__':
    word_dict, model = train_model()
    predict(word_dict, model)