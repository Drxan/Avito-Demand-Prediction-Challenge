#encoding:utf-8
"""
@project : Avito-Demand-Prediction-Challenge
@file : textCNN1
@author : Drxan
@create_time : 18-4-29 上午8:25
"""
import pandas as pd
import numpy as np
import os
import drxan
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras import losses
from keras import optimizers


# 词嵌入维度
EMBEDDING_SIZE = 64
# 文本序列最大长度
MAX_LEN = 50
# 最小词频
MIN_FREQ = 3
# 训练深度模型时用于参数更新的样本数量
BATCH_SIZE = 128


def train_model(data_dirs):
    print('[1] Extracting features from train data...')
    txt_seq, target, word_dict = drxan.data_helper.extract_features(data_dirs['train_data'],
                                                                    target='deal_probability',
                                                                    word_dict=None,
                                                                    min_freq=MIN_FREQ)
    x_train = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[2] Creating DNN model...')
    model = drxan.models.create_cnn(seq_length=MAX_LEN, word_num=len(word_dict), embedding_dim=EMBEDDING_SIZE)
    print(model.summary())
    model.compile(optimizer=optimizers.Adadelta(), loss=losses.mse)

    print('[3] Training model,find the best model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    check_point = ModelCheckpoint(data_dirs['model_path'], 'val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1 / math.e, verbose=1, patience=10, min_lr=0.0001)
    tensor_board = TensorBoard(log_dir=data_dirs['logs_path'], histogram_freq=0, write_graph=True, write_images=True)
    train_hist = model.fit(x_train, target, batch_size=BATCH_SIZE, epochs=1000, validation_split=0.25,
                           callbacks=[early_stop, check_point, reduce_lr, tensor_board])
    return word_dict


def predict(word_dict, data_dirs):
    print('[4] Extracting features from test data...')
    txt_seq, pred_items = drxan.data_helper.extract_features(data_dirs['test_data'], target=None, word_dict=word_dict)
    x_test = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[5] Predicting the test data...')
    model = load_model(data_dirs['model_path'])
    preds = model.predict(x_test)
    pred_items['deal_probability'] = preds.reshape(-1)
    pred_items.to_csv(data_dirs['pred_result'], index=False)


def train_predict(data_dirs):
    word_dict = train_model(data_dirs)
    predict(word_dict, data_dirs)

