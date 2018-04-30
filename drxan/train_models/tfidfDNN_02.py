#encoding:utf-8
"""
@project : Avito-Demand-Prediction-Challenge
@file : tfidfDNN_02
@author : Drxan
@create_time : 18-4-29 下午11:15
"""

import drxan
from keras import optimizers, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import math
from keras.models import load_model


BATCH_SIZE = 128


def train(data_dirs):
    print('[1] Converting train texts to TF-IDF vectors...')
    x_train, targets, tfidf_transformer = drxan.data_helper.get_tfidf(data_dirs['train_data'],
                                                                      tfidf_transformer=None,
                                                                      train=True)
    print('[2] Creating model...')
    model = drxan.models.create_multi_dnn(x_train.shape[1])
    print(model.summary())
    model.compile(optimizer=optimizers.Adadelta(), loss=losses.mse)

    print('[3] Training model,find the best model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    check_point = ModelCheckpoint(data_dirs['model_path'], 'val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1 / math.e, verbose=1, patience=10, min_lr=0.0001)
    tensor_board = TensorBoard(log_dir=data_dirs['logs_path'], histogram_freq=0, write_graph=True, write_images=True)
    train_hist = model.fit(x_train, targets, batch_size=BATCH_SIZE, epochs=1000, validation_split=0.25,
                           callbacks=[early_stop, check_point, reduce_lr, tensor_board])
    return tfidf_transformer


def predict(tfidf_transformer, data_dirs):
    print('[4] Converting test texts to TF-IDF vectors...')
    x_test, pred_items, _ = drxan.data_helper.get_tfidf(data_dirs['test_data'],
                                                        tfidf_transformer=tfidf_transformer,
                                                        train=False)

    print('[5] Predicting the test data...')
    model = load_model(data_dirs['model_path'])
    preds = model.predict(x_test)
    pred_items['deal_probability'] = preds.reshape(-1)
    pred_items.to_csv(data_dirs['pred_result'], index=False)


def train_predict(data_dirs):
    tfidf_transformer = train(data_dirs)
    predict(tfidf_transformer, data_dirs)
