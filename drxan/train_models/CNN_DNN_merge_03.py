#encoding:utf-8
"""
@project : Avito-Demand-Prediction-Challenge
@file : CNN_DNN_merge
@author : Drxan
@create_time : 18-4-30 下午8:17
"""
import drxan
from keras import optimizers, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import math
from keras.models import load_model

# 词嵌入维度
EMBEDDING_SIZE = 32
# 文本序列最大长度
MAX_LEN = 50
# 最小词频
MIN_FREQ = 3
# 训练深度模型时用于参数更新的样本数量
BATCH_SIZE = 128


def train(data_dirs):
    print('[1] Converting train texts to TF-IDF vectors...')
    x_train_tfidf, targets, tfidf_transformer = drxan.data_helper.get_tfidf(data_dirs['train_data'],
                                                                            tfidf_transformer=None,
                                                                            train=True)
    print('[2] Converting train texts into sequences...')
    txt_seq, _, word_dict = drxan.data_helper.extract_features(data_dirs['train_data'],
                                                               target='deal_probability',
                                                               word_dict=None,
                                                               min_freq=MIN_FREQ)
    x_train_seq = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[3] Creating model...')
    model = drxan.models.create_cnn_dense(seq_length=MAX_LEN,
                                          word_num=len(word_dict),
                                          embedding_dim=EMBEDDING_SIZE,
                                          x_dim=x_train_tfidf.shape[1],
                                          sparse=True)

    print(model.summary())
    model.compile(optimizer=optimizers.Adadelta(lr=1.0), loss=losses.mse)
    print('[3] Training model,find the best model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    check_point = ModelCheckpoint(data_dirs['model_path'], 'val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1 / math.e, verbose=1, patience=3, min_lr=0.0001)
    tensor_board = TensorBoard(log_dir=data_dirs['logs_path'], histogram_freq=0, write_graph=True, write_images=True)
    train_hist = model.fit([x_train_seq, x_train_tfidf],
                           targets, batch_size=BATCH_SIZE,
                           epochs=1000,
                           validation_split=0.25,
                           callbacks=[early_stop, check_point, reduce_lr, tensor_board])
    return word_dict, tfidf_transformer


def predict(word_dict, tfidf_transformer, data_dirs): 
    print('[4] Converting test texts to sequences...')
    txt_seq, pred_items = drxan.data_helper.extract_features(data_dirs['test_data'], target=None, word_dict=word_dict)
    x_test_seq = drxan.data_helper.pad_sequences(txt_seq, max_len=MAX_LEN)

    print('[5] Converting test texts to TF-IDF vectors...')
    x_test_tfidf, _, _ = drxan.data_helper.get_tfidf(data_dirs['test_data'],
                                                     tfidf_transformer=tfidf_transformer,
                                                     train=False)

    print('[6] Predicting the test data...')
    model = load_model(data_dirs['model_path'])
    preds = model.predict([x_test_seq, x_test_tfidf])
    pred_items['deal_probability'] = preds.reshape(-1)
    pred_items.to_csv(data_dirs['pred_result'], index=False)


def train_predict(data_dirs):
    word_dict, tfidf_transformer = train(data_dirs)
    predict(word_dict, tfidf_transformer, data_dirs)