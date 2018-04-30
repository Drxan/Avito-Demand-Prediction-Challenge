# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 14:41
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : models.py
# @Software: PyCharm


from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Dense
from keras.layers.merge import concatenate
from keras import backend as K


def create_cnn(seq_length, word_num, embedding_dim):
    """
    将原始文本数值序列化，利用Text-CNN网络结构提取文本语义信息
    :param seq_length:
    :param word_num:
    :param embedding_dim:
    :return:
    """
    x_input = Input(shape=(seq_length,))
    embed = Embedding(input_shape=(seq_length,), input_dim=word_num+1, output_dim=embedding_dim)(x_input)
    hidden = Conv1D(filters=128, kernel_size=3, activation='relu')(embed)
    # hidden = MaxPooling1D(pool_size=2)(hidden)
    # hidden = Conv1D(filters=64, kernel_size=3, activation='relu')(hidden)
    hidden = MaxPooling1D(pool_size=K.get_variable_shape(hidden)[1])(hidden)
    hidden = Flatten()(hidden)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(units=64, activation='tanh')(hidden)
    hidden = Dense(units=64, activation='tanh')(hidden)
    output = Dense(units=1, activation='sigmoid')(hidden)

    model = Model(inputs=x_input, outputs=output)
    return model


def create_dnn(x_dim):
    """
    将原始文本（title,description)转换成TF-IDF向量,直接构造全连接的深度网络
    :param x_dim:
    :return:
    """
    x_input = Input(shape=(x_dim,), sparse=True)
    hidden = Dense(units=32, activation='relu')(x_input)
    hidden = Dense(units=64, activation='relu')(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)
    output = Dense(units=1, activation='sigmoid')(hidden)

    model = Model(inputs=x_input, outputs=output)
    return model


def create_multi_dnn(x_dim, sparse=True):
    """
    将原始文本（title,description)转换成TF-IDF向量，利用多个子网络从原始tf-idf向量中学习不同的信息，并合并成文本的语义信息
    :param x_dim:
    :return:
    """
    x_input = Input(shape=(x_dim,), sparse=sparse)

    hidden1 = Dense(units=16, activation='relu')(x_input)
    hidden1 = Dense(units=64, activation='relu')(hidden1)
    hidden1 = Dense(units=64, activation='relu')(hidden1)

    hidden2 = Dense(units=16, activation='relu')(x_input)
    hidden2 = Dense(units=64, activation='relu')(hidden2)
    hidden2 = Dense(units=64, activation='relu')(hidden2)

    hidden3 = Dense(units=16, activation='relu')(x_input)
    hidden3 = Dense(units=64, activation='relu')(hidden3)
    hidden3 = Dense(units=64, activation='relu')(hidden3)

    hidden4 = Dense(units=16, activation='relu')(x_input)
    hidden4 = Dense(units=64, activation='relu')(hidden4)
    hidden4 = Dense(units=64, activation='relu')(hidden4)

    hidden = concatenate([hidden1, hidden2, hidden3, hidden4])

    hidden = BatchNormalization()(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)

    output = Dense(units=1, activation='sigmoid')(hidden)

    model = Model(inputs=x_input, outputs=output)
    return model


def create_cnn_dense(seq_length, word_num, embedding_dim, x_dim, sparse=True):
    cnn_input = Input(shape=(seq_length,))
    embed = Embedding(input_shape=(seq_length,), input_dim=word_num + 1, output_dim=embedding_dim)(cnn_input)
    cnn_hidden = Conv1D(filters=128, kernel_size=3, activation='relu')(embed)
    cnn_hidden = MaxPooling1D(pool_size=K.get_variable_shape(cnn_hidden)[1])(cnn_hidden)
    cnn_hidden = Flatten()(cnn_hidden)
    cnn_hidden = BatchNormalization()(cnn_hidden)

    dense_input = Input(shape=(x_dim,), sparse=sparse)

    dense_hidden1 = Dense(units=16, activation='relu')(dense_input)
    dense_hidden1 = Dense(units=64, activation='relu')(dense_hidden1)
    dense_hidden1 = BatchNormalization()(dense_hidden1)
    dense_hidden1 = Dense(units=64, activation='relu')(dense_hidden1)
    dense_hidden1 = BatchNormalization()(dense_hidden1)

    dense_hidden2 = Dense(units=16, activation='relu')(dense_input)
    dense_hidden2 = Dense(units=64, activation='relu')(dense_hidden2)
    dense_hidden2 = BatchNormalization()(dense_hidden2)
    dense_hidden2 = Dense(units=64, activation='relu')(dense_hidden2)
    dense_hidden2 = BatchNormalization()(dense_hidden2)

    dense_hidden3 = Dense(units=16, activation='relu')(dense_input)
    dense_hidden3 = Dense(units=64, activation='relu')(dense_hidden3)
    dense_hidden3 = BatchNormalization()(dense_hidden3)
    dense_hidden3 = Dense(units=64, activation='relu')(dense_hidden3)
    dense_hidden3 = BatchNormalization()(dense_hidden3)

    dense_hidden4 = Dense(units=16, activation='relu')(dense_input)
    dense_hidden4 = Dense(units=64, activation='relu')(dense_hidden4)
    dense_hidden4 = BatchNormalization()(dense_hidden4)
    dense_hidden4 = Dense(units=64, activation='relu')(dense_hidden4)
    dense_hidden4 = BatchNormalization()(dense_hidden4)

    hidden = concatenate([cnn_hidden, dense_hidden1, dense_hidden2, dense_hidden3, dense_hidden4])
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dense(units=64, activation='relu')(hidden)

    output = Dense(units=1, activation='sigmoid')(hidden)

    model = Model(inputs=[cnn_input, dense_input], outputs=output)
    return model


