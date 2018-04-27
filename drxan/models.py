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
    x_input = Input(shape=(seq_length,))
    embed = Embedding(input_shape=(seq_length,), input_dim=word_num+1, output_dim=embedding_dim)(x_input)
    hidden = Conv1D(filters=128, kernel_size=5, activation='relu')(embed)
    # hidden = MaxPooling1D(pool_size=2)(hidden)
    # hidden = Conv1D(filters=64, kernel_size=3, activation='relu')(hidden)
    hidden = MaxPooling1D(pool_size=K.get_variable_shape(hidden)[1])(hidden)
    hidden = Flatten()(hidden)
    hidden = Dense(units=64,activation='tanh')(hidden)
    hidden = Dense(units=32, activation='tanh')(hidden)
    output = Dense(units=1, activation='sigmoid')(hidden)

    model = Model(inputs=x_input, outputs=output)
    return model
