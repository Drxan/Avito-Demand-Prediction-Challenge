# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 22:21
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : data_helper.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from keras.preprocessing import sequence


def extract_features(data_path, target=None, word_dict=None):
    data = pd.read_csv(data_path)
    if target is not None:
        data = pd.read_csv(data_path, nrows=800000)
        txt_sequences, word_dict = convert_text_to_sequence(data[['title', 'description']], word_dict)
    else:
        txt_sequences = convert_text_to_sequence(data[['title', 'description']], word_dict)
    if target is not None:
        return txt_sequences, data[target].values, word_dict
    else:
        return txt_sequences, data[['item_id']]


def convert_text_to_sequence(texts, word_dict=None):
    # 将title和description合并
    texts = texts.apply(lambda x: str(x[0])+('' if x[1] is np.nan else ' '+str(x[1])), axis=1)
    txt_sequences = []

    if word_dict is None:  # train data
        word_dict = {}
        for txt in texts:
            txt_seq = []
            tokens = [w.strip() for w in txt.split(' ') if w not in ['', ' ']]
            for tk in tokens:
                if tk not in word_dict:
                    word_dict[tk] = len(word_dict)+1
                txt_seq.append(word_dict[tk])
            txt_sequences.append(txt_seq)
        word_dict['UNK'] = len(word_dict)+1  # index for unknown words
        return txt_sequences, word_dict
    else:  # test data
        for txt in texts:
            txt_seq = []
            tokens = [w.strip() for w in txt.split(' ') if w not in ['', ' ']]
            for tk in tokens:
                txt_seq.append(word_dict.get(tk, word_dict['UNK']))
            txt_sequences.append(txt_seq)
        return txt_sequences


def pad_sequences(txt_seqs, max_len=50, padding='post', truncating='post'):
    return sequence.pad_sequences(txt_seqs, maxlen=max_len, padding=padding, truncating=truncating)
