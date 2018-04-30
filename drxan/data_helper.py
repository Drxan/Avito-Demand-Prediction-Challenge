# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 22:21
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : data_helper.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from string import punctuation as en_punctuation
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


def extract_features(data_path, target=None, word_dict=None, min_freq=5):
    data = pd.read_csv(data_path)
    if target is not None:
        # data = pd.read_csv(data_path, nrows=500000)
        txt_sequences, word_dict = convert_text_to_sequence(data[['title', 'description']], word_dict, min_freq=min_freq)
    else:
        txt_sequences = convert_text_to_sequence(data[['title', 'description']], word_dict)
    if target is not None:
        return txt_sequences, data[target].values, word_dict
    else:
        return txt_sequences, data[['item_id']]


def load_text_data(data_path, train=False):
    if train:
        texts = pd.read_csv(data_path, usecols=['title', 'description', 'deal_probability'])
        targets = texts['deal_probability'].values
        texts = texts[['title', 'description']].apply(lambda x: str(x[0])+('' if x[1] is np.nan else ' '+str(x[1])), axis=1)
        texts = texts.apply(filter_text)
        return texts, targets
    else:
        texts = pd.read_csv(data_path, usecols=['title', 'description', 'item_id'])
        pred_items = texts[['item_id']]
        texts = texts[['title', 'description']].apply(lambda x: str(x[0])+('' if x[1] is np.nan else ' '+str(x[1])), axis=1)
        texts = texts.apply(filter_text)
        return texts, pred_items


def get_tfidf(data_path, tfidf_transformer=None, train=False):
    print('get_tfidf:loading raw text data...')
    texts, targets = load_text_data(data_path, train=train)
    if train:
        print('get_tfidf:training TfidfVectorizer...')
        stop_words = list(stopwords.words('russian')) + list(stopwords.words('english'))
        tfidf_transformer = TfidfVectorizer(binary=False, norm='l2', stop_words=stop_words, max_features=200000)
        tfidf_transformer = tfidf_transformer.fit(texts)
    tfidf = tfidf_transformer.transform(texts)
    return tfidf, targets, tfidf_transformer


def filter_text(text, filters=None, lower=True):
    """
    对文本进行过滤
    :param text: 需要过滤处理的文本
    :param filters: 需要过滤掉的字符集
    :param lower: 是否忽略英文字符的大小写
    :return: 过滤后的文本
    """
    if lower:
        text = text.lower()
    if filters is None:
        filters = en_punctuation+r'\t\r\n\f\v'
    sub_parttern = '['+filters+']'
    text = re.sub(sub_parttern, ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def get_word_dict(texts, stop_words=[], min_freq=5):
    """
    根据原始训练语料建立词到数值的映射字典
    :param texts:
    :param stop_words:
    :param min_freq:
    :return:
    """
    word_count = Counter()
    for txt in texts:
        txt = filter_text(txt, filters=None, lower=True)
        tokens = [w.strip() for w in re.split(r'\s+', txt)]
        word_count.update(tokens)
    # 去掉低频词和停止词
    for w in list(word_count.keys()):
        if (word_count[w] < min_freq) or (w in stop_words):
            del word_count[w]
    word_dict = dict(zip(word_count, range(1, len(word_count)+1)))
    word_dict['UNK'] = len(word_dict)+1
    return word_dict


def convert_text_to_sequence(texts, word_dict=None, min_freq=5):
    # 将title和description合并
    texts = texts.apply(lambda x: str(x[0])+('' if x[1] is np.nan else ' '+str(x[1])), axis=1)
    txt_sequences = []

    if word_dict is None:  # train data
        stop_words = list(stopwords.words('russian'))+list(stopwords.words('english'))
        word_dict = get_word_dict(texts, stop_words=stop_words, min_freq=min_freq)
        for txt in texts:
            # 对每一条文本进行分词、过滤
            txt = filter_text(txt, filters=None, lower=True)
            txt_seq = []
            tokens = [w for w in re.split(r'\s+', txt)]
            # 对每一条文本中的词进行数值编码
            for tk in tokens:
                word_idx = word_dict.get(tk, word_dict['UNK'])
                txt_seq.append(word_idx)
            txt_sequences.append(txt_seq)
        return txt_sequences, word_dict
    else:  # test data
        for txt in texts:
            txt = filter_text(txt, filters=None, lower=True)
            txt_seq = []
            tokens = [w for w in re.split(r'\s+', txt)]
            for tk in tokens:
                word_idx = word_dict.get(tk, word_dict['UNK'])
                txt_seq.append(word_idx)
            txt_sequences.append(txt_seq)
        return txt_sequences


def pad_sequences(txt_seqs, max_len=50, padding='post', truncating='post'):
    return sequence.pad_sequences(txt_seqs, maxlen=max_len, padding=padding, truncating=truncating)


def generate_batch(x, y=None, batch_size=128):
    sample_num = len(x[0]) if isinstance(x, list) else len(x)
    idxs = np.arange(sample_num)
    if sample_num < batch_size:
        batches = 1
        batch_size = sample_num
    else:
        batches = sample_num // batch_size

    while True:
        np.random.shuffle(idxs)
        for batch in range(batches):
            batch_idx = idxs[batch:batch+batch_size]
            if isinstance(x, list):
                batch_x = [x_mem[batch_idx].toarray() if issparse(x_mem) else x_mem[batch_idx] for x_mem in x]
            else:
                batch_x = x[batch_idx].toarray() if issparse(x) else x[batch_idx]
            if y is not None:
                batch_y = y[batch_idx]
                batch_data = (batch_x, batch_y)
            else:
                batch_data = batch_x
            yield batch_data
