# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 22:21
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : train.py
# @Software: PyCharm

import os
from drxan import train_models


current_dir = os.getcwd()
data_dirs = dict()
data_dirs['train_data'] = os.path.join(current_dir, 'data/train.csv')
data_dirs['test_data'] = os.path.join(current_dir, 'data/test.csv')
data_dirs['pred_result'] = os.path.join(current_dir, 'data/lightGBM_04.csv')
data_dirs['model_path'] = os.path.join(current_dir, 'data/model_data/lightGBM_04.pkl')
# data_dirs['model_path'] = os.path.join(current_dir, 'data/model_data/model-ep{epoch:03d}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5')
# data_dirs['logs_path'] = os.path.join(current_dir, 'data/logs/cnn_dnn_merge_03/')


if __name__ == '__main__':
    train_models.M05_CNN_DNN_merge.train_predict(data_dirs)
