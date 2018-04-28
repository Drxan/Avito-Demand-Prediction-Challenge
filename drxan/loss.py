# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 15:57
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : loss.py
# @Software: PyCharm
# 由于根号函数的导数可能出现无穷大值，会破坏网络参数更新，弃用该自定义损失函数

from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


rmse = root_mean_squared_error
