#!/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd, autograd, gluon, init
from mxnet.gluon import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):

    if not autograd.is_training():  # 判断是否是预测模型
        X_hat = (X - moving_mean) / nd.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:   # 全链接层
            mean = X.mean(axis=0)
            var = ((X-mean)**2).mean(axis=0)
        else:                   # 卷积
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X-mean)**2).mean(axis=(0, 2, 3), keepdims=True)
        # 训练模式下,使用批量均值和方差执行标准化
        X_hat = (X-mean)/nd.sqrt(var + eps)
        # 更新均值和方差
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var

class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = self.params.get("gamma", shape=shape, init=init.One())
        self.beta = self.params.get("beta", shape=shape, init=init.Zero())
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X):
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentum=0.9)
        return Y
