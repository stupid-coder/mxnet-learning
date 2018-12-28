#!/bin/env python
# -*- coding: utf-8

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import sys, os
import helper
import logging
from residual import Residual

logger = logging.getLogger(__name__)

helper.parser.add_argument("--activation", help="network's activation function", type=str, default='relu')


def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock ,self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)
        return X

def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk

def build(restore_dir):

    network = nn.Sequential()

    network.add(
        nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        network.add(DenseBlock(num_convs, growth_rate))
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            network.add(transition_block(num_channels // 2))

    network.add(nn.GlobalAvgPool2D(), nn.Dense(10))

    if restore_dir:
        helper.restore(network, restore_dir)
    else:
        network.initialize(ctx=helper.ctx, init=init.Xavier())

    return network


def main():

    options = helper.parser.parse_args()

    logger.info("run config:{}".format(options))

    helper.ctx = helper.trygpu(options.gpu)

    network = build(options.restore_dir)

    helper.describe_net(network, options.resize)

    trainer = gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate': options.learning_rate})

    helper.run(network, trainer, gloss.SoftmaxCrossEntropyLoss(), options)

if __name__ == "__main__":
    main()
