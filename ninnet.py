#!/bin/env python
# -*- coding: utf-8

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import sys, os
import helper
import logging

logger = logging.getLogger(__name__)

helper.parser.add_argument("--activation", help="network's activation function", type=str, default='relu')

def build_NiN(restore_dir):

    def nin_block(num_channels, kernel_size, strides, padding):
        blk = nn.Sequential()
        blk.add(nn.Conv2D(num_channels, kernel_size,
                          strides, padding, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
        return blk

    network = nn.Sequential()
    network.add(
        nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(10, kernel_size=3, strides=1, padding=1),
        nn.GlobalAvgPool2D(),
        nn.Flatten())

    if restore_dir:
        helper.restore(network, restore_dir)
    else:
        network.initialize(ctx=helper.ctx)

    return network

def main():

    options = helper.parser.parse_args()

    logger.info("run config:{}".format(options))

    helper.ctx = helper.trygpu(options.gpu)

    network = build_NiN(options.restore_dir)

    helper.describe_net(network, options.resize)

    trainer = gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate': options.learning_rate})

    helper.run(network, trainer, gloss.SoftmaxCrossEntropyLoss(), options)

if __name__ == "__main__":
    main()
