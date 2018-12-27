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


def build(restore_dir):

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    network = nn.Sequential()
    network.add(
        nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1)
    )

    network.add(
        resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))

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
