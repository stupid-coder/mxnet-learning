#!/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, init, nd
from mxnet.gluon import loss as gloss, nn

import helper
import logging

logger = logging.getLogger(__name__)

def build_AlexNet(restore_dir):
    network = nn.Sequential()
    network.add(
        nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(10)
    )

    if restore_dir:
        helper.restore(network, restore_dir)
    else:
        network.initialize(ctx=helper.ctx, init=init.Xavier())

    return network

def main():
    options = helper.parser.parse_args()

    logger.info("run config:{}".format(options))

    helper.ctx = helper.trygpu(options.gpu)

    network = build_AlexNet(options.restore_dir)

    helper.describe_net(network)

    trainer = gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate':options.learning_rate})

    helper.run(network, trainer, gloss.SoftmaxCrossEntropyLoss(), options)

if __name__ == "__main__":
    main()
