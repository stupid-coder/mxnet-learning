#!/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, init, nd
from mxnet.gluon import loss as gloss, nn

import helper
import logging

logger = logging.getLogger(__name__)

def build_VGGNet(restore_dir):

    def vgg_block(num_convs, num_channels):
        blk = nn.Sequential()
        for _ in range(num_convs):
            blk.add(nn.Conv2D(num_channels, kernel_size=3,
                              padding=1, activation='relu'))
        blk.add(nn.MaxPool2D(pool_size=2, strides=2))
        return blk

    network = nn.Sequential()

    conv_arch = ((1,64), (1, 128), (2, 256), (2, 512), (2, 512))  # VGG11

    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))

    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))

    if restore_dir:
        helper.restore(network, restore_dir)
    else:
        network.initialize(ctx=helper.ctx, init=init.Xavier())

    return network

def main():
    options = helper.parser.parse_args()

    logger.info("run config:{}".format(options))

    helper.ctx = helper.trygpu(options.gpu)

    network = build_VGGNet(options.restore_dir)

    helper.describe_net(network, options.resize)

    trainer = gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate':options.learning_rate})

    helper.run(network, trainer, gloss.SoftmaxCrossEntropyLoss(), options)

if __name__ == "__main__":
    main()
