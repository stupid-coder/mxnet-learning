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

class Inception(nn.Block):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')

        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')

        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')

        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')


    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)

def build_googlenet(restore_dir):

    def block1():
        b = nn.Sequential()
        b.add( nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return b

    def block2():
        b = nn.Sequential()
        b.add( nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return b

    def block3():
        b = nn.Sequential()
        b.add( Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return b

    def block4():
        b = nn.Sequential()
        b.add( Inception(192, (96, 208), (16, 48), 64),
               Inception(160, (112, 224), (24, 64), 64),
               Inception(128, (128, 256), (24, 64), 64),
               Inception(112, (144, 288), (32, 64), 64),
               Inception(256, (160, 320), (32, 128), 128),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return b

    def block5():
        b = nn.Sequential()
        b.add( Inception(256, (160, 320), (32, 128), 128),
               Inception(384, (192, 384), (48, 128), 128),
               nn.GlobalAvgPool2D())
        return b

    network = nn.Sequential()

    network.add(
        block1(),
        block2(),
        block3(),
        block4(),
        block5(),
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

    network = build_googlenet(options.restore_dir)

    helper.describe_net(network, options.resize)

    trainer = gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate': options.learning_rate})

    helper.run(network, trainer, gloss.SoftmaxCrossEntropyLoss(), options)

if __name__ == "__main__":
    main()
