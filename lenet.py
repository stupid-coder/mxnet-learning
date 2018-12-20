#!/bin/env python
# -*- coding: utf-8

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import sys, os
from helper import *

def build_LeNet(activation='sigmoid'):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, strides=1, activation=activation),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation=activation),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation=activation),
        nn.Dense(84, activation=activation),
        nn.Dense(10)
    )

    if options.restore_dir:
        restore(network, options.restore_dir)
    else:
        network.initialize(ctx=_ctx)

    return net

def main(batch_size=64, lr=0.1):
    network = build_LeNet('sigmoid')
    describe_net(network)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':options.learning_rate})
    run(network, trainer, gloss.SoftmaxCrossEntropyLoss())

if __name__ == "__main__":
    main()
