#!/bin/env python
# -*- coding: utf-8

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
from utils import *

ctx = trygpu()

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
    net.initialize(ctx=ctx)
    return net

def main(batch_size=64, lr=0.1):
    net = build_LeNet('sigmoid')
    describe_net(net)
    train_iter, test_iter = dataset(batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
    plot_loss_and_acc(*train(net, trainer, train_iter, test_iter, gloss.SoftmaxCrossEntropyLoss(), ctx, num_epochs=10))

if __name__ == "__main__":
    main()
