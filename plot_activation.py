#!/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import sys
from mxnet import nd, autograd

def sigmoid(x):
    return 1.0 / (1 + (-x).exp())

def tanh(x):
    return (x.exp()-(-x).exp()) / (x.exp()+(-x).exp())

def softsign(x):
    return x / (1+x.abs())

def relu(x):
    return nd.maximum(x, 0)


def plot(x, y, p, name):
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,y)

    plt.subplot(2,1,2)
    plt.title("{}'".format(name))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,p)

    plt.savefig("./{}.png".format(name))


def run(act):
    x = nd.array(np.linspace(-10, 10, 1000))
    x.attach_grad()
    with autograd.record():
        y = act(x)
    y.backward()
    plot(x.asnumpy(), y.asnumpy(), x.grad.asnumpy(),act.__name__)

def main():
    if sys.argv[1] == "sigmoid":
        run(sigmoid)
    elif sys.argv[1] == "tanh":
        run(tanh)
    elif sys.argv[1] == "softsign":
        run(softsign)
    elif sys.argv[1] == "relu":
        run(relu)

if __name__ == "__main__":
    main()
