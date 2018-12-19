#!/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
from matplotlib import pyplot as plt
import numpy as np
import time

__all__ = ['dataset', 'accuracy', 'evaluate', 'train', 'describe_net', 'plot_loss_and_acc', 'trygpu']

labels_literature = ["T-shirt/top","Trouser","Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def get_labels(labels):
    if isinstance(labels, list) or isinstance(labels, tuple):
        return [labels_literature[label] for label in labels]
    else:
        return labels_literature[labels]


def data():
    return gdata.vision.FashionMNIST(train=True), gdata.vision.FashionMNIST(train=False)

def dataset(batch_size=64):
    train_data, test_data = data()

    transformer = gdata.vision.transforms.Compose([
        gdata.vision.transforms.ToTensor()
        ])

    train_iter = gdata.DataLoader(train_data.transform_first(transformer),
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=4)

    test_iter = gdata.DataLoader(test_data.transform_first(transformer),
                                 batch_size=batch_size, shuffle=True,
                                 num_workers=4)


    return train_iter, test_iter


def plot_dataset(images, labels, rows, cols):
    plt.figure(figsize=(12,12))
    for i in range(rows):
        for j in range(cols):
            image,label = images[i*cols+j], labels[i*cols+j]
            plt.subplot(rows, cols, i*cols+j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image.asnumpy().squeeze(), cmap=plt.cm.binary)
            plt.xlabel(get_labels(label))
    plt.show()


def plot_loss_and_acc(train_loss, train_acc, test_loss, test_acc):
    plt.figure(figsize=(12, 12))
    plt.subplot(1,2,1)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(range(1, len(train_loss)+1), train_loss, color='red', label="train-loss", marker='o')
    plt.plot(range(1, len(test_loss)+1), test_loss, color='blue', label="test-loss", marker='o')
    plt.legend()
    plt.title("loss")

    plt.subplot(1,2,2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(range(1, len(train_acc)+1), train_acc, color='red', label="train-acc", marker='o')
    plt.plot(range(1, len(test_acc)+1), test_acc, color='blue', label="test-acc", marker='o')
    plt.legend()
    plt.title("accuracy")

    plt.savefig("loss_acc.png")


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def evaluate(data_iter, net, loss_fn, ctx):
    acc = nd.array([0], ctx=ctx)
    loss = nd.array([0], ctx=ctx)
    for X,y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        y_hat = net(X)
        acc += accuracy(y_hat, y)
        loss += loss_fn(y_hat, y).mean().asscalar()
    return loss.asscalar() / len(data_iter), acc.asscalar() / len(data_iter)

def describe_net(net):
    X = nd.random.uniform(shape=(1, 1, 28, 28))
    for layer in net:
        X = layer(X)
        print(layer.name, "output shape:\t", X.shape)

def trygpu():
    try:
        ctx = mx.gpu()
        a = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def train(net, trainer, train_iter, test_iter, loss, ctx, num_epochs=5):
    train_ls = []
    train_acc = []
    test_ls = []
    test_acc = []
    for i in range(num_epochs):
        train_ls_sum, train_acc_sum = 0, 0
        begin_clock = time.clock()

        for X, y in train_iter:
            X,y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).mean()
            l.backward()
            trainer.step(1)
            train_ls_sum += l.asscalar()
            train_acc_sum += accuracy(y_hat, y)

        train_ls.append(train_ls_sum/len(train_iter))
        train_acc.append(train_acc_sum/len(train_iter))
        tloss, tacc = evaluate(test_iter, net, loss, ctx)
        test_ls.append(tloss)
        test_acc.append(tacc)

        end_clock = time.clock()

        print("epoch {} - train loss: {}, train accuracy: {}, test loss: {}, test_accuracy: {}, cost time:{}".format(
            i+1, train_ls[-1], train_acc[-1], test_ls[-1], test_acc[-1], end_clock-begin_clock))
    return train_ls, train_acc, test_ls, test_acc


if __name__ == "__main__":
    plot_loss_and_acc([1,2,3],[3,4,5],[1.1,2.2,3.3],[3.3,4.4,5.5])
    import sys; sys.exit(0)
    train_data, test_data = data()
    plot_dataset(train_data[0:25][0], train_data[0:25][1], 5, 5)
