#!/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import data as gdata, loss as gloss
from matplotlib import pyplot as plt
import numpy as np
import argparse
import time
import json
import os
import sys
import logging
import math

__all__ = ['dataset', 'accuracy', 'evaluate', 'train', 'run', 'describe_net', 'plot_loss_and_acc', 'trygpu', 'parser', 'ctx', 'restore']

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="train's batch size", type=int, default=256)
parser.add_argument("--num_epochs", help="train epochs number", type=int, default=10)
parser.add_argument("--begin_epoch", help="begin epoch in this train process", type=int, default=1)
parser.add_argument("--learning_rate", help="learning rate in this train process", type=float, default=0.1)
parser.add_argument("--restore_dir", help="from where directory to restore the model", type=str, default=None)
parser.add_argument("--save_dir", help="to where directory to save the model's parameters and train test information", type=str)
parser.add_argument('--gpu', help="which gpu to use", type=int, default=0)

parser.add_argument('--run_circle_test', help="run circle learning rate to find learning rate min max", action="store_true")
parser.add_argument('--use_circle_lr', help="use circle lr train model", type=bool, default=False)
parser.add_argument('--lr_min', help="min learning rate", type=float, default=0.0001)
parser.add_argument("--lr_max", help="max learning rate", type=float, default=1)
parser.add_argument("--step_size", help="step size for circle learning, multiple of the iter count of single epoch", type=int, default=None)

_labels_literature = ["T-shirt/top","Trouser","Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_NETWORK_PARAMS = 'network.params'
_TRAIN_INFO = 'train.info'
_LR_MIN_MAX = 'lr-min-max.info'

def trygpu(gpu):
    try:
        ctx = mx.gpu(gpu)
        a = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = None

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
def get_labels(labels):
    if isinstance(labels, list) or isinstance(labels, tuple):
        return [_labels_literature[label] for label in labels]
    else:
        return _labels_literature[labels]

def data():
    return gdata.vision.FashionMNIST(train=True), gdata.vision.FashionMNIST(train=False)

def dataset(batch_size):
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


def plot_loss_and_acc(train_info, save_path):
    epochs = len(train_info['train_loss'])
    plt.figure(figsize=(12, 12))
    plt.subplot(2,1,1)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(range(1, epochs+1), train_info['train_loss'], color='red', label="train-loss", marker='o', markersize=2)
    plt.plot(range(1, epochs+1), train_info['test_loss'], color='blue', label="test-loss", marker='o', markersize=2)
    plt.legend()
    plt.title("loss")

    plt.subplot(2,1,2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(range(1, epochs+1), train_info['train_acc'], color='red', label="train-acc", marker='o', markersize=2)
    plt.plot(range(1, epochs+1), train_info['test_acc'], color='blue', label="test-acc", marker='o', markersize=2)
    plt.legend()
    plt.title("accuracy")

    plt.savefig(os.path.join(save_path, "loss_acc.png"))


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def evaluate(data_iter, net, loss_fn):
    acc = nd.array([0], ctx=ctx)
    loss = nd.array([0], ctx=ctx)
    for X,y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        y_hat = net(X)
        acc += accuracy(y_hat, y)
        loss += loss_fn(y_hat, y).mean().asscalar()
    return loss.asscalar() / len(data_iter), acc.asscalar() / len(data_iter)

def describe_net(net):
    logger.info("ctx:{}".format(ctx))
    X = nd.random.uniform(shape=(1, 1, 28, 28), ctx=ctx)
    logger.info("network architecture")
    for layer in net:
        X = layer(X)
        logger.info("{} - output shape:{}".format(layer.name, X.shape))
    logger.info("network architecture finished")


def train(net, trainer, train_iter, test_iter, loss, options):
    logger.info("train on: {}".format(ctx))
    train_ls = []
    train_acc = []
    lr = []
    test_ls = []
    test_acc = []

    for i in range(options.num_epochs):
        train_ls_sum, train_acc_sum = 0, 0
        begin_clock = time.clock()

        epoch_count = 0
        for X, y in train_iter:

            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).mean()
            l.backward()

            if options.use_circle_lr or options.run_circle_test:
                circle_lr = circle_learning_rate(i*len(train_iter)+epoch_count, options.lr_min, options.lr_max, options.step_size*len(train_iter))
                print(circle_lr)
                lr.append(circle_lr)
                trainer.set_learning_rate(circle_lr)
                epoch_count += 1
            trainer.step(1)

            if options.run_circle_test:
                train_ls.append(float(l.asscalar()))
                train_acc.append(float(accuracy(y_hat, y)))
            else:
                train_ls_sum += l.asscalar()
                train_acc_sum += accuracy(y_hat, y)

        if not options.run_circle_test:
            train_ls.append(train_ls_sum/len(train_iter))
            train_acc.append(train_acc_sum/len(train_iter))


            tloss, tacc = evaluate(test_iter, net, loss)
            test_ls.append(tloss)
            test_acc.append(tacc)

        end_clock = time.clock()

        logger.info("epoch {} - train loss: {}, train accuracy: {}, test loss: {}, test_accuracy: {}, cost time:{}".format(
            i+1,
            train_ls[-1] if not options.run_circle_test else np.mean(train_ls[-len(train_iter):]),
            train_acc[-1] if not options.run_circle_test else np.mean(train_acc[-len(train_iter):]),
            test_ls[-1] if not options.run_circle_test else [],
            test_acc[-1] if not options.run_circle_test else [],
            end_clock-begin_clock))
    if options.run_circle_test:
        return train_ls, train_acc, lr
    else:
        return train_ls, train_acc, test_ls, test_acc


def circle_learning_rate(iter_count, base_lr, max_lr, step_size):
    cycle = math.floor( 1 + iter_count / (2 * step_size) )
    ratio = abs( iter_count / step_size - 2 * cycle + 1 )
    lr = base_lr + (max_lr-base_lr) * max( (1-ratio), 0 )
    print("cycle:{}\tratio:{}\tlr:{}".format(cycle, ratio, lr))
    return lr


def restore(network, restore_dir):

    if os.path.exists(restore_dir):
        network_params = os.path.join(restore_dir, _NETWORK_PARAMS)
        if os.access(network_params, os.R_OK):
            logger.info("restore the network from {}".format(network_params))
            network.load_parameters(network_params, ctx=ctx)
        else:
            logger.fatal("falure to load the network from {}".format(network_params))
    else:
        logger.fatal("{} restore directory not exists".format(restore_dir))


def save(network, restore_dir, save_dir, train_loss, train_acc, test_loss, test_acc):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if restore_dir:
        restore_train_info = os.path.join(restore_dir, _TRAIN_INFO)
        if os.access(restore_train_info, os.R_OK):
            train_info = json.load(open(restore_train_info, 'r'))
            train_info['train_loss'].extend(train_loss)
            train_info['train_acc'].extend(train_acc)
            train_info['test_loss'].extend(test_loss)
            train_info['test_acc'].extend(test_acc)
    else:
        train_info = {'train_loss':train_loss, 'train_acc':train_acc,
                      'test_loss':test_loss, 'test_acc':test_acc}

    #plot_loss_and_acc(train_info, save_dir)
    json.dump(train_info, open(os.path.join(save_dir, _TRAIN_INFO), "w"))
    network.save_parameters(os.path.join(save_dir, _NETWORK_PARAMS))

def run_train(network, trainer, loss_fn, options):

    train_iter, test_iter = dataset(options.batch_size)

    train_loss, train_acc, test_loss, test_acc = train(network, trainer, train_iter, test_iter, loss_fn, options)

    save(network,
         options.restore_dir,
         os.path.join(options.save_dir,"{}-{}-{}".format(options.begin_epoch,options.begin_epoch+options.num_epochs,int(test_acc[-1]*100))),
         train_loss, train_acc, test_loss, test_acc)

def plot_lr_min_max(info, save_dir):
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    plt.xlabel("learning-rate")
    plt.xticks()
    plt.ylabel("loss")
    plt.plot(info["lr"], info["loss"], color='red', marker='o', markersize=2)
    plt.title("learning-rate-loss")

    plt.subplot(2,1,2)
    plt.xlabel("learning-rate")
    plt.ylabel("accuracy")
    plt.plot(info["lr"], info["acc"], color='red', marker='o', markersize=2)
    plt.title("learning-rate-accuracy")

    plt.savefig(os.path.join(save_dir,"lr_min_max.png"))


def save_lr_min_max(loss, acc, lr, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    info = {"loss": loss, "acc": acc, "lr": lr}

    json.dump(info, open(os.path.join(save_dir, _LR_MIN_MAX), 'w'))

def run_lr_min_max(network, trainer, loss_fn, options):
    train_iter, test_iter = dataset(options.batch_size)

    trainer = gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate':0})

    loss, acc, lr = train(network, trainer, train_iter, test_iter, loss_fn, options)

    save_lr_min_max(loss,acc,lr,save_dir=options.save_dir)

def run(network, trainer, loss_fn, options):
    if options.run_circle_test:
        run_lr_min_max(network, trainer, loss_fn, options)
    else:
        run_train(network, trainer, loss_fn, options)



