#!/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import helper

def main():
    if len(sys.argv) != 3:
        print("Usage: plot [type] [info's path]")
        sys.exit(-1)
    else:
        if sys.argv[1] == "acc_loss":
            info = json.load(open(os.path.join(sys.argv[2], 'train.info'), "r"))
            helper.plot_loss_and_acc(train_info, sys.argv[2])
        elif sys.argv[1] == "circle_lr_test":
            info = json.load(open(os.path.join(sys.argv[2], 'lr-min-max.info'), "r"))
            helper.plot_lr_min_max(info, sys.argv[2])
        else:
            print("no plot")

if __name__ == "__main__":
    main()
