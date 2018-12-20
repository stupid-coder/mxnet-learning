#!/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import helper

def main():
    if len(sys.argv) != 2:
        print("Usage: plot [train_info's path]")
        sys.exit(-1)
    else:
        train_info = json.load(open(os.path.join(sys.argv[1], 'train.info'), "r"))
        helper.plot_loss_and_acc(train_info, sys.argv[1])
        sys.exit(0)

if __name__ == "__main__":
    main()