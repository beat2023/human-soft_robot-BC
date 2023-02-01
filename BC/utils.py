# https://github.com/jerrylin1121/BCO/blob/tf2.0/models/utils.py

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--demo_name", default="", help="the demonstration type")
parser.add_argument("--remap_type", default="norm", choices=["minMax", "norm", "zScore", "confInt",
                                                             "ellipsoid"], help="the remapping type", required=True)
parser.add_argument("--mode", default="train", choices=["train", "test", "train_human", "test_human", "idm_perform",
                                                        "idm_perform_human"], required=True)

parser.add_argument("--max_episodes", type=int, default=20, help="the number of training episodes")

parser.add_argument("--batch_percentage", type=float, default=0.25, help="number of examples in batch")
parser.add_argument("--lr_policy", type=float, default=0.005, help="initial learning rate for adam SGD of policy")

args = parser.parse_args()


def get_shuffle_idx(num, batch_percentage):
    batch = int(num*batch_percentage)
    tmp = np.arange(num)
    np.random.shuffle(tmp)
    split_array = []
    cur = 0
    while num > batch:
        num -= batch
        if num != 0:
            split_array.append(cur+batch)
            cur += batch
    return np.split(tmp, split_array)


def get_ordered_idx(num, batch_percentage):
    batch = int(num*batch_percentage)
    tmp = np.arange(num)
    split_array = []
    cur = 0
    while num > batch:
        num -= batch
        if num != 0:
            split_array.append(cur+batch)
            cur += batch
    return np.split(tmp, split_array)
