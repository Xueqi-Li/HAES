"""
main.py
Description: Running HAES and evaluating
Author: Xueqi Li (lee_xq@hnu.edu.cn)
Date: 2020.09.22
"""
import os

import numpy as np
import argparse

from Dataset import Dataset
import HAES
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Some parameters of HAES.")
    parser.add_argument("--dataset_name", default="ml-1m", help="Select a dataset.")
    parser.add_argument("--epoch", type=int, default=50, help="Number of epochs in GRU training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate in GRU training.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size in GRU training.")
    parser.add_argument("--len_cat", type=int, default=20, help="Length of category sequence in GRU training.")
    parser.add_argument("--K_cat", type=int, default=2, help="Number of selected categories.")

    return parser.parse_args()


if __name__ == "__main__":
    """
    - Initializing parameters
    - Running model HAES
    - Evaluating performance
    """
    # Initializing parametrs
    args = parse_args()
    dataset_name = args.dataset_name
    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    K_cat = args.K_cat
    len_cat = args.len_cat

    # Constructing the Dataset object
    dataset = Dataset(dataset_name, len_cat)

    # Running HAES model
    HAES.cal_elasticity(dataset_name, dataset.mat_rating_train, dataset.mat_cat)
    HAES.cal_relevance(dataset_name, dataset.mat_rating_train)
    HAES.pred_category(dataset_name, dataset.GRU_input, dataset.GRU_label, dataset.GRU_test_input, epoch, batch_size, lr)
    HAES.recommend(dataset, K_cat, epoch, batch_size, lr)

    # Evaluating performance
    evaluate.evaluate(dataset_name)

