"""
evaluate.py
Description: Evaluating functions
Author: Xueqi Li (lee_xq@hnu.edu.cn)
Date: 2020.09.23
"""
import os

import bottleneck as bn
import numpy as np

import baseline


def rec_pre_f1(pred, label):
    """
    - Calculate recall, precision and F1 score
    Inputs:
        pred: predicted vlaues, 0, 1
        label: 0, 1 
    Outputs:
        recall, precision, F1 score
    """
    tmp = np.sum(np.logical_and(label, pred), axis=1).astype(np.float32)
    recall = tmp / label.sum(axis=1)
    recall[np.isnan(recall)] = 0
    recall = np.mean(recall)
    precision = tmp / pred.sum(axis=1)
    precision[np.isnan(precision)] = 0
    precision = np.mean(precision)

    f1 = 2 * recall * precision / (recall + precision)

    return recall, precision, f1


def avg_hamming_dis(pred, label):
    """
    - Calculating Average Hamming Distance
    Inputs:
        pred: predicted vlaues, 0, 1
        label: 0, 1 
    Outputs:
        hamming distance
    """
    hamming_dis = np.mean(np.logical_xor(label, pred)).astype(np.float32)

    return hamming_dis


def difference(mat_rel, pred):
    """
    - Calculating difference between the user and predicted items
    Inputs:
        mat_rel: relevance matrix
        pred: predicted recommendation matrix
    Outputs:
        dif: average difference factor 
    """
    num = pred.shape[0]
    mat_rel_u2i = mat_rel[:num, num:]
    mat_rel_u2i = mat_rel_u2i * pred

    return 1 - np.sum(mat_rel_u2i) / np.sum(mat_rel_u2i != 0)


def diversity(mat_rel, pred):
    """
    - Calculating diversity of recommendations
    Inputs:
        mat_rel: relevance matrix
        pred: predicted recommendation matrix
    Outputs:
        div: average diversity factor 
    """
    num = pred.shape[0]

    mat_rel_u2i = mat_rel[:num, num:]
    mat_rel_u2i = mat_rel_u2i * pred
    mat_rel_i2i = mat_rel[num:, num:]

    list_rel = []
    for uind in range(num):
        list_iind = np.nonzero(mat_rel_u2i[uind])[0]
        for i in list_iind:
            for j in list_iind:
                if i != j:
                    list_rel.append(mat_rel_i2i[i, j])
    div = np.mean(np.array(list_rel))

    return 1 - div


def evaluate(dataset_name):
    """
    - Evaluating performance on category accuracy 
    - Evaluating performance on content difference 
    Inputs:
        dataset_name: dataset name
        K_cat: employing the top-K_cat categories as predictions 
        K_item: employing the top-K_item items as recommendations 
    Outputs:
        f1: F1 score on category accuracy
        hamming_dis: average hamming distance on category accuracy
        dif: user-item difference on content difference 
        div: diversity of recommendations on content difference
    """
    label = np.load(os.path.join("data", dataset_name, "test", "ratings.npy"))
    pred = np.load(os.path.join("data", dataset_name, "result", "pred.npy"))
    mat_rel = np.load(os.path.join("data", dataset_name, "result", "rel.npy"))
    mat_cat = np.load(os.path.join("data", dataset_name, "item_cat.npy"))

    list_K = [5, 10, 15]
    for K in list_K:
        num = pred.shape[0]
        idx = bn.argpartition(-pred, K, axis=1)
        pred_binary = np.zeros_like(pred, dtype=bool)
        pred_binary[np.arange(num)[:, np.newaxis], idx[:, :K]] = True
        pred_cat = np.matmul(pred_binary, mat_cat)
        pred_cat[pred_cat > 0] = 1
        label[label > 0] = 1
        label_cat = np.matmul(label, mat_cat)
        label_cat[label_cat > 0] = 1

        list_r = rec_pre_f1(pred_cat, label_cat)
        print("in top-{} recommendation:".format(K))
        print("rec_pre_f1: {:.6f},{:.6f},{:.6f}".format(list_r[0], list_r[1], list_r[2]))
        print("hamming distance: {:.6f}".format(avg_hamming_dis(pred_cat, label_cat)))
        print("difference:{:.6f}".format(difference(mat_rel, pred_binary)))
        print("diversity:{:.6f}".format(diversity(mat_rel, pred_binary)))
