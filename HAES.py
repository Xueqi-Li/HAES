"""
HAES.py
Description: Creating HAES model
Author: Xueqi Li (lee_xq@hnu.edu.cn)
Date: 2020.09.22
"""
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import bottleneck as bn
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


class Gru(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layer):
        super(Gru, self).__init__()
        self.gru = nn.GRU(dim_in, dim_hidden, num_layer, batch_first=True)
        self.classifier = nn.Linear(dim_hidden, dim_out)

    def forward(self, input):
        output, _ = self.gru(input)
        output = self.classifier(output[:, -1, :])
        return output


def cal_elasticity(dataset_name, mat_rating, mat_item_cat):
    """
    - Calculating User Elasticity
    - Calculating Item Elasticity
    Inputs:
        dataset_name: dataset name
        mat_rating: rating matrix
        mat_item_cat: item category matrix
    Outputs:
        list_ela_user: user elasticity
        list_ela_item: item elasticity
    """
    path_ela_user = os.path.join("data", dataset_name, "result", "ela_user.npy")
    path_ela_item = os.path.join("data", dataset_name, "result", "ela_item.npy")
    if os.path.exists(path_ela_user) and os.path.exists(path_ela_item):
        return np.load(path_ela_user), np.load(path_ela_item)

    if not os.path.exists(os.path.join("data", dataset_name, "result")):
        os.mkdir(os.path.join("data", dataset_name, "result"))

    num_user, num_item = mat_rating.shape
    list_uind, list_iind = np.nonzero(mat_rating)
    df_rating = pd.DataFrame({"uind": list_uind, "iind": list_iind})

    # Calculating User Elasticity
    list_ela_user = []
    df_rating_by_user = df_rating.groupby("uind")
    for uind in range(num_user):
        mat_cat_cur = np.array([mat_item_cat[iind] for iind in df_rating_by_user.get_group(uind)["iind"].values])
        num_cat_cur = len(np.nonzero(np.sum(mat_cat_cur, axis=0))[0])
        list_ela_user.append(num_cat_cur)
    list_ela_user = np.array(list_ela_user) / max(list_ela_user)
    list_ela_user[list_ela_user == np.nan] = 0
    np.save(path_ela_user, list_ela_user)

    # Calculating Item Elasticity
    ## calculate item elasticity based on the number of corresponding categories
    list_ela_item0 = np.sum(mat_item_cat, axis=1)
    list_ela_item0 = list_ela_item0 / np.max(list_ela_item0)
    ## iterately calculate item elasticity based on user elasticity
    df_rating_by_item = df_rating.groupby("iind")
    list_ela_item1 = []
    for iind in range(num_item):
        if iind in list_iind:  # some items may do not appear in training set
            ela_item1_cur = np.sum([list_ela_user[uind] for uind in df_rating_by_item.get_group(iind)["uind"].values])
        else:
            ela_item1_cur = 0
        list_ela_item1.append(ela_item1_cur)
    list_ela_item1 = np.array(list_ela_item1) / max(list_ela_item1)
    list_ela_item = (list_ela_item0 + list_ela_item1) / 2
    list_ela_item[list_ela_item == np.nan] = 0
    np.save(path_ela_item, list_ela_item)

    print("==========================FINISH Calculating Elasticity=========================")
    return list_ela_user, list_ela_item


def cal_relevance_ori(dataset_name, mat_rating):
    """
    - Building relevance network
    Inputs:
        dataset_name: a dataset name
        mat_rating: a rating matrix
    Outputs:
        mat_rel: a relevance matrix
    """
    path_rel = os.path.join("data", dataset_name, "result", "rel_ori.npy")
    num_user, num_item = mat_rating.shape

    # Building relevance network
    ## user-item relevance
    list_mean_user = (mat_rating.sum(1) / (mat_rating != 0).sum(1)).reshape(num_user, 1)
    mat_rel_u2i = mat_rating / list_mean_user
    mat_rel_u2i[np.isnan(mat_rel_u2i)] = 0
    mat_rel_u2i = mat_rel_u2i / np.max(mat_rel_u2i)

    ## item-user relevance
    list_mean_item = (mat_rating.sum(0) / (mat_rating != 0).sum(0)).reshape(num_item, 1)
    mat_rel_i2u = mat_rating.T / list_mean_item
    mat_rel_i2u[np.isnan(mat_rel_i2u)] = 0
    mat_rel_i2u = mat_rel_i2u / np.max(mat_rel_i2u)

    ## user-user relevance
    mat_rating[mat_rating > 0] = 1
    list_num_user = np.sum(mat_rating, axis=1)
    mat_rel_u2u = np.zeros((num_user, num_user))
    for i in range(num_user):
        for j in range(i):
            num_and = np.sum(mat_rating[i] * mat_rating[j])
            mat_rel_u2u[i, j] = num_and / list_num_user[i]
            mat_rel_u2u[j, i] = num_and / list_num_user[j]
    mat_rel_u2u[np.isnan(mat_rel_u2u)] = 0
    mat_rel_u2u = mat_rel_u2u / np.max(mat_rel_u2u)

    ## item-item relevance
    list_num_item = np.sum(mat_rating, axis=0)
    mat_rel_i2i = np.zeros((num_item, num_item))
    for i in range(num_item):
        for j in range(i):
            num_and = np.sum(mat_rating[:, i] * mat_rating[:, j])
            mat_rel_i2i[i, j] = num_and / list_num_item[i]
            mat_rel_i2i[j, i] = num_and / list_num_item[j]
    mat_rel_i2i[np.isnan(mat_rel_i2i)] = 0
    mat_rel_i2i = mat_rel_i2i / np.max(mat_rel_i2i)

    mat_rel = np.concatenate(
        [np.concatenate([mat_rel_u2u, mat_rel_u2i], 1), np.concatenate([mat_rel_i2u, mat_rel_i2i], 1)], 0
    )
    np.save(path_rel, mat_rel)

    return mat_rel


def DijkstraMax(mat_rel_ori, index_cur, t):
    """
    a subfunction of JohnsonMax
    """
    print(index_cur)
    list_index = [i for i in range(mat_rel_ori.shape[0])]
    list_index.pop(index_cur)
    list_value = mat_rel_ori[index_cur]
    while len(list_index):
        value_max = np.max(list_value[list_index])
        index_max = np.argmax(list_value[list_index])
        if value_max < t:  # to improve efficiency
            break
        list_index.pop(index_max)
        for j in list_index:
            value_t = value_max * mat_rel_ori[index_max, j]
            if value_t > mat_rel_ori[index_cur, j]:
                mat_rel_ori[index_cur, j] = value_t

    return mat_rel_ori[index_cur].reshape((1, mat_rel_ori.shape[1]))


def JohnsonMax(mat_rel_ori, t):
    """
    - update relevance network
    Inputs:
        mat_rel_ori: the original relevance matrix
        t: the threshold to abandon update
    Outputs:
        mat_rel: the updated relevance matrix
    """
    pool = Pool(12)  # set the number of processes
    list_res = []
    for i in range(mat_rel_ori.shape[0]):
        list_res.append(pool.apply_async(DijkstraMax, (mat_rel_ori, i, t,)))
    pool.close()
    pool.join()

    list_rel = [res.get() for res in list_res]
    mat_rel = np.concatenate(list_rel)

    return mat_rel


def cal_relevance(dataset_name, mat_rating, t=0.05):
    """
    - Building relevance network
    - Updating relevance network
    Input:
        dataset_name: a dataset name
        mat_rating: a rating matrix
        t: the threshold to abandon update
    Output:
        mat_rel: the updated relevance matrix
    """
    path_rel_ori = os.path.join("data", dataset_name, "result", "rel_ori.npy")
    path_rel = os.path.join("data", dataset_name, "result", "rel.npy")

    if os.path.exists(path_rel):
        return np.load(path_rel)

    if os.path.exists(path_rel_ori):
        mat_rel_ori = np.load(path_rel_ori)
    else:
        mat_rel_ori = cal_relevance_ori(dataset_name, mat_rating)

    mat_rel = JohnsonMax(mat_rel_ori, t)
    np.save(path_rel, mat_rel)

    print("==========================FINISH Calculating Relevance==========================")
    return mat_rel


def pred_category(dataset_name, input, label, test, epoch, batch_size, lr):
    """
    - Train the GRU model
    - Predict the categories
    Inputs:
        dataset_name: dataset name
        input: input for GRU training
        label: label for GRU training
        test: input for category prediction
        epoch: the epoch of GRU training
    Outputs:
        pred: the predicted categories
    """
    path_pred = os.path.join("data", dataset_name, "result", "GRU_pred.npy")
    if os.path.exists(path_pred):
        return np.load(path_pred)

    dataset = TensorDataset(torch.from_numpy(input).float(), torch.from_numpy(label).float())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    dim_input = label.shape[1]
    model = nn.GRU(dim_input, dim_input, 2, batch_first=True).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Training the model
    for epoch in range(epoch):
        train_loss = 0
        for input, label in dataloader:
            output, _ = model(input.to(device))[-1]
            loss = -torch.mean(torch.log_softmax(output, 1) * label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, train_loss))

    # Predicting the categories
    pred, _ = model(torch.from_numpy(test).float().to(device))[-1]
    pred = pred.cpu().detach().numpy()
    np.save(path_pred, pred)
    return pred


def recommend(dataset, K_cat, epoch, batch_size, lr):
    """
    - Generating Recommendations
    Inputs:
        dataset: Dataset object
    Outputs:
        mat_rec: recommendation factor matrix
    """
    # Calculating elastic relevance
    mat_ela_user, mat_ela_item = cal_elasticity(dataset.dataset_name, dataset.mat_rating_train, dataset.mat_cat)
    mat_ela_user = mat_ela_user.reshape((mat_ela_user.shape[0], 1))
    mat_ela = (mat_ela_user + mat_ela_item) / 2
    mat_ela = mat_ela / np.max(mat_ela)
    mat_rel = cal_relevance(dataset.dataset_name, dataset.mat_rating_train)[: dataset.num_user, dataset.num_user :]
    mat_ela_rel = (mat_ela + mat_rel) / 2

    # Filtering out items that are not consistent with user category preference, Calculating recommendation factors
    mat_user_cat = pred_category(
        dataset.dataset_name, dataset.GRU_input, dataset.GRU_label, dataset.GRU_test_input, epoch, batch_size, lr
    )
    idx = bn.argpartition(-mat_user_cat, K_cat, axis=1)
    mat_user_cat = np.zeros_like(mat_user_cat, dtype=bool)
    mat_user_cat[np.arange(dataset.num_user)[:, np.newaxis], idx[:, :K_cat]] = True
    mat_item_cat = np.load(os.path.join("data", dataset.dataset_name, "item_cat.npy"))
    mat_filter = np.matmul(mat_user_cat, mat_item_cat.T)
    mat_filter = mat_filter > 0

    # Predicting recommendation factors
    mean_ela_rel = np.sum(mat_ela_rel) / (mat_ela_rel != 0).sum()
    mat_rec = (1 - np.abs(mat_ela_rel - mean_ela_rel)) * mat_filter
    np.save(os.path.join("data", dataset.dataset_name, "result", "pred.npy"), mat_rec)

    return mat_rec
