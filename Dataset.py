"""
Dataset.py
Description: Loading and preprocessing dataset
Author: Xueqi Li (lee_xq@hnu.edu.cn)
Date: 2020.09.22
"""

import os
from operator import itemgetter
from collections import Counter
from multiprocessing import Pool

import pandas as pd
import numpy as np


class Dataset(object):
    def __init__(self, dataset_name, len_cat, t_user=20, num_test=20):
        """
        - Initializing Dataset
        Inputs:
            dataset_name: the selected dataset name
            t_user: filter out users with less than t_user rated items
            num_test: the number of items for test for each user
            len_cat: the length of category sequence for GRU training
        """
        # fmt:off
        self.list_category = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                              "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                              "Sci-Fi", "Thriller", "War", "Western"]
        # fmt:on
        self.dataset_name = dataset_name
        self.mat_rating_train, self.mat_rating_test = self.split_train_test(num_test, t_user)
        self.num_user, self.num_item = self.mat_rating_train.shape
        self.mat_cat = self.pre_category()
        self.GRU_input, self.GRU_label, self.GRU_test_input, self.GRU_test_label = self.create_category_sequence(len_cat)

    def pre_dataset(self, t_user):
        """
        - Filtering users with less than 20 ratings
        - Creating index for users and items
        Inputs:
            t_user: filter out users with less than t_user rated items
        Outputs:
            df_rating_index: a DataFrame object with index for users and items
        """
        path_rating = os.path.join("data", self.dataset_name, "ratings_index.csv")
        if os.path.exists(path_rating):
            return pd.read_csv(path_rating)

        # Loading rating dataset
        df_rating = pd.read_csv(os.path.join("data", self.dataset_name, "ratings.csv"))
        df_rating = df_rating.drop_duplicates(["userId", "itemId"])

        # Filtering out users with less than 20 ratings
        df_rating = df_rating.groupby("userId").filter(lambda x: len(x) > t_user)

        # Creating index for users and items
        list_uidset = sorted(list(set(df_rating["userId"].values)))
        list_iidset = sorted(list(set(df_rating["itemId"].values)))
        dict_uid_uind = dict(zip(list_uidset, [i for i in range(len(list_uidset))]))
        dict_iid_iind = dict(zip(list_iidset, [i for i in range(len(list_iidset))]))
        list_uid = df_rating["userId"].values
        list_iid = df_rating["itemId"].values
        list_rating = df_rating["rating"].values
        list_timestamp = df_rating["timestamp"].values
        list_uind = itemgetter(*list_uid)(dict_uid_uind)
        list_iind = itemgetter(*list_iid)(dict_iid_iind)
        df_ratings_index = pd.DataFrame(
            {
                "userInd": list_uind,
                "itemInd": list_iind,
                "rating": list_rating,
                "timestamp": list_timestamp,
                "userId": list_uid,
                "itemId": list_iid,
            }
        )
        df_rating_index = df_ratings_index.sort_values(by=["userInd", "timestamp"])
        df_rating_index.to_csv(path_rating, index=False)

        print("==========================FINISH Preprocessing Dataset==========================")
        return df_rating_index

    def split_train_test(self, num_test, t_user):
        """
        - Spliting dataset into training set and test set by timestamp
        Inputs:
            num_test: the number of items for test for each user
            t_user: filter out users with less than t_user rated items
        Outputs:
            mat_train: a rating matrix for training
            mat_test: a rating matrix for test
        """
        path_train = os.path.join("data", self.dataset_name, "train", "ratings.npy")
        path_test = os.path.join("data", self.dataset_name, "test", "ratings.npy")

        if os.path.exists(path_train) and os.path.exists(path_test):
            return np.load(path_train), np.load(path_test)

        # Loading preprocessed dataset
        df_rating = self.pre_dataset(t_user)
        list_col = df_rating.columns.tolist()
        list_uind = df_rating["userInd"].values.tolist()
        counter_uind = Counter(list_uind)
        num_user = df_rating["userInd"].nunique()
        num_item = df_rating["itemInd"].nunique()
        max_rating = np.max(df_rating["rating"].values)

        # for each user, Employing the latest 20 items as test set and the others as training set
        df_train = pd.DataFrame(columns=list_col)
        df_test = pd.DataFrame(columns=list_col)
        for uind in range(num_user):
            num_test = num_test if num_test < 0.2 * counter_uind[uind] else int(0.2 * counter_uind[uind])
            num_train = counter_uind[uind] - num_test
            df_train = df_train.append(df_rating[:num_train])
            df_test = df_test.append(df_rating[num_train : counter_uind[uind]])
            df_rating = df_rating[df_rating["userInd"] > uind]
        path_train_ = os.path.join("data", self.dataset_name, "train")
        path_test_ = os.path.join("data", self.dataset_name, "test")
        if not os.path.exists(path_train_):
            os.mkdir(path_train_)
        if not os.path.exists(path_test_):
            os.mkdir(path_test_)
        df_train.to_csv(os.path.join(path_train_, "ratings.csv"), index=False)
        df_test.to_csv(os.path.join(path_test_, "ratings.csv"), index=False)

        # Saving as numpy matrix
        mat_train = np.zeros((num_user, num_item), dtype=np.float)
        mat_test = np.zeros((num_user, num_item), dtype=np.float)
        mat_train[df_train["userInd"].values.tolist(), df_train["itemInd"].values.tolist()] = df_train["rating"].values
        mat_test[df_test["userInd"].values.tolist(), df_test["itemInd"].values.tolist()] = df_test["rating"].values
        np.save(path_train, mat_train / max_rating)
        np.save(path_test, mat_test / max_rating)

        print("============================FINISH Splitting Dataset============================")
        return mat_train, mat_test

    def pre_category(self):
        """
        - Creating category labels for items
        Outputs:
            mat_item_cat: the item category matrix
        """
        path_cat = os.path.join("data", self.dataset_name, "item_cat.npy")
        if os.path.exists(path_cat):
            return np.load(path_cat)

        # Loading dataset
        df_item = pd.read_csv(os.path.join("data", self.dataset_name, "items.csv"))
        df_rating = pd.read_csv(os.path.join("data", self.dataset_name, "ratings_index.csv"))

        # create category labels for items
        mat_item_cat = np.zeros((self.num_item, len(self.list_category)), dtype=np.int8)
        for itemInd in range(self.num_item):
            itemId = df_rating[df_rating["itemInd"] == itemInd]["itemId"].values[0]
            str_cat = df_item[df_item["itemId"] == itemId]["genres"].values[0]
            list_catId = [self.list_category.index(cat) for cat in str_cat.split("|") if cat in self.list_category]
            mat_item_cat[itemInd, list_catId] = [1 for i in range(len(list_catId))]
        np.save(path_cat, mat_item_cat)

        print("=========================FINISH Creating Category Labels========================")
        return mat_item_cat

    def create_category_sequence_sub(self, uind, len_cat, list_iind, mat_item_cat):
        print(uind)
        mat_input = np.empty((0, len_cat, len(self.list_category)), dtype=np.float)
        mat_label = np.empty((0, len(self.list_category)), dtype=np.float)
        for i in range(len(list_iind) - len_cat):
            mat_input_cur = np.array([[mat_item_cat[iind] for iind in list_iind[:len_cat]]])
            mat_label_cur = mat_item_cat[[list_iind[len_cat]]]
            mat_input = np.concatenate([mat_input, mat_input_cur])
            mat_label = np.concatenate([mat_label, mat_label_cur])
            list_iind.pop(0)
        return mat_input, mat_label

    def create_category_sequence(self, len_cat):
        """
        - Creating category sequences as the input for GRU
        - Creating category sequences for prediction
        Inputs:
            len_cat: the length of category sequence for GRU training
        Outputs:
            mat_input: input for GRU training
            mat_label: label for GRU training
            mat_test_input: input for category prediction
            mat_test_label: label for category prediction
        """
        path_input = os.path.join("data", self.dataset_name, "train", "GRU_input.npy")
        path_label = os.path.join("data", self.dataset_name, "train", "GRU_label.npy")
        path_test_input = os.path.join("data", self.dataset_name, "test", "GRU_test_input.npy")
        path_test_label = os.path.join("data", self.dataset_name, "test", "GRU_test_label.npy")
        if (
            os.path.exists(path_input)
            and os.path.exists(path_label)
            and os.path.exists(path_test_input)
            and os.path.exists(path_test_label)
        ):
            return (
                np.load(path_input),
                np.load(path_label),
                np.load(path_test_input),
                np.load(path_test_label),
            )

        # Loading dataset and Initializing
        df_rating_train = pd.read_csv(os.path.join("data", self.dataset_name, "train", "ratings.csv"))
        df_rating_train = df_rating_train.groupby("userInd").filter(lambda x: len(x) > len_cat + 1)
        list_uind_train = list(set(df_rating_train["userInd"].values))
        df_rating_test = pd.read_csv(os.path.join("data", self.dataset_name, "test", "ratings.csv"))
        mat_item_cat = np.load(os.path.join("data", self.dataset_name, "item_cat.npy"))
        mat_test_input = np.empty((self.num_user, len_cat, len(self.list_category)), dtype=np.float)
        mat_test_label = np.empty((self.num_user, len(self.list_category)), dtype=np.float)

        # Creating input and label for GRU training
        list_res = []
        pool = Pool(20)
        for uind in list_uind_train:
            list_iind = df_rating_train[df_rating_train["userInd"] == uind]["itemInd"].values.tolist()
            num_cat = len(self.list_category)
            list_res.append(pool.apply_async(self.create_category_sequence_sub, (uind, len_cat, list_iind, mat_item_cat,)))
        pool.close()
        pool.join()
        list_input, list_label = [], []
        for res in list_res:
            input, label = res.get()
            list_input.append(input)
            list_label.append(label)
        mat_input = np.concatenate(list_input)
        mat_label = np.concatenate(list_label)

        # Creating squence for category prediction
        df_rating_train = pd.read_csv(os.path.join("data", self.dataset_name, "train", "ratings.csv"))
        list_uind_train = list(set(df_rating_train["userInd"].values))
        for uind in list_uind_train:
            list_iind = df_rating_train[df_rating_train["userInd"] == uind]["itemInd"].values.tolist()
            if len(list_iind) >= 20:
                mat_test_input_cur = np.array([mat_item_cat[iind] for iind in list_iind[-len_cat:]])
            else:
                mat_test_input_cur = np.concatenate(
                    [
                        np.array([[0 for i in range(len(self.list_category))] for j in range(20 - len(list_iind))]),
                        np.array([mat_item_cat[iind] for iind in list_iind]),
                    ],
                    axis=0,
                )
            mat_test_input[uind] = mat_test_input_cur

            list_iind = df_rating_test[df_rating_test["userInd"] == uind]["itemInd"].values.tolist()
            mat_test_label[uind] = np.sum(np.array([mat_item_cat[iind] for iind in list_iind]), 0)

        np.save(path_input, mat_input)
        np.save(path_label, mat_label)
        np.save(path_test_input, mat_test_input)
        np.save(path_test_label, mat_test_label)

        print("=======================FINISH Creating GRU Input and Label======================")
        return mat_input, mat_label, mat_test_input, mat_test_label


if __name__ == "__main__":
    """
    Preprocessing dataset
    """
    dataset_name = "ml-1m"
    dataset = Dataset(dataset_name, 20)
    # mat_rating_train, mat_rating_test = dataset.mat_rating_train, dataset.mat_rating_test
    # input_GRU, label_GRU, _, _ = dataset.GRU_input, dataset.GRU_label, dataset.GRU_test_input, dataset.GRU_test_label
