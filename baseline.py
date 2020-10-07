import numpy as np
import random
import os


def baseline_acc(dataset_name):
    """
    - Accuracy-based method
    """
    path_acc = os.path.join("data", dataset_name, "result", "acc.npy")
    if os.path.exists(path_acc):
        return np.load(path_acc)

    mat_rel = np.load(os.path.join("data", dataset_name, "result", "rel.npy"))
    mat_rating_train = np.load(os.path.join("data", dataset_name, "train", "ratings.npy"))
    num_user = mat_rating_train.shape[0]
    mat_rel_u2i = mat_rel[:num_user, num_user:]
    mat_rel_u2i[mat_rating_train > 0] = 0

    np.save(path_acc, mat_rel_u2i)
    return mat_rel_u2i


def baseline_rand(dataset_name, K):
    """
    - Random-based method
    """
    mat_rating_train = np.load(os.path.join("data", dataset_name, "train", "ratings.npy"))
    mat_rating_train[mat_rating_train > 0] = 1
    mat_rating_train = mat_rating_train - 1
    num_user = mat_rating_train.shape[0]
    mat_rand = np.zeros_like(mat_rating_train)
    for uind in range(num_user):
        list_non = np.nonzero(mat_rating_train[uind])[0]
        list_rec = list_non[random.sample([i for i in range(len(list_non))], k=K)]
        mat_rand[uind, list_rec] = 1

    return mat_rand
