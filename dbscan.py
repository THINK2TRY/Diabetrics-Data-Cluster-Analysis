import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.neighbors import NearestNeighbors
import torch
import os

from data1 import load_data, preprocessing


def to_one_hot(x):
    x = torch.from_numpy(x)
    one_hot_x = []
    for i in range(x.shape[1]):
        _x = torch.zeros((x.shape[0], x[:, i].max()+1))
        _x[:, x[:, i]] = 1
        one_hot_x.append(_x.numpy())
    _x = np.concatenate(one_hot_x, axis=1)
    _x = _x / np.sqrt(2)
    return _x


def _dist(p, q, metric="euclidean"):
    if metric == "euclidean":
        return np.sqrt(np.power(p-q,2).sum(-1))
    elif metric == "cosine":
        p = p / np.linalg.norm(p, keepdims=True)
        q = q / np.linalg.norm(q, keepdims=True)
        return 1 - (p * q).sum(1)
    else:
        raise NotImplementedError


def find_neighbor(point_id, m, eps, metric="euclidean"):
    vec = m[point_id, :].reshape(1, -1)
    dist = _dist(vec, m, metric=metric)
    seeds = np.where(dist <= eps)[0]
    return set(seeds.tolist())


def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类

    nfunc = NearestNeighbors(n_neighbors=min_Pts).fit(X)
    neighbor_list = nfunc.radius_neighbors(X, radius=eps, return_distance=False)
    neighbor_list = [set(x.tolist()) for x in neighbor_list]
    omega_list = [i for i, x in enumerate(neighbor_list) if len(x) >= min_Pts]

    idx = 0
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:
        print(idx)
        idx += 1
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster


if __name__ == "__main__":
    data = load_data('./diabetic_data.csv')
    print(data.shape)
    X1, X2, _ = preprocessing(data)
    # X2 = X2 / (X2.max(0, keepdims=True) + 1)
    if os.path.exists("one_hot.npy"):
        X2 = np.load("one_hot.npy")
    else:
        X2 = to_one_hot(X2)
    x = np.concatenate((X1, X2), axis=1)
    print(x.shape)
    # model = DBSCAN(eps=0.02, min_points=5, metric="euclidean")
    # results = model.fit_predict(x)
    # skDBSCAN(eps=0.4, min_samples=5).fit(x)
    results = DBSCAN(x, eps=0.35, min_Pts=3)
    results = np.array(results)
    print(results.max())
    print((results == -1).sum())
    results[results == -1] = results.max() + 1
    print(results[:100])
    np.save("./cluster_dbscan23.npy", results)