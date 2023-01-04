import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data1 import load_data, preprocessing

if __name__ == "__main__":
    data = load_data('./diabetic_data.csv')
    X1, X2 = preprocessing(data)
    x = np.concatenate((X1, X2), axis=1)
    model = PCA(n_components=2)
    x = model.fit_transform(x)
    y = np.load("./cluster.npy")
    for label in np.unique(y):
        idx = np.where(y == label)
        plt.scatter(x[idx, 0], x[idx, 1])