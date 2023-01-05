import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data1 import load_data, preprocessing

if __name__ == "__main__":
    plt.figure(figsize=(25, 6))
    data = load_data('./diabetic_data.csv')
    X1, X2, _ = preprocessing(data)
    x = np.concatenate((X1, X2), axis=1)
    """
    tsne = TSNE(n_components=2, method='barnes_hut')
    x = tsne.fit_transform(x)
    """
    """
    model = PCA(n_components=2)
    x = model.fit_transform(x)
    y = np.load("./cluster.npy")[:5000]
    for label in np.unique(y):
        idx = np.where(y == label)
        plt.scatter(x[idx, 0], x[idx, 1], s=5)
    plt.show()
    """
    
    y = np.load("./cluster.npy")
    k = len(np.unique(y))
    center = np.zeros((k, X1.shape[1]))
    for label in range(k):
        idx = np.where(y == label)
        center[label] = np.mean(X1[idx], axis=0)
    print(center, k)
    std = np.std(center, axis=0)
    idx = np.argsort(std)[-5:]
    print(std, idx)
    title = ['diag3', 'number_diagnoses', 'admission_type_id', 'time_in_hospital', 'num_procedures']
    for i in range(5):
        ax = plt.subplot2grid((2, 5), (0, i))
        for j in range(k):
            x = X1[np.where(y == j), idx[i]][0]
            print(x)
            bins = np.zeros(4, dtype=float)
            for val in x:
                pos = int(val * 4)
                if pos >= 4:
                    pos = 3
                bins[pos] += 1
            bins /= len(x)
            #print(np.linspace(0, 1, 100))
            #ax.plot(np.linspace(0, 1, 10), bins, label="cluster" + str(j))
            #plt.xticks(np.linspace(0, 1, 4), ["[%.2lf, %.2lf)" % (i * 0.25, i * 0.25 + 0.25) for i in range(4)])
            ax.bar(np.linspace(0, 1, 4) - 0.075 + 0.05 * j, bins, width=0.05, label="cluster" + str(j))
        ax.set_title(title[i])
        #ax.legend()

    idx = [0, 1, 2, 7, -8]
    title = ['race', 'gender', 'age', 'A1CResult', 'insulin']
    xlabel = [['?', 'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'],
        ['Female', 'Male', 'Unknown/Invalid'],
        ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
        ['>7', '>8', 'None', 'Norm'],
        ['Down', 'No', 'Steady', 'Up']]
    rotation = [30, 20, 45, 0, 0]
    ha = ["right", "right", "right", "center", "center"]
    for i in range(5):
        ax = plt.subplot2grid((2, 5), (1, i))
        for j in range(k):
            N = X2[:, idx[i]].max() + 1
            x = X2[np.where(y == j), idx[i]][0]
            print(x, x.max())
            bins = np.zeros(N, dtype=float)
            for val in x:
                bins[int(val)] += 1
            bins /= len(x)
            print(bins)
            #print(np.linspace(0, 1, 100))
            #ax.plot(np.linspace(0, 1, 10), bins, label="cluster" + str(j))
            #plt.xticks(np.linspace(0, 1, 4), ["[%.2lf, %.2lf)" % (i * 0.25, i * 0.25 + 0.25) for i in range(4)])
            w = 0.8 / 5
            ax.bar(np.linspace(0, N - 1, N) + w * j - w * 2, bins, width=w, label="cluster" + str(j))
        ax.set_xticks(range(len(xlabel[i])))
        ax.set_xticklabels(xlabel[i], rotation=rotation[i], ha=ha[i])
        ax.set_title(title[i])
        #ax.legend()
    plt.show()
        