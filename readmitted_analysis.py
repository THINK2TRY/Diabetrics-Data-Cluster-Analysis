import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def score(x, y):
    n = x.shape[0]
    c = max(x)+1
    l = max(y)+1
    onehot_x = pd.get_dummies(x).values
    onehot_y = pd.get_dummies(y).values
    print(c)
    print(l)
    M = np.zeros(c,l)
    for i in range(n):
        M = M + np.outer(onehot_y[i,:], onehot_x[i,:])
    return M,sum(np.max(M, axis=1))/n

if __name__ == "__main__":
    df = pd.read_csv('./readmitted_data.csv')
    readmitted = df.to_numpy().ravel()
    print(readmitted.shape)

    y = np.load("./cluster.npy")
    print(y.shape)

    M,p = score(readmitted, y)
    print("purity score:")
    print(p)
    print(np.max(M,axis=1)/np.sum(M,axis=1))
    e = np.zeros(M.shape[0])
    e0 = -0.54*np.log(0.54) - 0.11*np.log(0.11) - 0.35*np.log(0.35)
    for i in range(M.shape[0]):
        e[i] = -(0.54*np.log(M[i,0]/sum(M[i,:]))+ 0.11*np.log(M[i,1]/sum(M[i,:]))+ 0.35*np.log(M[i,2]/sum(M[i,:]))) - e0
    print("entropy score")
    print(e)
