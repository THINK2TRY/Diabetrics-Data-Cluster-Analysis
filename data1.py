import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    df = pd.read_csv(path)
    data = df.to_numpy()[:, 2:].astype(str)
    return data

def preprocessing(data):
    categories_data_cols = [0,1,2,3,8,9,20,21] + list(range(22,46+1))
    value_data_cols = [4,5,6,7,10,11,12,13,14,15,19]
    # value data no missing data
    value_like_cols = [16,17,18]

    data_new = data[:,value_data_cols].astype(np.float64)
    #print(data_new)

    for col in value_like_cols:
        for row in range(data.shape[0]):
            tmp = data[row, col]
            if not tmp.isalpha() and not tmp.isdigit():
                data[row, col] = tmp.strip('>').strip('V').strip('E')
        
        #print(col, data[:, col])
        v = data[:,col]
        NA_index = np.where(v == '?')
        complete_index = np.where(v != '?')
        v[complete_index] = v[complete_index].astype(np.float64)
        mean = np.mean(v[complete_index].astype(np.float64))
        v[NA_index] = mean
        data[:, col] = v 
        data_new = np.insert(data_new, -1, data[:,col], axis=1)
    data_new = MinMaxScaler().fit_transform(data_new)
    #print(data_new)

    cat_data = []
    flags = []
    for col in categories_data_cols:
        onehot_code = pd.get_dummies(data[:,col]).values
        keys = pd.get_dummies(data[:,col]).columns.tolist()
        #print(keys)
        flag = -1
        for i, k in enumerate(keys):
            if k == '?' or k == 'None':
                flag = i
        flags.append(flag)
        #print(np.argmax(onehot_code, axis=1))
        cat_data.append(np.argmax(onehot_code, axis=1))

    return data_new, np.transpose(np.array(cat_data)), flags

if __name__ == "__main__":
    data = load_data('./diabetic_data.csv')
    print(data.shape)
    data1, data2, NAcols = preprocessing(data)
    print(data1.shape, data2.shape)