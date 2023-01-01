import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

#   对数据作预处理
#   将数据的列分为以下4个类别：
#   类别数据、值数据、类值数据、区间数据
#   类别数据：采用onehot编码，变成更高维的boolean值
#   值数据：直接保留
#   类值数据：类似'字母+值'，'>+值'，将字母和符号去掉，保留值
#   区间数据：保留左区间端点

#   缺失数据的填补：采用全部有效数据的平均值

def load_data(path):
    df = pd.read_csv(path)
    data = df.to_numpy()[:, 2:].astype(str)
    return data

def preprocessing(data):
    categories_data_cols = [0,1,8,9] + list(range(22,46+1))
    value_data_cols = [4,5,6,7,10,11,12,13,14,15,19]
    # value data no missing data
    value_like_cols = [16,17,18,20,21]
    interval_data_cols = [2,3]

    data_new = data[:,value_data_cols]

    for col in value_like_cols:
        for row in range(data.shape[0]):
            tmp = data[row, col]
            if not tmp.isalpha() and not tmp.isdigit():
                data[row, col] = tmp.strip('>').strip('V').strip('E')
        
        if col < 20:
            v = data[:,col]
            NA_index = np.where(v == '?')
            complete_index = np.where(v != '?')
            mean = np.mean(v[complete_index].astype(np.float64))
            v[NA_index] = mean
            data[:, col] = v 
        data_new = np.insert(data_new, -1, data[:,col], axis=1)

    for col in interval_data_cols:
        for row in range(data.shape[0]):
            tmp = data[row, col]
            if tmp.find('[') != -1:
                pattern = r'\[(.*?)-'
                data[row, col] = re.findall(pattern, tmp)[0]
            else:
                data[row, col] = tmp.strip('>')
        
        v = data[:,col]
        NA_index = np.where(v == '?')
        complete_index = np.where(v != '?')
        mean = np.mean(v[complete_index].astype(np.int32))
        v[NA_index] = mean
        data[:, col] = v
        data_new = np.insert(data_new, -1, data[:,col], axis=1)
                
        
    for col in categories_data_cols:
        onehot_code = pd.get_dummies(data[:,col]).values
        # first col means NAN col
        complete_index = np.where(onehot_code[:,0] == 0)
        NA_index = np.where(onehot_code[:,0] == 1)
        onehot_code = onehot_code[:,1:]
        for col in range(onehot_code.shape[1]):
            tmp = onehot_code[:,col]
            # use mean value to fill up NA categories data
            mean = np.mean(tmp[complete_index])
            tmp[NA_index] = mean
            onehot_code[:,col] = tmp
        data_new = np.append(data_new, onehot_code, axis=1)

    return data_new

if __name__ == "__main__":
    data = load_data('./diabetic_data.csv')
    print(data.shape)
    data = preprocessing(data)
    print(data.shape)