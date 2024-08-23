import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE

def get_data(filename, all_attributes):
    df = pd.read_csv(filename, index_col=False, usecols=all_attributes)[all_attributes]
    X_original = np.array(df.values)
    print('Before', X_original)
    X = X_original[~np.isnan(X_original).any(axis=1), :]
    print('After', X)
    X_out = []
    y_out = []
    for i in range(len(X) - numdays + 1):
        row = []
        for j in range(i, i + numdays - 1):
            for k in range(num_attributes):
                row.append(X[j][k])
        j = i + numdays - 1
        if (len(know_attributes) > 0):
            for k in range(len(know_attributes)):
                row.append(X[j][k])
        X_out.append(row)
        y_out.append(X[j][len(all_attributes) - 1])
    return X

'''filename='Data/Test_Data.csv'
all_attributes = ['Num','WL_lake', 'ALT1-1', 'ALT5-1']
get_data(filename, all_attributes)'''

def data_SMOTE(X, y, smote, smote_threshold):
    print('X_train_original: ', X.shape[0])
    print('y_train_original: ', y.shape)
    if (smote == False):
        X_smote = X
        y_smote = y
    else:
        y_class = []
        X_class = np.concatenate((X, y.reshape(len(y),1)), axis=1)
        for i in range(X_class.shape[0]):
            if (X_class[i][-1]>=smote_threshold):
                y_class.append(1)
            else:
                y_class.append(0)
        sm = SMOTE(k_neighbors=2)
        #sm = SMOTE(k_neighbors=2)
        X_res, y_res = sm.fit_resample(X_class, y_class)
        X_smote, y_smote = np.split(X_res, [X.shape[1]], axis=1)
        print('X_train_smote: ', X_smote.shape[0])
        print('y_train_smote: ', y_smote.shape)
    return X_smote, y_smote.flatten()

X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
y=np.array([30, 40, 50, 60, 70, 80, 90, 100])
X_out, y_out = data_SMOTE(X, y, True, 75)
for i in range(len(y_out)):
    print(X_out[i], y_out[i])
'''print('X_out', X_out)
print('y_out', y_out)
print('type X_out', type(X_out))
print('type y_out', type(y_out))'''

