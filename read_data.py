import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE

def remove_missing_data(filename, all_attributes, name_ref_column, numdays, afterdays):
    columns = name_ref_column + all_attributes
    df = pd.read_csv(filename, index_col=False, usecols=columns, encoding='utf-8')[columns]
    df = df.dropna()
    X = np.array(df.values)
    if (len(name_ref_column)!=0):
        start_day = numdays + afterdays -1
        ref_column = X[start_day:, 0]
        X = X[:, 1:]
    else:
        ref_column=[]
    return X, ref_column

def data_formating(X, all_attributes, attributes, know_attributes, numdays, afterdays):
    num_attributes = len(attributes)
    #print('num_attributes:', num_attributes)
    #print('number colums:', num_attributes)
    X_out=[]
    y_out=[]
    for i in range(len(X) - numdays - afterdays + 1):
        row = []
        for j in range(i, i + numdays):
            for k in range(num_attributes):
                row.append(X[j][k])
        if(len(know_attributes) > 0):
            for j in range(i + numdays, i + numdays + afterdays):
                for k in range(len(know_attributes)):
                    row.append(X[j][k])
        X_out.append(row)
        y_out.append(X[i+numdays + afterdays - 1][len(all_attributes)-1])

    return np.array(X_out), np.array(y_out)

def data_normalizing(X_train, y_train, X_test, y_test, normalize=False):
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    if (normalize == False):
        X_train_normalize, y_train_normalize = X_train, y_train
        X_test_normalize, y_test_normalize = X_test, y_test
    else:
        X_train_normalize = scalerX.fit_transform(X_train)
        y_train_normalize = scalerY.fit_transform(y_train.reshape(len(y_train), 1))
        X_test_normalize = scalerX.transform(X_test)
        y_test_normalize = scalerY.transform(y_test.reshape(len(y_test), 1))
    return X_train_normalize, y_train_normalize.flatten(), X_test_normalize, y_test_normalize.flatten(), scalerX, scalerY

def data_SMOTE(X, y, smote, smote_threshold):
    print('X_train_original: ', X.shape[0])
    print('y_train_original: ', y.shape)
    if (smote == False):
        X_smote = X
        y_smote = y
    else:
        y_class = []
        X_class = np.concatenate((X, y.reshape(len(y), 1)), axis=1)
        for i in range(X_class.shape[0]):
            if (X_class[i][-1]>=smote_threshold):
                y_class.append(1)
            else:
                y_class.append(0)
        sm = SMOTE(sampling_strategy=1, k_neighbors=6)
        X_res, y_res = sm.fit_resample(X_class, y_class)
        X_smote, y_smote = np.split(X_res, [X.shape[1]], axis=1)
        print('X_train_smote: ', X_smote.shape)
        print('y_train_smote: ', y_smote.shape)
    return X_smote, y_smote.flatten()

def get_data(train_file, test_file, all_attributes, attributes, know_attributes, numdays, afterdays, normalize, smote, smote_threshold, name_ref_column):
    Data_train, train_ref_column = remove_missing_data(train_file, all_attributes, name_ref_column, numdays, afterdays)
    X_train_original, y_train_original = data_formating(Data_train, all_attributes, attributes, know_attributes, numdays, afterdays)
    print('y_train', y_train_original)
    X_train_smote, y_train_smote = data_SMOTE(X_train_original, y_train_original, smote, smote_threshold)

    Data_test, test_ref_column = remove_missing_data(test_file, all_attributes, name_ref_column, numdays, afterdays)
    X_test_original, y_test_original = data_formating(Data_test, all_attributes, attributes, know_attributes, numdays, afterdays)

    X_train, y_train, X_test, y_test, scalerX, scalerY = data_normalizing(X_train_smote, y_train_smote, X_test_original, y_test_original, normalize)

    return X_test_original, y_test_original, X_train, y_train, X_test, y_test, scalerX, scalerY, test_ref_column


''''X=np.array([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [17, 18, 19, 20]])
y=np.array([10, 10, 20, 30, 40, 50, 50])
X_out, y_out = data_SMOTE(X, y, True, 35)
for i in range(len(y_out)):
    print(X_out[i], y_out[i]
print('X_out', X_out)
print('y_out', y_out)
print('type X_out', type(X_out))
print('type y_out', type(y_out))'''''

