import numpy as np
import pandas as pd

def split_train_test(df):
    train_size = int(len(df) * 0.85)
    train_subset = df.sample(train_size, random_state=1)
    test_subset = df.drop(train_subset.index)
    return train_subset, test_subset

def feature_select(feature, train_X, thershold):
    feature_std = [feature[:, i].std() for i in range(feature.shape[1])]
    feature_std = np.array(feature_std)
    unique = np.array(build_fz_name(train_X.columns))
    fe = dict(zip(unique, feature_std))
    unimport = [i for i in fe.keys() if fe[i] < thershold]

    return fe, unimport , feature_std

def build_fz_name(column):
    columns = []
    for fuz in column:
        columns.append(fuz+'_fz1')
        columns.append(fuz+'_fz2')
        columns.append(fuz+'_fz3')
    return columns

def delt_unimportant(unimport):
    delt = []
    for i in range(len(unimport)):
        delt.append(unimport[i][:-4]) # 去除fz名稱
    return delt

def select_unimportant(delt):
    delete = pd.value_counts(delt)[pd.value_counts(delt)>1].index
    return delete

def important(unimport, train_X):
    delete = set(select_unimportant(delt_unimportant(unimport)))
    important = train_X.drop(columns=delete).columns.tolist()
    return delete, important