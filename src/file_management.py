import pickle

import pandas as pd


def add_result(name, result):
    """Adds the results to a dict containing all the runs.
    The dict is saved binary using pickle, and is read when plotting the results"""
    obj = {}
    with open('./results/results.pkl', 'rb') as f:
        try:
            obj = pickle.load(f)
        except:
            pass
        obj[name] = result
    with open('./results/results.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_train_and_test_data():
    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    dfte = pd.read_csv('./data/LC50_test_CID.csv').dropna()
    Xtr, ytr = list(dftr['cid']), list(dftr['y'])
    Xte, yte = list(dfte['cid']), list(dfte['y'])
    return Xte, Xtr, yte, ytr