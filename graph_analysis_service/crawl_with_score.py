import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collections


def main():
    create_data_structure()


def create_data_structure():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh

    #kg = kg[:200]

    kg_dict = dict(zip(kg,np.zeros(len(kg))))

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)

    ytr = ytr - median

    for chemical, score in zip(xtr, ytr):
        neighbors = find_neighbors_and_reset(kg, {chemical})
        l = len(neighbors)
        pass
        for neighbor in neighbors:
            kg_dict[neighbor] = kg_dict[neighbor] + score[0]
            pass
        pass

    kg_dict = {k: np.abs(v) for k, v in kg_dict.items()}
    kg_dict = sorted(kg_dict.items(), key=lambda x: x[1], reverse=True)
    kg_dict = [(sublist[0], sublist[1], sublist[2], score) for sublist, score in kg_dict]
    kg = pd.DataFrame(kg_dict, columns=['s', 'p', 'o', 'score'])
    kg.to_csv('kg/sorted_kg.csv')


explored = set([])


def find_neighbors_and_reset(kg, h1):
    global explored
    return find_neighbors(kg, h1)
    explored = set([])


def find_neighbors(kg, h1):
    global explored
    if not h1:
        return
    explored = explored | h1
    Subject2 = [(s, p, o) for s, p, o in kg if o in h1]
    s = [s for s, p, o in Subject2]
    find_neighbors(kg, set(s) - explored)
    Subject3 = [(s, p, o) for s, p, o in kg if s in h1]
    o = [o for s, p, o in Subject3]
    find_neighbors(kg, set(o) - explored)
    return Subject2 + Subject3





if __name__ == '__main__':
    main()
