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

    listOfv = [{'score': 0.0, 'touched': 0.0} for _ in range(len(kg))]
    kg_dict = dict(zip(kg, listOfv))

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)

    ytr = ytr - median
    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter / len(xtr) * 100), 2)) + "%")
        counter += 1
        neighbors = find_neighbors_and_reset(kg, {chemical})
        pass
        triples = [(s, p, o) for s, p, o, step in neighbors]
        steps = [step for s, p, o, step in neighbors]
        for triple, step in zip(triples, steps):
            kg_dict[triple] = {'score': kg_dict[triple]['score'] + score[0] / step,
                               'touched': kg_dict[triple]['touched'] + 1}
            pass
        pass

    kg_dict = {k: {'score': np.abs(v['score']), 'touched': v['touched']} for k, v in kg_dict.items()}
    kg_dict = sorted(kg_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    kg_dict = [(sublist[0], sublist[1], sublist[2], v['score'], v['touched']) for sublist, v in kg_dict]
    kg = pd.DataFrame(kg_dict, columns=['s', 'p', 'o', 'score', 'touched'])
    kg.to_csv('kg/sorted_kg_w_touched_descending_influence.csv')


explored = set([])


def find_neighbors_and_reset(kg, to_be_explored):
    pass
    global explored
    explored = set([])
    return find_neighbors(kg, to_be_explored, 1)


def find_neighbors(kg, to_be_explored, step):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o, step) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o, step in connected_by_objects]
    results_when_searching_subjects = find_neighbors(kg, set(s) - explored, step + 1)
    connected_by_subject = [(s, p, o, step) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o, step in connected_by_subject]
    results_when_searching_objects = find_neighbors(kg, set(o) - explored, step + 1)
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


if __name__ == '__main__':
    main()
