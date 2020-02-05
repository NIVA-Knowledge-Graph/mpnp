import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collections


def main():
    create_data_structure()


def create_data_structure():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    #kg = [(s, p, o) for s, p, o, in kg if o != 'http://www.biopax.org/release/biopax-level3.owl#SmallMolecule']
    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    from collections import Counter
    c = Counter(s)
    print(c)


    c3 = Counter(o)
    print(c3)
    c3 = sorted(c3.items(), key=lambda x: x[1], reverse=True)
    print(c3)

    # cut the top?

    #kg = kg[:200]
    listOfv = [{'score': 0.0, 'touched': 0.0} for _ in range(len(kg))]
    kg_dict = dict(zip(kg, listOfv))

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    xtr, ytr = list(dftr['cid']), list(dftr['y'])
    xtr, ytr = list(xtr), np.asarray(ytr).reshape((-1, 1))
    median = np.median(ytr)

    ytr = ytr - median
    counter = 0
    for chemical, score in zip(xtr, ytr):
        print("progress: " + str(round((counter/len(xtr)*100),2)) + "%")
        counter += 1
        neighbors = find_neighbors_and_reset(kg, {chemical})
        pass
        for neighbor in neighbors:
            kg_dict[neighbor] = {'score': kg_dict[neighbor]['score'] + score[0],
                                 'touched': kg_dict[neighbor]['touched'] + 1}
            pass
        pass

    kg_dict = {k: {'score': np.abs(v['score']), 'touched': v['touched']} for k, v in kg_dict.items()}
    kg_dict = sorted(kg_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    kg_dict = [(sublist[0], sublist[1], sublist[2], v['score'], v['touched']) for sublist, v in kg_dict]
    kg = pd.DataFrame(kg_dict, columns=['s', 'p', 'o', 'score', 'touched'])
    kg.to_csv('kg/sorted_kg_w_touched4.csv')


explored = set([])


def find_neighbors_and_reset(kg, to_be_explored):
    #print("removing the explored")
    pass
    global explored
    explored = set([])
    return find_neighbors(kg, to_be_explored)
    #print("removing the explored")


def find_neighbors(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o in connected_by_objects]
    results_when_searching_subjects = find_neighbors(kg, set(s) - explored) # this returns someting i dont use
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o in connected_by_subject]
    #print("just a stop")
    results_when_searching_objects = find_neighbors(kg, set(o) - explored)
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


if __name__ == '__main__':
    main()
