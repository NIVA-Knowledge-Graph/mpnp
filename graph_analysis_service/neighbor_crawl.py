import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collections


def main():
    crawl_all()


def crawl_relevant():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))

    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))

    kg = kg + kgmesh
    kg = set(kg)  # remove duplicates

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    #dfte = pd.read_csv('./data/LC50_test_CID.csv').dropna()
    #dfte = list(zip(dfte['cid'], dfte['y']))

    train = [cid for cid, y in dftr]
    #test = [cid for cid, y in dfte]

    cids = list(set(train) ) #| set(test))

    subject = [(s, p, o) for s, p, o in kg if o in cids]
    h1 = set([s for s, p, o in subject])
    find1 = find_neighbors(kg, h1)

    object = [(s, p, o) for s, p, o in kg if
              s in cids and o != 'http://www.biopax.org/release/biopax-level3.owl#SmallMolecule']
    h2 = set([o for s, p, o in object])
    find2 = find_neighbors(kg, h2)

    pass
    all = find1 + find2 + subject + object
    all = set(all)

    all = pd.DataFrame(list(all), columns=['s', 'p', 'o'])
    all.to_csv('kg/relevant_neighbors_only_train.csv')
    pass


def crawl_all():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))

    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))

    kg = kg + kgmesh
    kg = set(kg)  # remove duplicates

    cidsSub = [s for s, p, o in kg if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    cidsObj = [o for s, p, o in kg if o.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    cids = cidsSub + cidsObj

    subject = [(s, p, o) for s, p, o in kg if o in cids]
    h1 = set([s for s, p, o in subject])
    find1 = find_neighbors(kg, h1)

    object = [(s, p, o) for s, p, o in kg if
              s in cids and o != 'http://www.biopax.org/release/biopax-level3.owl#SmallMolecule']
    h2 = set([o for s, p, o in object])
    find2 = find_neighbors(kg, h2)

    pass
    all = find1 + find2 + subject + object
    all = set(all)

    all = pd.DataFrame(list(all), columns=['s', 'p', 'o'])
    all.to_csv('kg/neighbors.csv')
    pass


explored = set([])


def find_neighbors(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored

    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o in connected_by_objects]
    results_when_searching_subjects = find_neighbors(kg, set(s) - explored)

    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o in connected_by_subject]
    results_when_searching_objects = find_neighbors(kg, set(o) - explored)

    print(type(connected_by_objects))
    print(type(connected_by_subject))
    print(type(results_when_searching_objects))
    print(type(results_when_searching_subjects))
    return connected_by_objects + connected_by_subject + results_when_searching_objects + results_when_searching_subjects


if __name__ == '__main__':
    main()