import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collections


def main():
    #create_reduced_kg()
    #insdlval()
    #orderbytouch()
    #print_graph_stat()
    #find_duplicates()
    shared_elements()



def orderbytouch():
    #kg = pd.read_csv('../kg/two_step.csv')
    #kg = list(zip(kg['s'], kg['p'], kg['o'], kg['score'], kg['touched']))
    #g = kg['touched'].value_counts()
    #print_predicats_from_mesh()

    pass


def insdlval():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh
    #kg = kgmesh

    s = [s for s, p, o, in kg if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID') and o == 'http://id.nlm.nih.gov/mesh/2019/D007306']
    o = [o for s, p, o, in kg if o.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID') and s == 'http://id.nlm.nih.gov/mesh/2019/D007306']
    #http://id.nlm.nih.gov/mesh/2019/D007306
    e = s + o

    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    all = dftr + dfte


    allinsd = [y for c, y in all if c in e]
    all = [y for c, y in all]

    print(np.average(all))
    print(np.average(allinsd))


def print_predicats_from_mesh():
    kg = pd.read_csv('../kg/kg_mesh_CID.csv')
    kg = list(kg['p'])
    p = set(kg)
    for pr in p:
        print(pr)
    print(p)


def find_shard_triples_from_test_and_train():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))

    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = [cid for cid, y in dftr]
    test = [cid for cid, y in dfte]


def print_graph_stat():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    #kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh
    #kg = kgmesh

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    subjects = set([s for s, p, o in kg])
    objects = set([o for s, p, o in kg])
    entities = set([s for s, p, o in kg]) | set([o for s, p, o in kg])

    leaf = objects - subjects
    roots = subjects - objects
    others = objects & subjects

    print("Number of leafs: " + str(len(leaf))
          + ", roots: " + str(len(roots))
          + ", others: " + str(len(others))
          + ". In total: " + str(len(entities)))

    print("len sub:" + str(len(subjects)) + " len object: " + str(len(objects)))


def find_duplicates():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh
    #kg = kgmesh

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    sduplicates = set([item for item, count in collections.Counter(s).items() if count > 1])
    oduplicates = set([item for item, count in collections.Counter(o).items() if count > 1])

    print("duplicates in subjects: " + str(len(sduplicates)))
    print("duplicates in objects: " + str(len(oduplicates)))

    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = [cid for cid, y in dftr]
    test = [cid for cid, y in dfte]

    subjects = set([s for s, p, o in kg])
    objects = set([o for s, p, o in kg])
    data = set(train) | set(test)

    subjects_and_data_and_dup = subjects & set(train) & (sduplicates | oduplicates)
    objects_and_data_and_dup = objects & set(train) & (sduplicates | oduplicates)

    print("duplicates in subjects and in train: " + str(len(subjects_and_data_and_dup)))
    print("duplicates in objects and in train: " + str(len(objects_and_data_and_dup)))

    subjects_and_data_and_dup = subjects & data & (sduplicates | oduplicates)
    objects_and_data_and_dup = objects & data & (sduplicates | oduplicates)

    print("duplicates in subjects and in test or train: " + str(len(subjects_and_data_and_dup)))
    print("duplicates in objects and in test or train: " + str(len(objects_and_data_and_dup)))

    subjects_and_data_and_dup = subjects & set(train) & set(test) & (sduplicates | oduplicates)
    objects_and_data_and_dup = objects & set(train) & set(test) & (sduplicates | oduplicates)

    print("duplicates in subjects and in test and train: " + str(len(subjects_and_data_and_dup)))
    print("duplicates in objects and in test and train: " + str(len(objects_and_data_and_dup)))

    pass

    # sduplicates =
    # oduplicates =

    # print("duplicates in subjects: " + str(len(sduplicates)))
    # print("duplicates in objects: " + str(len(oduplicates)))


def shared_elements():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    #kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh
    #kg = kgmesh

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = [cid for cid, y in dftr]
    test = [cid for cid, y in dfte]

    subjects = set([s for s, p, o in kg])
    objects = set([o for s, p, o in kg])
    data = set(train) | set(test)

    subjects_and_data = subjects & data
    objects_and_data = objects & data


    leaf = objects - subjects
    roots = subjects - objects
    others = objects & subjects

    roots_and_data = roots & data
    others_and_data = others & data
    leaf_and_data = leaf & data


    print("Number of elements in both data and subjects of knowledge Graph: " + str(len(subjects_and_data)))
    print("Number of elements in both data and objects of knowledge Graph: " + str(len(objects_and_data)))

    print("Number of elements in both data and roots of knowledge Graph: " + str(len(roots_and_data)))
    print("Number of elements in both data and internal of knowledge Graph: " + str(len(others_and_data)))
    print("Number of elements in both data and leaf of knowledge Graph: " + str(len(leaf_and_data)))
    pass


def corr_chebi_mesh():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]


def create_reduced_kg():

    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('./data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = [cid for cid, y in dftr]
    test = [cid for cid, y in dfte]

    subjects = set([s for s, p, o in kg])
    objects = set([o for s, p, o in kg])
    data = set(train) | set(test)

    subjects_and_data = subjects & data

    objects_and_data = objects & data

    kg = [(s, p, o) for s, p, o, in kg if s in subjects_and_data or o in objects_and_data]


    kg = set(kg)

    print(kg)
    print(str(len(kg)))

    kg = pd.DataFrame(list(kg), columns=['s', 'p', 'o'])

    kg.to_csv('kg/reduced_kg.csv')


def compare_train_test():
    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = set([cid for cid, y in dftr])
    test = set([cid for cid, y in dfte])

    print("in test: " + str(len(test)))
    print("in test but not in train " + str(len(test - train)))


if __name__ == '__main__':
    main()
