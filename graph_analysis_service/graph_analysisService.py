import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collections


def print_graph_stat():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    # kg = kg + kgmesh
    kg = kgmesh

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

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    # kg = kg + kgmesh
    kg = kgmesh

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

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    # kg = kg + kgmesh
    kg = kgmesh

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

    print("Number of elements in both data and subjects of knowledge Graph: " + str(len(subjects_and_data)))
    print("Number of elements in both data and objects of knowledge Graph: " + str(len(objects_and_data)))

    pass


def corr_chebi_mesh():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]


def create_reduced_kg():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

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

    kg = [(s, p, o) for s, p, o, in kg if s in subjects_and_data]

    print(kg)
    print(str(len(kg)))
    kg = pd.DataFrame(kg, columns=['s', 'p', 'o'])

    kg.to_csv('kg/reduced_kg.csv')


def create_reduced_kg2():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh

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

    kg = [(s, p, o) for s, p, o, in kg if s in subjects_and_data or o in objects_and_data]

    print(kg)
    print(str(len(kg)))

    # kg = set(kg)

    kg = pd.DataFrame(kg, columns=['s', 'p', 'o'])

    kg.to_csv('kg/reduced_kg.csv')


def finddup():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh]  # if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh

    pscounter = collections.Counter(kg)
    pstopCom = pscounter.most_common(15)

    pass


def create_reduced_kg3():
    kg = pd.read_csv('../kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    # kg = [(s, p, o) for s, p, o, in kg ]#if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('../kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    # kgmesh = [(s, p, o) for s, p, o, in kgmesh] # if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    kg = kg + kgmesh

    kg = set(kg)

    entities = set([s for s, p, o in kg]) | set([o for s, p, o in kg])
    relations = set([p for s, p, o in kg])
    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    Xtr, ytr = list(dftr['cid']), list(dftr['y'])
    Xte, yte = list(dfte['cid']), list(dfte['y'])

    l1 = len((set(Xtr) | set(Xte)) - entities)
    print('Proportion of entites missing in KG:', l1 / len(set(Xtr) | set(Xte)))

    tr = [(x, y) for x, y in zip(Xtr, ytr) if x in entities]
    te = [(x, y) for x, y in zip(Xte, yte) if x in entities]

    Xtr, ytr = zip(*tr)
    Xte, yte = zip(*te)
    Xtr, ytr = list(Xtr), np.asarray(ytr).reshape((-1, 1))
    Xte, yte = list(Xte), np.asarray(yte).reshape((-1, 1))

    # convert to binary
    scaler = MinMaxScaler()
    ytr = scaler.fit_transform(ytr)
    yte = scaler.transform(yte)
    ytr = np.abs(np.around(ytr - np.median(ytr) + 0.5) - 1)  # Only care about triples with high distance from median?
    yte = np.abs(np.around(yte - np.median(yte) + 0.5) - 1)  #

    ytr = list(ytr.reshape((-1,)))
    yte = list(yte.reshape((-1,)))

    ytr = ytr + yte  # too see with test
    Xtr = Xtr + Xte

    positive = [x for x, y in zip(Xtr, ytr) if y == 1]

    negative = [x for x, y in zip(Xtr, ytr) if y == 0]

    positiveSubject = [(s, p, o) for s, p, o in kg if o in positive]
    h1 = set([s for s, p, o in positiveSubject])
    positiveSubject2 = [(s, p, o) for s, p, o in kg if o in h1]
    positiveSubject3 = [(s, p, o) for s, p, o in kg if s in h1]

    positiveObject = [(s, p, o) for s, p, o in kg if
                      s in positive and o != 'http://www.biopax.org/release/biopax-level3.owl#SmallMolecule']
    h2 = [o for s, p, o in positiveObject]
    positiveObject2 = [(s, p, o) for s, p, o in kg if s in h2]
    positiveObject3 = [(s, p, o) for s, p, o in kg if o in h2]

    # slow
    negativeSubject = [(s, p, o) for s, p, o in kg if o in negative]
    h3 = [s for s, p, o in negativeSubject]
    negativeSubject2 = [(s, p, o) for s, p, o in kg if o in h3]
    negativeSubject3 = [(s, p, o) for s, p, o in kg if s in h3]

    negativeObject = [(s, p, o) for s, p, o in kg if
                      s in negative and o != 'http://www.biopax.org/release/biopax-level3.owl#SmallMolecule']
    h4 = [o for s, p, o in negativeObject]
    negativeObject2 = [(s, p, o) for s, p, o in kg if s in h4]
    negativeObject3 = [(s, p, o) for s, p, o in kg if o in h4]

    all_negative = negativeSubject2 + negativeSubject3 + negativeSubject + negativeObject + negativeObject2 + negativeObject3
    all_positive = positiveSubject + positiveSubject2 + positiveSubject3 + positiveObject + positiveObject2 + positiveObject3

    all = all_negative + all_positive

    all = set(all)

    pass

    # undeObject = positiveObject & negativeObject

    # undeSub = positiveSubject & negativeSubject

    pocounter = collections.Counter(positiveObject)
    potopCom = pocounter.most_common(15)

    # print(kg)
    # print(str(len(all)))
    # kg = pd.DataFrame(list(all), columns=['s', 'p', 'o'])

    # kg.to_csv('kg/reduced_kg_3.csv')

    pscounter = collections.Counter(positiveSubject)
    pstopCom = pscounter.most_common(15)

    nocounter = collections.Counter(negativeObject)
    notopCom = nocounter.most_common(15)

    nscounter = collections.Counter(negativeSubject)
    nstopCom = nscounter.most_common(15)

    # kg.to_csv('kg/reduced_kg2.csv')
    pass


def compair_train_test():
    dftr = pd.read_csv('../data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('../data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = set([cid for cid, y in dftr])
    test = set([cid for cid, y in dfte])

    print("in test: " + str(len(test)))
    print("in test but not in train " + str(len(test - train)))

    pass


def main():
    # print_graph_stat()
    # find_duplicates()
    # shared_elements()

    # create_reduced_kg2()
    # compair_train_test()
    create_reduced_kg3()
    # trying_something()

    # finddup()

    #create_reduced_kg_4()


if __name__ == '__main__':
    main()
