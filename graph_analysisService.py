import pandas as pd
import collections


def print_graph_stat():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('./kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    #kg = kg + kgmesh
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
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('./kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    #kg = kg + kgmesh
    kg = kgmesh

    s = [s for s, p, o in kg]
    o = [o for s, p, o in kg]

    sduplicates = [item for item, count in collections.Counter(s).items() if count > 1]
    oduplicates = [item for item, count in collections.Counter(o).items() if count > 1]

    print("duplicates in subjects: " + str(len(sduplicates)))
    print("duplicates in objects: " + str(len(oduplicates)))


def shared_elements():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('./kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]
    #kg = kg + kgmesh
    kg = kgmesh

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

    print("Number of elements in both data and subjects of knowledge Graph: " + str(len(subjects_and_data)))
    print("Number of elements in both data and objects of knowledge Graph: " + str(len(objects_and_data)))

    pass

def corr_chebi_mesh():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('./kg/kg_chebi_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kgmesh = [(s, p, o) for s, p, o, in kgmesh if s.startswith('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID')]



def create_reduced_kg():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

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

    kg = [(s, p, o) for s, p, o, in kg if s in subjects_and_data]

    print(kg)
    print(str(len(kg)))
    kg = pd.DataFrame(kg, columns=['s', 'p', 'o'])

    kg.to_csv('kg/reduced_kg.csv')

def create_reduced_kg2():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [(s, p, o) for s, p, o, in kg if p != 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    kgmesh = pd.read_csv('./kg/kg_chebi_CID.csv')
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

    print(kg)
    print(str(len(kg)))
    kg = pd.DataFrame(kg, columns=['s', 'p', 'o'])

    kg.to_csv('kg/reduced_kg.csv')

def compair_train_test():
    dftr = pd.read_csv('./data/LC50_train_CID.csv').dropna()
    dftr = list(zip(dftr['cid'], dftr['y']))
    dfte = pd.read_csv('./data/LC50_test_CID.csv').dropna()
    dfte = list(zip(dfte['cid'], dfte['y']))

    train = set([cid for cid, y in dftr])
    test = set([cid for cid, y in dfte])

    print("in test: " + str(len(test)))
    print("in test but not in train " + str(len(test-train)))


    pass

def main():
    #print_graph_stat()
    #find_duplicates()
    #shared_elements()
    #create_reduced_kg2()
    compair_train_test()


if __name__ == '__main__':
    main()
