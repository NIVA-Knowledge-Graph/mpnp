import pickle
from asyncio import sleep
import pandas as pd
from pubchempy import get_compounds
from re import search

def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def inchikey2cid(ids, mapping):
    out = {}
    for c in ids:
        if c in mapping:
            continue
        sleep(0.2)
        try:
            r = get_compounds(str(c), 'inchikey')
            r = r.pop()
            r = r.to_dict(properties=['cid'])
            mapping[c] = r['cid']
        except:
            mapping[c] = 'no mapping'
    return mapping


def create_dict():
    mapping = {}
    try:
        mapping = load_obj("inchikey2cid")
    except:
        pass
    test = pd.read_csv('./data/LC50_test.csv')
    train = pd.read_csv('./data/LC50_train.csv')
    test = test['inchikey'].tolist()
    train = train['inchikey'].tolist()
    ids = set(train) | set(test)
    inchikey2cid(ids, mapping)
    save_obj(mapping, "inchikey2cid")


def create_dict_kg():
    mapping = {}
    try:
        mapping = load_obj("inchikey2cid")
    except:
        pass
    kg = pd.read_csv('./kg/kg_mesh.csv')
    # kg = pd.read_csv('./kg/kg_chebi.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kg = [s for s, p, o, in kg if not search('http://', s)]
    # kg = kg[:500]
    # kg = [s for s, p, o, in kg if p == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']
    ids = set(s for s in kg)

    inchikey2cid(ids, mapping)
    save_obj(mapping, "inchikey2cid")


def convert_csv():
    prefix = 'http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'
    test = pd.read_csv('./data/LC50_test.csv')
    train = pd.read_csv('./data/LC50_train.csv')
    mapping = load_obj("inchikey2cid")
    for i, row in test.iterrows():
        test.loc[i, 'inchikey'] = prefix + str(mapping[test.loc[i, 'inchikey']])
    for i, row in train.iterrows():
        train.loc[i, 'inchikey'] = prefix + str(mapping[train.loc[i, 'inchikey']])
    test.drop(test.columns[0], axis=1, inplace=True)
    train.drop(train.columns[0], axis=1, inplace=True)
    test.to_csv('data/LC50_test_CID.csv')
    train.to_csv('data/LC50_train_CID.csv')


def convert_csv_kg():
    prefix = 'http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'
    kg = pd.read_csv('./kg/kg_mesh.csv')
    mapping = load_obj("inchikey2cid")
    for i, row in kg.iterrows():
        if not search('http://', kg.loc[i, 's']): # if kg.loc[i, 'p'] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
            kg.loc[i, 's'] = prefix + str(mapping[kg.loc[i, 's']])
    kg.drop(kg.columns[0], axis=1, inplace=True)
    kg.to_csv('kg/kg_mesh_CID.csv')


    # kg = [s for s, p, o, in kg if not search('http://', s)]

def main():
    # create_dict()
    # create_dict_kg()
    # convert_csv()
    # convert_csv_kg()

    convert_csv_kg()
    #create_dict_kg()


if __name__ == '__main__':
    main()
