import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collections


def main():
    create_only_cid_mapped_kg()


def create_only_cid_mapped_kg():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh
    cid = 'http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'
    kg = [(s, p, o) for s, p, o, in kg if s.startswith(cid) or o.startswith(cid)]
    kg = pd.DataFrame(list(kg), columns=['s', 'p', 'o'])
    kg.to_csv('kg/only_cid_mapped_kg.csv')


if __name__ == '__main__':
    main()