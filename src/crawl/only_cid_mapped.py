import pandas as pd


def main():
    create_file()


def create_file():
    kg = pd.read_csv('./kg/kg_chebi_CID.csv')
    kg = list(zip(kg['s'], kg['p'], kg['o']))
    kgmesh = pd.read_csv('./kg/kg_mesh_CID.csv')
    kgmesh = list(zip(kgmesh['s'], kgmesh['p'], kgmesh['o']))
    kg = kg + kgmesh
    cid = 'http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'
    kg = [(s, p, o) for s, p, o, in kg if s.startswith(cid) or o.startswith(cid)]
    kg = pd.DataFrame(list(kg), columns=['s', 'p', 'o'])
    kg.to_csv('kg/processed/only_cid_mapped.csv')


if __name__ == '__main__':
    main()
