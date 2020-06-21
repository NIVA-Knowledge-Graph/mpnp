import pandas as pd


def create_file(kg):
    cid = 'http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'
    kg = [(s, p, o) for s, p, o, in kg if s.startswith(cid) or o.startswith(cid)]
    kg = pd.DataFrame(list(kg), columns=['s', 'p', 'o'])
    kg.to_csv('kg/processed/only_cid_mapped.csv')
