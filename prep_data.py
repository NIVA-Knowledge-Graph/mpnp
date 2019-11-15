### image recongnition of chemicals.

from rdkit import Chem
from rappt import Chemistry
import pandas as pd
from collections import defaultdict
import os


def load_data(filename, mapping = None):
    suppl = Chem.SDMolSupplier(filename)
    res = {}
    for mol in suppl:
        props = mol.GetPropsAsDict()
        cas = props['CAS'].replace('-','')
        if mapping:
            try:
                ids = mapping[cas]
            except KeyError:
                ids = cas
        else:
            ids = cas
        res[ids] = props['Tox']
    return res

c = Chemistry(pubchem_directory='../pubchem/', cas_from_wikidata=True, load_rdf=True, verbose=True)

all_chemicals = set()

d = 'TEST_datasets/'
files = os.listdir(d)
files = set([f.split('_').pop(0) for f in files])

for f in files:
    tr_data = load_data(d + f + '_training.sdf')
    te_data = load_data(d + f + '_prediction.sdf')

    chemicals = set(tr_data.keys()) | set(te_data.keys())
    mapping = {c:m for c,m in zip(chemicals, c.convert_ids('cas','inchikey',chemicals)) if m != 'no mapping'}
    
    all_chemicals |= set([mapping[c] for c in chemicals if c in mapping])
    
    for data, filename in zip([tr_data, te_data], ['_train.csv','_test.csv']):
        tmp = {}
        tmp['inchikey'] = [mapping[k] for k in data if k in mapping]
        tmp['y'] = [data[k] for k in data if k in mapping]
        df = pd.DataFrame(data=tmp)
        df.to_csv('data/'+f+filename)

all_chemicals = list(all_chemicals)
ch = c.class_hierarchy(all_chemicals, from_='inchikey',to_='inchikey',use_mesh=False)   
ch |= c.class_hierarchy(all_chemicals, from_='inchikey',to_='inchikey',use_mesh=True)
data = {}
data['s'],data['o'],_ = zip(*ch)
data['p'] = ['subClassOf' for _ in ch]
df = pd.DataFrame(data=data)
df.to_csv('kg/kg.csv')



