### image recongnition of chemicals.

from rdkit import Chem
from rappt import Chemistry
import pandas as pd
from collections import defaultdict
import os
from rdflib import RDFS, Graph, RDF


def load_data(filename, mapping=None):
    suppl = Chem.SDMolSupplier(filename)
    res = {}
    for mol in suppl:
        props = mol.GetPropsAsDict()
        cas = props['CAS']
        if mapping:
            try:
                ids = mapping[cas]
            except KeyError:
                ids = cas
        else:
            ids = cas
        res[ids] = props['Tox']
    return res


chemistry = Chemistry(pubchem_directory='../pubchem/', load_rdf=False, endpoint='http://localhost:8890/sparql',
                      verbose=True)

all_chemicals = set()

d = 'TEST_datasets/'
files = os.listdir(d)
files = set([f.split('_').pop(0) for f in files])
files = set(['LC50'])

for f in files:
    tr_data = load_data(d + f + '_training.sdf')
    te_data = load_data(d + f + '_prediction.sdf')

    chemicals = set(tr_data.keys()) | set(te_data.keys())
    mapping = {c: m for c, m in zip(chemicals, chemistry.convert_ids('cas', 'inchikey', chemicals)) if
               m != 'no mapping'}

    all_chemicals |= chemicals

    for data, filename in zip([tr_data, te_data], ['_train.csv', '_test.csv']):
        tmp = {}
        tmp['inchikey'] = [mapping[k] for k in data if k in mapping]
        tmp['y'] = [data[k] for k in data if k in mapping]
        df = pd.DataFrame(data=tmp)
        df.to_csv('data/' + f + filename)

all_chemicals = list(all_chemicals)

for use_mesh, filename in zip([True, False], ['_mesh.csv', '_chebi.csv']):

    # skip mesh for now
    if use_mesh:
        continue

    kg = chemistry.get_subgraph(all_chemicals, from_='cas', to_='inchikey', use_mesh=use_mesh)

    s, p, o = zip(*list(kg))
    data = {'s': s, 'p': p, 'o': o}

    df = pd.DataFrame(data=data)
    df.to_csv('kg/kg' + filename)

    data = {}
    col1 = []
    col2 = []
    col3 = []
    for s, p, o in kg:
        if s == o: continue
        if not p in [RDFS.subClassOf, RDF.type]: continue
        col1.append(str(s))
        col2.append(str(RDFS.subClassOf))
        col3.append(str(o))

    data['s'] = col1
    data['p'] = col2
    data['o'] = col3

    df = pd.DataFrame(data=data)
    df.to_csv('kg/hier' + filename)
