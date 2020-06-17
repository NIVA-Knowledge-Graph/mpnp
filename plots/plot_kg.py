import matplotlib.pyplot as plt
import pandas as pd

from pubchempy import Compound, get_compounds

import networkx as nx

#get_compounds(str("hei"), 'inchikey')

#kg = pd.read_csv('././kg/kg_chebi_CID.csv')
#kg = list(zip(kg['s'], kg['p'], kg['o']))
#kg = kg[:100]


#kg_mesh = pd.read_csv('./kg/kg_mesh_CID.csv')
#kg_mesh = list(zip(kg_mesh['s'], kg_mesh['p'], kg_mesh['o']))

#kg_mesh = kg_mesh[:1000]

#kg = kg + kg_mesh



#objects = [o for s, p, o in kg]
#subjects = [s for s, p, o in kg]

#entities = set([s for s, p, o in kg]) | set([o for s, p, o in kg])
#relations = [p for s, p, o in kg]


subjects = ['A']
objects = ['B']
relations = ['C']
kg_df = pd.DataFrame({'source': subjects, 'target': objects, 'edge': relations})

G = nx.from_pandas_edgelist(kg_df[kg_df['edge'] == "C"],
                            "source", "target",
                            edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12, 12))

pos = nx.spring_layout(G, k=0.05)

nx.draw(G, with_labels=False, node_color='skyblue', node_size=10, edge_cmap=plt.cm.Blues, pos=pos)
plt.show()
