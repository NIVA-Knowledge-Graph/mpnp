import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


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
