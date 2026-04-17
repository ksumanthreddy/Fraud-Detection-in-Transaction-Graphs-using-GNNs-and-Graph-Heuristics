import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

heuristics = pd.read_csv("output_data/graph_heuristics_features.csv")
features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
edges = pd.read_csv("data/elliptic_txs_edgelist.csv")

num_cols = features.shape[1]
features.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(num_cols - 2)]

df_full = features.merge(heuristics, on='txId')   # Merge heuristics with features to get labels

print("Dataset shape:", df_full.shape)
print(df_full['label'].value_counts())  


G = nx.DiGraph()   # Creating directed graph

for _, row in df_full.iterrows():
    G.add_node(row['txId'], label=row['label'])

for _, row in edges.iterrows():
    if row['txId1'] in G and row['txId2'] in G:
        G.add_edge(row['txId1'], row['txId2'])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Take subset of nodes for visualization
sample_nodes = list(G.nodes)[:2000]

G_sub = G.subgraph(sample_nodes).copy()

print("Subgraph nodes:", G_sub.number_of_nodes())

colors = []

for n in G_sub.nodes:
    label = G_sub.nodes[n]['label']

    if label == 1:
        colors.append('red')      #fraud
    elif label == 0:
        colors.append('blue')     #legit
    else:
        colors.append('gray')     #unknown


plt.figure(figsize=(12, 10))

pos = nx.spring_layout(G_sub, seed=42)
colors = []

for n in G_sub.nodes():
    label = G_sub.nodes[n].get('label', -1)  # safe access

    if label == 1:
        colors.append('red')
    elif label == 0:
        colors.append('blue')
    else:
        colors.append('gray')

nx.draw(
    G_sub, pos,
    node_color=colors,
    node_size=20,
    edge_color='lightgray',
    with_labels=False
)

plt.title("Transaction Graph\nRed = Fraud | Blue = Legit | Gray = Unknown")
plt.show()