import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


classes = pd.read_csv('data/elliptic_txs_classes.csv')
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

classes = classes.rename(columns={'class': 'label'})

print("Shape of classes data:", classes.shape)
print("No of accounts :", classes['txId'].nunique())
print(classes['label'].value_counts())

#distrubution plot for labels
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=classes, palette='viridis', hue='label', legend=False)
plt.title('Distribution of Transaction Labels (Consolidated EDA)')
plt.xlabel('Label (1=Illicit, 2=Licit, unknown=Not Labeled)')
plt.ylabel('Count')
plt.show()

features.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(features.shape[1] - 2)]
print("Shape of features data:", features.shape)
print("No of accounts :", features['txId'].nunique())
print("No of features :", features.shape[1] - 2)

# Time step distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='time_step', data=features, palette='viridis', hue='time_step', legend=False)
plt.title('Distribution of Transactions Across Time Steps')
plt.xlabel('Time Step')
plt.ylabel('Number of Transactions')
plt.show()


# graph creation
print("Shape of edges data:", edges.shape)
print("Total edges:", len(edges))

G=nx.from_pandas_edgelist(
    edges,
    source='txId1',
    target='txId2',
    create_using=nx.DiGraph()
)

print("Total nodes in graph:", G.number_of_nodes())

sample_nodes = list(G.nodes())[:500]

G_sub=G.subgraph(sample_nodes)

print("Nodes in sample graph:", G_sub.number_of_nodes())
print("Edges in sample graph:", G_sub.number_of_edges())


# Graph visualization
plt.figure(figsize=(12,10))

pos=nx.spring_layout(G_sub, seed=42)

nx.draw_networkx_nodes(
    G_sub,
    pos,
    node_size=20
)

nx.draw_networkx_edges(
    G_sub,
    pos,
    alpha=0.3
)

plt.title("Basic Transaction Graph Visualization (500 Nodes)")
plt.axis("off")
plt.show()
