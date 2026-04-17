import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

features=pd.read_csv("data/elliptic_txs_features.csv",header=None)
heuristics=pd.read_csv("output_data/graph_heuristics_features.csv")
edges=pd.read_csv("data/elliptic_txs_edgelist.csv")

# Assign column names
features.columns=['txId','time_step']+[f'f{i}' for i in range(features.shape[1]-2)]

# Merge node features + graph features
df=features.merge(heuristics,on='txId',how='inner')

# Remove unlabeled
df=df[df['label']!=-1]

print("Final shape:",df.shape)
print(df['label'].value_counts())

X_df=df.drop(columns=['txId','label'])
y_df=df['label']

print("Features:",X_df.shape)

# Convert to tensors
X=torch.tensor(X_df.values,dtype=torch.float)
y=torch.tensor(y_df.values,dtype=torch.long)

# Map txId → node index (important for graph)
tx_ids=df['txId'].values
node_map={tx:i for i,tx in enumerate(tx_ids)}

# Build edge index using mapped indices
edge_list=[]
for _,row in edges.iterrows():
    u,v=row['txId1'],row['txId2']
    if u in node_map and v in node_map:
        edge_list.append([node_map[u],node_map[v]])

edge_index=torch.tensor(edge_list,dtype=torch.long).t().contiguous()
print("Edges:",edge_index.shape)

# Train test split (node-level)
idx=np.arange(len(y))
train_idx,test_idx=train_test_split(
    idx,test_size=0.25,stratify=y.numpy(),random_state=42
)

# Masks instead of splitting graph
train_mask=torch.zeros(len(y),dtype=torch.bool)
test_mask=torch.zeros(len(y),dtype=torch.bool)

train_mask[train_idx]=True
test_mask[test_idx]=True

# Class weights (manual imbalance handling)
class_weights=torch.tensor([1.0,2.0])
print("Class weights:",class_weights)

# Graph data object
data=Data(
    x=X,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

# GCN model
class GCN(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1=GCNConv(in_channels,128)
        self.conv2=GCNConv(128,64)
        self.out=torch.nn.Linear(64,2)

    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=self.conv2(x,edge_index)
        x=F.relu(x)
        return self.out(x)

model=GCN(data.num_features)

optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights)

# Training loop
for epoch in range(60):
    model.train()
    optimizer.zero_grad()

    out=model(data)

    # Loss only on training nodes
    loss=loss_fn(out[data.train_mask],data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch%5==0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#Saving model
torch.save(model.state_dict(), "gnn_model.pth")

# Evaluation
model.eval()

out=model(data)
probs=torch.softmax(out,dim=1)

#threshold
threshold=0.5
preds=(probs[:,1]>threshold).long()

print(classification_report(
    data.y[data.test_mask].cpu(),
    preds[data.test_mask].cpu()
))

# Confusion matrix from here
y_true = data.y[data.test_mask].cpu().numpy()
y_pred = preds[data.test_mask].cpu().numpy()

cm = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Pred 0", "Pred 1"]
)

plt.figure()
sns.heatmap(cm_df, annot=True, fmt='d')
plt.title("GNN Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ROC Curve from here
y_prob = probs[data.test_mask][:,1].detach().cpu().numpy()
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GNN ROC Curve")
plt.legend()
plt.show()