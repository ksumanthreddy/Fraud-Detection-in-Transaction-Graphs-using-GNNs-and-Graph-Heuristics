import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


features=pd.read_csv("data/elliptic_txs_features.csv",header=None)
heuristics=pd.read_csv("output_data/graph_heuristics_features.csv")
edges=pd.read_csv("data/elliptic_txs_edgelist.csv")

# Set column names
features.columns=['txId','time_step']+[f'f{i}' for i in range(features.shape[1]-2)]

# Merge features + labels
df=features.merge(heuristics,on="txId")
df=df[df["label"]!=-1]

tx_ids=df["txId"].values

# Convert to tensors
X=torch.tensor(df.drop(columns=["txId","label"]).values,dtype=torch.float)
y=torch.tensor(df["label"].values,dtype=torch.long)

# Map txId → node index
node_map={tx:i for i,tx in enumerate(tx_ids)}

# Build edge index
edge_list=[]
for _,r in edges.iterrows():
    u,v=r["txId1"],r["txId2"]
    if u in node_map and v in node_map:
        edge_list.append([node_map[u],node_map[v]])

edge_index=torch.tensor(edge_list,dtype=torch.long).t().contiguous()

# Graph data
data=Data(x=X,edge_index=edge_index,y=y)


# GCN model (same architecture as training)
class GCN(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1=GCNConv(in_channels,128)
        self.conv2=GCNConv(128,64)
        self.out=torch.nn.Linear(64,2)
    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x=F.relu(self.conv1(x,edge_index))
        x=F.relu(self.conv2(x,edge_index))
        return self.out(x)

# Load trained GNN
model=GCN(data.num_features)
model.load_state_dict(torch.load("gnn_model.pth"))
model.eval()

# GNN predictions
out=model(data)
gcn_probs=torch.softmax(out,dim=1)[:,1].detach().numpy()
gcn_preds=(gcn_probs>0.5).astype(int)

gcn_df=pd.DataFrame({
    "txId":tx_ids,
    "gcn_prob":gcn_probs,
    "gcn_pred":gcn_preds,
    "label":df["label"].values
})

# Load trained XGBoost
xgb=XGBClassifier()
xgb.load_model("xgboost_model.json")

# Use heuristic features only (same as training)
X_xgb=heuristics[heuristics["label"]!=-1].iloc[:, :13]

# XGBoost predictions
xgb_probs=xgb.predict_proba(X_xgb)[:,1]
xgb_preds=(xgb_probs>0.5).astype(int)

xgb_df=pd.DataFrame({
    "txId":heuristics[heuristics["label"]!=-1]["txId"].values,
    "xgb_prob":xgb_probs,
    "xgb_pred":xgb_preds
})

# Merge both model outputs
final_df=gcn_df.merge(xgb_df,on="txId")

# Save combined predictions
final_df.to_csv("final_predictions_probability.csv",index=False)


# Ensemble (weighted average)
y_true = final_df["label"].values
gcn_prob = final_df["gcn_prob"].values
xgb_prob = final_df["xgb_prob"].values

alphas = np.arange(0.0, 1.1, 0.1)

results = []

for a in alphas:
    # a = weight for GCN, (1-a) for XGB
    final_prob = a * gcn_prob + (1 - a) * xgb_prob
    y_pred = (final_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append([a, 1-a, acc, prec, rec, f1])

# Store results
alpha_df = pd.DataFrame(results, columns=[
    "alpha_gcn", "beta_xgb", "accuracy", "precision", "recall", "f1_score"
])

print(alpha_df)
