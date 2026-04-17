#  Fraud Detection in Transaction Graphs using GNN + Graph Heuristics


This project detects fraudulent transactions in a blockchain network using a hybrid approach that combines Graph Neural Networks (GNNs) with classical graph algorithms.

Transactions are modeled as a graph:
Nodes → Transactions
Edges → Fund transfers

The system integrates:
Deep Learning (GCN) for relational pattern learning
Graph heuristics + ML (XGBoost / Random Forest) for structural insights which will taken from different graph algorithm

Build a robust fraud detection system by combining:

* Graph Theory (DFS, BFS, SCC, PageRank, Clustering Coefficient, degree)
* Machine Learning
* Deep Learning (GNNs)

## Dataset

**Elliptic Bitcoin Transaction Dataset**

* `elliptic_txs_classes.csv` → Labels

  * `1` = Illicit (Fraud)
  * `2` = Licit
  * `-1` = Unknown
* `elliptic_txs_features.csv` → 165 features + timestep
* `elliptic_txs_edgelist.csv` → Transaction graph

Total Transactions: 203,769

## Exploratory Data Analysis

* Severe class imbalance (few fraud cases)
* Time-based transaction trends
* Graph visualization using NetworkX


## Graph Feature Engineering

We engineered **13 graph-based features** using classical algorithms:

### 🔹 Degree Features

* In-degree → fund aggregation
* Out-degree → fund dispersion

### 🔹 SCC (Kosaraju)

* Detects cycles → potential fraud patterns

### 🔹 Topological Depth (Kahn’s Algorithm)

* Measures position in transaction chain

### 🔹 PageRank

* Identifies influential transactions

### 🔹 Clustering Coefficient

* Detects dense fraud communities

### 🔹 Motif Features

* Chain score
* In-star / Out-star
* Triangle patterns

### 🔹 BFS / DFS Scores

* BFS → multi-path connectivity
* DFS → cycle-heavy behavior

---

##  Models Used

### 1️⃣ Random Forest

* Accuracy: **~83%**
* AUC: **0.85**
* Strong interpretability

### 2️⃣ XGBoost

* Accuracy: **~82%**
* AUC: **0.90**
* Better fraud recall

### 3️⃣ Graph Neural Network (GCN)

**Architecture:**
GCNConv → ReLU → GCNConv → ReLU → Linear

* Accuracy: **96%+**
* AUC: **0.95**
* Captures graph structure effectively

---

## Hybrid Model (Main Contribution)

The final prediction combines GCN and XGBoost:

P_final = α · P_GCN + (1 − α) · P_XGB

* Best performance at: **α = 0.5**

---

##  Final Results

| Model         | Accuracy  | Precision | Recall    | F1 Score  | AUC       |
| ------------- | --------- | --------- | --------- | --------- | --------- |
| Random Forest | 0.83      | 0.34      | 0.68      | 0.45      | 0.85      |
| XGBoost       | 0.82      | 0.34      | 0.82      | 0.48      | 0.90      |
| GCN           | 0.96      | 0.80      | 0.78      | 0.79      | 0.95      |
| **Ensemble**  | **0.966** | **0.809** | **0.856** | **0.832** | **0.978** |

Ensemble significantly improves performance

---

## Evaluation Metrics

* Accuracy
* Precision / Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

##  Project Structure

```
├── data/                  # (not included)
├── output_data/
├── models/
├── eda.py
├── heuristics.py
├── random_forest.py
├── xgboost_model.py
├── gnn_model.py
├── ensemble.py
├── evaluation.py
└── README.md
```

---

##  How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run pipeline

```bash
python eda.py
python heuristics.py
python random_forest.py
python xgboost_model.py
python gnn_model.py
python ensemble.py
python evaluation.py
```

---
##  Key Insights

* Graph structure is crucial for fraud detection
* Heuristic features improve interpretability
* GNN captures deep relational patterns
* Ensemble model achieves best performance


