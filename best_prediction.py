import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

final_df = pd.read_csv("output_data/final_predictions_probability.csv")  

# Weight for ensemble (GCN vs XGB)
alpha = 0.5

# Weighted probability + final prediction
final_df["final_prob"] = alpha * final_df["gcn_prob"] + (1 - alpha) * final_df["xgb_prob"]
final_df["final_pred"] = (final_df["final_prob"] > 0.5).astype(int)

# True vs predicted
y_true = final_df["label"].values
y_pred = final_df["final_pred"].values
y_prob = final_df["final_prob"].values

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Accuracy:",acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)


# Confusion matrix
cm=confusion_matrix(y_true,y_pred)

plt.figure()
sns.heatmap(cm,annot=True,fmt='d')
plt.title(f"Confusion Matrix(alpha={alpha})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ROC Curve
fpr, tpr,_=roc_curve(y_true,y_prob)
roc_auc=auc(fpr, tpr)

plt.figure()
plt.plot(fpr,tpr,label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1],[0, 1],linestyle='--')  # baseline
plt.title(f"ROC Curve(alpha={alpha})")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


# Precision Recall Curve (better for imbalance)
precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.title(f"Precision-Recall Curve (alpha={alpha})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()