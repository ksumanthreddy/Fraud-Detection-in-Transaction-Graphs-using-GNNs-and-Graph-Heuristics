import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

features = pd.read_csv('output_data/graph_heuristics_features.csv')

features = features[features["label"] != -1]          # remove unlabeled

x = features.iloc[:, :13]
y = features["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42                          # split X and y
)

imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()       # imbalance ratio

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=imbalance_ratio,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

# train
model.fit(x_train, y_train)

model.save_model("xgboost_model.json")
# predict
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Pred 0", "Pred 1"]
)

print(cm_df)

# feature importance
feature_importance = pd.DataFrame({
    "feature": x.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

y_prob = model.predict_proba(x_test)[:, 1]

# Confusion matrix plot
plt.figure()
sns.heatmap(cm_df, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature importance plot
importances = model.feature_importances_
feature_names = x.columns
feat_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure()
plt.barh(feat_df["feature"], feat_df["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()
