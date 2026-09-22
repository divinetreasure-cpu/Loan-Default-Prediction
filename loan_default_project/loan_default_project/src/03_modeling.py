"""
03_modeling.py
---------------
Feature engineering, preprocessing pipeline, model training and evaluation
for predicting loan default. Compares Logistic Regression (interpretable
baseline) against Random Forest (stronger, non-linear model).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/loan_default_project/data/loan_data.csv"
PLOTS = "/home/claude/loan_default_project/plots"

df = pd.read_csv(DATA)

# --- Feature engineering --------------------------------------------------
df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
df["income_per_credit_line"] = df["annual_income"] / (df["open_credit_lines"] + 1)

target = "default"
num_features = [
    "annual_income", "credit_score", "employment_length", "loan_amount",
    "loan_term", "interest_rate", "previous_defaults", "open_credit_lines",
    "debt_to_income", "loan_to_income", "income_per_credit_line",
]
cat_features = ["home_ownership", "purpose"]

X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_features),
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

results = {}
roc_data = {}
plt.figure(figsize=(6, 5))

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    results[name] = {
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision_default": round(report["1"]["precision"], 4),
        "recall_default": round(report["1"]["recall"], 4),
        "f1_default": round(report["1"]["f1-score"], 4),
        "confusion_matrix": cm,
    }
    roc_data[name] = (fpr, tpr, auc)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    if name == "Random Forest":
        best_pipe = pipe  # keep the stronger model for importance plot

plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_roc_curves.png")
plt.close()

# --- Confusion matrix for the best model (Random Forest) ------------------
rf_cm = np.array(results["Random Forest"]["confusion_matrix"])
plt.figure(figsize=(4.5, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
plt.title("Random Forest — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_confusion_matrix.png")
plt.close()

# --- Feature importance (Random Forest) -----------------------------------
ohe = best_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(cat_features))
all_feature_names = num_features + cat_names

importances = best_pipe.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({
    "feature": all_feature_names, "importance": importances
}).sort_values("importance", ascending=False).head(12)

plt.figure(figsize=(7, 6))
sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
plt.title("Top Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_feature_importance.png")
plt.close()

# --- Save metrics summary ---------------------------------------------
with open("/home/claude/loan_default_project/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved ROC curve, confusion matrix, and feature importance plots.")
