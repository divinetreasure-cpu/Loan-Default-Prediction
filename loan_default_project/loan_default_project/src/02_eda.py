"""
02_eda.py
---------
Exploratory data analysis: missingness, distributions, class balance,
and feature-vs-default relationships. Saves all plots to ../plots/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/loan_default_project/data/loan_data.csv"
PLOTS = "/home/claude/loan_default_project/plots"

df = pd.read_csv(DATA)

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print("\nMISSING VALUES:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nCLASS BALANCE:\n", df["default"].value_counts(normalize=True))
print("\nSUMMARY STATS:\n", df.describe().T)

# --- 1. Class balance --------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x="default", data=df, palette=["#4C72B0", "#C44E52"])
plt.title("Class Balance: Default vs. No Default")
plt.xlabel("Default (1 = yes)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_class_balance.png")
plt.close()

# --- 2. Numeric feature distributions by default class -----------------
num_cols = ["annual_income", "credit_score", "loan_amount",
            "interest_rate", "debt_to_income", "previous_defaults"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, num_cols):
    sns.kdeplot(data=df, x=col, hue="default", fill=True, alpha=0.35,
                common_norm=False, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_feature_distributions.png")
plt.close()

# --- 3. Correlation heatmap ---------------------------------------------
plt.figure(figsize=(8, 6))
corr = df[num_cols + ["open_credit_lines", "employment_length", "default"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_correlation_heatmap.png")
plt.close()

# --- 4. Default rate by categorical features -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df.groupby("home_ownership")["default"].mean().sort_values().plot(
    kind="barh", ax=axes[0], color="#55A868")
axes[0].set_title("Default Rate by Home Ownership")
axes[0].set_xlabel("Default Rate")

df.groupby("purpose")["default"].mean().sort_values().plot(
    kind="barh", ax=axes[1], color="#DD8452")
axes[1].set_title("Default Rate by Loan Purpose")
axes[1].set_xlabel("Default Rate")
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_default_rate_by_category.png")
plt.close()

print("\nSaved 4 EDA plots to", PLOTS)
