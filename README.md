# Loan Default Prediction

A full data science pipeline predicting whether a borrower will default on a
loan, using borrower financial and demographic data, a core credit risk
problem for fintech lenders.

#
**Note on data:** this run uses a synthetically generated dataset built to
mirror the statistical patterns of real consumer lending data (income,
credit score, DTI, previous defaults driving default probability). The
pipeline is written to work unchanged on a real dataset such as
[LendingClub's public loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) —
just point `01_generate_data.py`'s output path to your real CSV with
matching column names, or adjust `num_features`/`cat_features` in
`03_modeling.py`.

## 1. Dataset

8,000 loan applicants with 12 features: income, credit score, loan amount
and term, interest rate, employment length, home ownership, loan purpose,
previous defaults, open credit lines, and debt-to-income ratio. Target:
`default` (1 = defaulted, 17.75% of loans — a realistic, imbalanced rate).

## 2. Exploratory Data Analysis

Key findings from `plots/`:
- **Class imbalance**: ~18% default rate — informed the modeling choice to
  use `class_weight="balanced"` rather than plain accuracy.
- **Credit score & DTI** are the clearest separators between defaulters and
  non-defaulters (see `02_feature_distributions.png`).
- **Previous defaults, DTI, and interest rate** correlate most strongly with
  default in the correlation heatmap.
- **Renters** and **small-business loan purposes** show elevated default
  rates vs. mortgage-holders and debt-consolidation loans.

## 3. Feature Engineering

- `loan_to_income`: loan amount relative to income (a standard underwriting
  ratio)
- `income_per_credit_line`: income spread across existing credit obligations
- Missing values (median/mode imputation), categorical one-hot encoding,
  numeric standardization — all inside a single `sklearn` `Pipeline` /
  `ColumnTransformer` to prevent train/test leakage.

## 4. Models & Results

Two models trained on an 80/20 stratified split:

| Model | ROC-AUC | Avg. Precision | Recall (default) | F1 (default) |
|---|---|---|---|---|
| Logistic Regression | **0.880** | 0.698 | 0.792 | 0.599 |
| Random Forest | 0.865 | 0.645 | 0.729 | 0.576 |

Logistic Regression slightly outperforms the Random Forest here and is more
interpretable — a common and defensible outcome in credit risk modeling,
where regulators often favor transparent models. Both catch roughly
73–79% of actual defaulters, at the cost of some false positives (an
acceptable trade-off for a lender screening high-risk applicants).

**Top predictive features** (Random Forest importance): credit score,
debt-to-income, loan-to-income ratio, previous defaults, and interest rate —
consistent with real-world underwriting practice.

## 5. How to Extend

- Swap in the real LendingClub dataset for production-grade results
- Try gradient boosting (XGBoost/LightGBM) for a stronger non-linear model
- Add a cost-sensitive threshold analysis (false negatives — missed
  defaults — typically cost lenders more than false positives)
- Deploy via a simple Flask/FastAPI scoring endpoint

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python src/01_generate_data.py
python src/02_eda.py
python src/03_modeling.py
```
