"""
01_generate_data.py
--------------------
Generates a synthetic loan-applicant dataset with realistic relationships
between borrower features and default risk. Used as a stand-in for a
real-world source such as the LendingClub public loan dataset -- the same
pipeline in this project works unchanged if you swap in real data (see
README.md for column-mapping notes).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 8000

# --- Base borrower features -------------------------------------------------
annual_income = np.random.lognormal(mean=10.8, sigma=0.45, size=N).round(-2)
annual_income = np.clip(annual_income, 15000, 400000)

credit_score = np.random.normal(680, 65, N).round().astype(int)
credit_score = np.clip(credit_score, 300, 850)

employment_length = np.random.choice(
    [0, 1, 2, 3, 4, 5, 7, 10, 15, 20],
    size=N,
    p=[0.08, 0.10, 0.10, 0.10, 0.09, 0.12, 0.13, 0.13, 0.09, 0.06],
)

loan_amount = np.random.lognormal(mean=9.3, sigma=0.5, size=N).round(-2)
loan_amount = np.clip(loan_amount, 1000, 40000)

loan_term = np.random.choice([36, 60], size=N, p=[0.7, 0.3])

home_ownership = np.random.choice(
    ["RENT", "MORTGAGE", "OWN"], size=N, p=[0.42, 0.44, 0.14]
)

purpose = np.random.choice(
    ["debt_consolidation", "credit_card", "home_improvement",
     "small_business", "major_purchase", "medical", "other"],
    size=N,
    p=[0.35, 0.22, 0.11, 0.08, 0.08, 0.06, 0.10],
)

previous_defaults = np.random.choice(
    [0, 1, 2, 3], size=N, p=[0.78, 0.15, 0.05, 0.02]
)

open_credit_lines = np.clip(
    np.random.poisson(9, N), 1, 30
)

# Debt-to-income ratio: loosely tied to income and loan amount, plus noise
debt_to_income = np.clip(
    (loan_amount / annual_income) * 100
    + np.random.normal(8, 6, N)
    + previous_defaults * 3,
    0, 65,
).round(1)

# Interest rate assigned similarly to how lenders price risk (credit score,
# DTI, term, previous defaults) -- becomes a useful engineered feature later
interest_rate = np.clip(
    22
    - (credit_score - 300) / 550 * 15
    + debt_to_income * 0.08
    + (loan_term == 60) * 1.5
    + previous_defaults * 1.2
    + np.random.normal(0, 1.5, N),
    5.0, 30.0,
).round(2)

# --- Latent default probability (ground truth generating process) ---------
z = (
    -6.5
    + (700 - credit_score) * 0.018
    + debt_to_income * 0.045
    + previous_defaults * 1.1
    + (loan_amount / annual_income) * 3.5
    + (interest_rate - 12) * 0.12
    + (employment_length < 2) * 0.5
    + (home_ownership == "RENT") * 0.25
    + (purpose == "small_business") * 0.4
    + np.random.normal(0, 0.6, N)
)
default_prob = 1 / (1 + np.exp(-z))
default = np.random.binomial(1, default_prob)

df = pd.DataFrame({
    "annual_income": annual_income,
    "credit_score": credit_score,
    "employment_length": employment_length,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "interest_rate": interest_rate,
    "home_ownership": home_ownership,
    "purpose": purpose,
    "previous_defaults": previous_defaults,
    "open_credit_lines": open_credit_lines,
    "debt_to_income": debt_to_income,
    "default": default,
})

# Inject a small amount of realistic missingness
for col in ["employment_length", "debt_to_income"]:
    mask = np.random.rand(N) < 0.02
    df.loc[mask, col] = np.nan

out_path = "/home/claude/loan_default_project/data/loan_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(f"Default rate: {df['default'].mean():.3%}")
print(df.head())
