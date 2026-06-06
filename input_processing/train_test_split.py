# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from input_processing.feature_engineering import df   # includes all engineered features

CSV_DIR = ROOT_DIR / 'output_storage' / 'csv_files'
os.makedirs(CSV_DIR, exist_ok=True)

# ==============================
# Define X and y
# ==============================
drop_cols = [
    'CustomerID',
    'AdvertisingPlatform',
    'AdvertisingTool',
    'AgeBand',
    'Conversion'
]

X = df.drop(columns=drop_cols)
y = df['Conversion']

# Sanity check — no unexpected string columns
leftover_strings = X.select_dtypes(include='object').columns.tolist()
unexpected = [c for c in leftover_strings if c not in ['Gender', 'CampaignChannel', 'CampaignType']]
assert not unexpected, f"Unhandled string columns: {unexpected}"

print(f"Features (X): {X.shape}")
print(f"Target   (y): {y.shape}")
print(f"Conversion rate in full dataset: {y.mean():.3f}")
print(f"\nFeature columns:\n{X.columns.tolist()}")

# ==============================
# Train / Test Split
# ==============================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nOptimization and Cross-Validation data (X_temp): {X_temp.shape}")
print(f"UNTOUCHED Final Hold-Out Test data    (X_test):  {X_test.shape}")

print(f"\nConversion rate in y_temp : {y_temp.mean():.3f}")
print(f"Conversion rate in y_test : {y_test.mean():.3f}")

# ==============================
# Save Splits → CSV
# ==============================
X_temp.to_csv(CSV_DIR / "X_temp.csv", index=False)
X_test.to_csv(CSV_DIR / "X_test.csv", index=False)
y_temp.to_csv(CSV_DIR / "y_temp.csv", index=False)
y_test.to_csv(CSV_DIR / "y_test.csv", index=False)

print("\n[Saved] X_temp → X_temp.csv")
print("[Saved] X_test → X_test.csv")
print("[Saved] y_temp → y_temp.csv")
print("[Saved] y_test → y_test.csv")