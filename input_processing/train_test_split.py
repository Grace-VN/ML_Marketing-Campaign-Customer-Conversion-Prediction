# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))          # ← fixes ModuleNotFoundError

CSV_DIR = ROOT_DIR / 'output_storage' / 'csv_files'
os.makedirs(CSV_DIR, exist_ok=True)

from input_processing.feature_engineering import df

# ==============================
# Define X and y
# ==============================
drop_cols = [
    'CustomerID',        # identifier, no signal
    'AdvertisingPlatform',   # ← 'IsConfid' — this must be dropped
    'AdvertisingTool',       # ← 'ToolConfid' — this must be dropped
    'AgeBand',        # ← add this; Age numeric is already in num_cols
    'Conversion'            # target — must not leak into X
]

X = df.drop(columns=drop_cols)
y = df['Conversion']

print(f"Features (X): {X.shape}")
print(f"Target   (y): {y.shape}")
print(f"Conversion rate in full dataset: {y.mean():.3f}")
print(f"\nFeature columns:\n{X.columns.tolist()}")

# ==============================
# Train / Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nOptimization and Cross-Validation data (X_train): {X_train.shape}")
print(f"UNTOUCHED Final Hold-Out Test data    (X_test):  {X_test.shape}")

# Class balance check
print(f"\nConversion rate in y_train : {y_train.mean():.3f}")
print(f"Conversion rate in y_test : {y_test.mean():.3f}")

# ==============================
# Save Splits → CSV
# ==============================
X_train.to_csv(CSV_DIR / "X_train.csv", index=False)
X_test.to_csv(CSV_DIR / "X_test.csv", index=False)
y_train.to_csv(CSV_DIR / "y_train.csv", index=False)
y_test.to_csv(CSV_DIR / "y_test.csv", index=False)

print("\n[Saved] X_train → X_train.csv")
print("[Saved] X_test → X_test.csv")
print("[Saved] y_train → y_train.csv")
print("[Saved] y_test → y_test.csv")