# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
import pandas as pd

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processing.data_loading import df

# ── Check for Missing Values ──────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df) == 0:
    print("No missing values found! Our data is clean.")
else:
    print("Missing values found:")
    print(missing_df)

# ── Check for Duplicate Rows ──────────────────────────────────────────────────
duplicates = df.duplicated().sum()
print(f"\n Duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   → Removed {duplicates} duplicates. New shape: {df.shape}")
else:
    print("   → No duplicates.")

