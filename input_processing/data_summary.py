# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processing.data_loading import df

# ── Basic Info ────────────────────────────────────────────────────────────────
print("📋 Column Names and Data Types:")
print("=" * 45)
df.info()
# ── Statistical Summary ───────────────────────────────────────────────────────
print("📈 Statistical Summary (for number columns):")
print(df.describe().round(2))
# ── Unique Values in Categorical Columns ──────────────────────────────────────
cat_cols = ['Gender', 'CampaignChannel', 'CampaignType', 'AdvertisingPlatform', 'AdvertisingTool']

print("🏷️ Unique Values in Categorical Columns:")
print("=" * 45)
for col in cat_cols:
    unique_vals = df[col].unique().tolist()
    print(f"  {col}: {unique_vals}")
    