# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processing.data_loading import df

# ── Basic Info ────────────────────────────────────────────────────────────────
print("📋 Column Names and Data Types:")
print("=" * 45)
df.info()

# ── Statistical Summary ───────────────────────────────────────────────────────
print("\n📈 Statistical Summary (numeric columns):")
print(df.describe().round(2))

# ── Unique Values in Categorical Columns ──────────────────────────────────────
cat_cols = ['Gender', 'CampaignChannel', 'CampaignType', 'AdvertisingPlatform', 'AdvertisingTool']

print("\n🏷️ Unique Values in Categorical Columns:")
print("=" * 45)
for col in cat_cols:
    unique_vals = df[col].unique().tolist()
    print(f"  {col}: {unique_vals}")