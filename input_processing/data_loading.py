import sys
import io
import pandas as pd

# Configure UTF-8 output for Windows compatibility
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Load the dataset ──────────────────────────────────────────────────────────

df = pd.read_csv('D:\Job\Portfolio\Machine Learning\ML_Marketing Campaign-Customer Conversion\data\digital_marketing_campaign_dataset.csv')

print(f"✅ Dataset loaded!")
print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print()
print("👀 First 5 rows:")
print(df.head())