import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path() 

# Import original dataframe
from input_processing.data_loading import df

# ── STEP 2: Label Encode all categorical (text) columns ───────────────────────
le = LabelEncoder()

# This will now safely catch any new text columns too
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nAutomatically detected categorical columns:")
print(categorical_columns)

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str)) # astype(str) keeps it safe
    print(f"  ✅ Encoded: {col}")

print("\n🔢 After encoding — first 3 rows:")
print(df[categorical_columns].head())