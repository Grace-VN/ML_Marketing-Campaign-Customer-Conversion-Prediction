# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import feature engineering to ensure CostPerVisit exists
from input_processing.feature_engineering import df  # ← Ensures engineered features are created
from input_processing.train_test_split import X_temp   # ← fixed module path

# ==============================
# Output Paths
# ==============================
CSV_DIR   = ROOT_DIR / 'output_storage' / 'csv_files'
IMAGE_DIR = ROOT_DIR / 'output_storage' / 'images'
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Feature Groups
# ==============================
num_cols = [
    'Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
    'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
    'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints',
    # Engineered features (excluding AgeBand)
    'CostPerVisit', 'CostPerClick', 'EmailEngagementRate',
    'CustomerValue', 'IncomeToAdSpend', 'SiteEngagementScore',
    'SocialAmplification', 'CTR_x_PagesPerVisit'
]
cat_cols = ['Gender', 'CampaignChannel', 'CampaignType']

# ==============================
# Pipelines
# ==============================
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)

# ==============================
# Fit & Transform
# ==============================
X_transformed = preprocessor.fit_transform(X_temp)

# ==============================
# Feature Name Resolution
# ==============================
encoded_features = (
    preprocessor.named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(cat_cols)
)
passthrough_cols = [col for col in X_temp.columns if col not in num_cols + cat_cols]
all_features     = num_cols + list(encoded_features) + passthrough_cols

# ==============================
# Console Summary
# ==============================
print("=" * 50)
print("Preprocessing Completed")
print("=" * 50)
print(f"\nOriginal Feature Shape    : {X_temp.shape}")
print(f"Transformed Feature Shape : {X_transformed.shape}")
print(f"\nTotal Engineered Features : {len(all_features)}")
print("\nSample Features:")
print(all_features[:10])

# ==============================
# Save CSVs
# ==============================

# --- Transformed Data → CSV ---
transformed_df   = pd.DataFrame(X_transformed, columns=all_features)
transformed_path = CSV_DIR / "X_transformed.csv"
transformed_df.to_csv(transformed_path, index=False)
print(f"\n[Saved] Transformed data        → {transformed_path}")

# --- Feature Inventory → CSV ---
feature_inventory_df = pd.DataFrame({
    "feature": all_features,
    "type": (
        ["numerical"] * len(num_cols) +
        ["categorical_encoded"] * len(encoded_features) +
        ["passthrough"] * len(passthrough_cols)
    )
})
feature_inventory_path = CSV_DIR / "feature_inventory.csv"
feature_inventory_df.to_csv(feature_inventory_path, index=False)
print(f"[Saved] Feature inventory       → {feature_inventory_path}")

# --- Preprocessing Summary → CSV ---
summary_df = pd.DataFrame([{
    "original_shape":    str(X_temp.shape),
    "transformed_shape": str(X_transformed.shape),
    "n_numerical":       len(num_cols),
    "n_categorical_raw": len(cat_cols),
    "n_encoded":         len(encoded_features),
    "n_passthrough":     len(passthrough_cols),
    "n_total_features":  len(all_features)
}])
summary_path = CSV_DIR / "preprocessing_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"[Saved] Preprocessing summary   → {summary_path}")

# ==============================
# Plot 1: Feature Type Breakdown (Pie) → Image
# ==============================
type_counts = feature_inventory_df["type"].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct='%1.1f%%',
    colors=['steelblue', 'darkorange', 'seagreen'],
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
ax.set_title('Engineered Feature Type Breakdown', fontsize=13)
plt.tight_layout()

pie_img_path = IMAGE_DIR / "preprocessing_feature_breakdown.png"
fig.savefig(pie_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Feature breakdown pie   → {pie_img_path}")
plt.show()
plt.close()

# ==============================
# Plot 2: Scaled Numerical Feature Distributions → Image
# ==============================
plot_cols = [
    'Age', 'Income', 'AdSpend', 'ClickThroughRate',
    'WebsiteVisits', 'TimeOnSite', 'ConversionRate',
    'PreviousPurchases', 'LoyaltyPoints', 'SocialShares',
    'EmailOpens', 'EmailClicks'
]
num_df = transformed_df[plot_cols]

fig2, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(plot_cols):
    axes[i].hist(num_df[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('Scaled Value')
    axes[i].set_ylabel('Count')
    axes[i].grid(alpha=0.3)

plt.suptitle('Scaled Numerical Feature Distributions (Post-StandardScaler)', fontsize=13, y=1.01)
plt.tight_layout()

dist_img_path = IMAGE_DIR / "preprocessing_numerical_distributions.png"
fig2.savefig(dist_img_path, dpi=150, bbox_inches='tight')
print(f"[Saved] Numerical distributions → {dist_img_path}")
plt.show()
plt.close()