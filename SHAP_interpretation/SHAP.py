# -*- coding: utf-8 -*-
import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.evaluation import final_pipeline, X_test, y_test

# ==============================
# Output Paths
# ==============================
CSV_DIR   = ROOT_DIR / 'output_storage' / 'csv_files'
IMAGE_DIR = ROOT_DIR / 'output_storage' / 'images'
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Preprocessing & SHAP Setup
# ==============================
X_test_transformed = final_pipeline.named_steps['preprocessor'].transform(X_test)
feature_names      = final_pipeline.named_steps['preprocessor'].get_feature_names_out()
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
model              = final_pipeline.named_steps['classifier']

# KernelExplainer — works with any sklearn model including AdaBoost
background  = shap.sample(X_test_transformed, 100, random_state=42)
explainer   = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer.shap_values(X_test_transformed, nsamples=100)

# Robust shape handling — KernelExplainer returns (n_samples, n_features, n_classes)
shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]      # slice last axis for class 1 (conversion)
elif isinstance(shap_values, list):
    shap_values = np.array(shap_values[1])

assert shap_values.ndim == 2, f"Unexpected shape: {shap_values.shape}"
print(f"SHAP values shape: {shap_values.shape}")

# expected_value — take class 1
base_val = explainer.expected_value[1] if isinstance(
    explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

# ==============================
# Save SHAP Values → CSV
# ==============================
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv(CSV_DIR / "shap_values.csv", index=False)
print(f"[Saved] SHAP values             → shap_values.csv")

mean_shap_df = (
    pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": shap_df.abs().mean().values
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
mean_shap_df.to_csv(CSV_DIR / "shap_feature_importance.csv", index=False)
print(f"[Saved] SHAP feature importance → shap_feature_importance.csv")

print("\nTop 10 features driving Customer Conversion:")
print(mean_shap_df.head(10).to_string(index=False))

# ==============================
# Plot 1: SHAP Summary (Beeswarm)
# ==============================
shap.summary_plot(shap_values, X_test_transformed, show=False)
plt.title('SHAP Feature Impact on Customer Conversion\n'
          '(red = pushes toward conversion, blue = pushes away)', fontsize=12)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "shap_summary_plot.png", dpi=150, bbox_inches='tight')
print(f"[Saved] SHAP summary plot       → shap_summary_plot.png")
plt.show(); plt.close()

# ==============================
# Plot 2: SHAP Bar Plot (Mean |SHAP|)
# ==============================
shap.summary_plot(shap_values, X_test_transformed, plot_type='bar', show=False)
ax = plt.gca()
ax.set_title('Mean |SHAP| — Feature Importance for Conversion Prediction', fontsize=12)
for bar in ax.patches:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "shap_bar_plot.png", dpi=150, bbox_inches='tight')
print(f"[Saved] SHAP bar plot           → shap_bar_plot.png")
plt.show(); plt.close()

# ==============================
# Plot 3: SHAP Dependence — Top 2 Features
# ==============================
for i in range(min(2, len(mean_shap_df))):
    top_feature = mean_shap_df.iloc[i]["feature"]
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(top_feature, shap_values, X_test_transformed, ax=ax, show=False)
    ax.set_title(f'SHAP Dependence — {top_feature}\n'
                 f'(How this feature drives conversion probability)', fontsize=11)
    plt.tight_layout()
    fig.savefig(IMAGE_DIR / f"shap_dependence_{top_feature}.png", dpi=150, bbox_inches='tight')
    print(f"[Saved] Dependence plot         → shap_dependence_{top_feature}.png")
    plt.show(); plt.close()

# ==============================
# Plot 4: Waterfall — One Converted, One Not Converted
# ==============================
for label, idx in [("converted",    y_test.values.tolist().index(1)),
                   ("not_converted", y_test.values.tolist().index(0))]:
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = base_val,
            data          = X_test_transformed.iloc[idx].values,
            feature_names = feature_names
        ),
        show=False
    )
    plt.title(
        f'Why this customer {"converted" if label == "converted" else "did not convert"}',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches='tight')
    print(f"[Saved] Waterfall ({label})     → shap_waterfall_{label}.png")
    plt.show(); plt.close()

# ==============================
# Plot 5: Calibration Curve
# ==============================
proba                = final_pipeline.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, marker='o', color='steelblue', label='AdaBoost')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
ax.set_title('Calibration Curve — AdaBoost\n'
             '(Are predicted conversion probabilities reliable?)', fontsize=12)
ax.set_xlabel('Mean Predicted Conversion Probability')
ax.set_ylabel('Actual Conversion Rate')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "calibration_curve.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Calibration curve       → calibration_curve.png")
plt.show(); plt.close()

# ==============================
# Plot 6: Profit Curve (Business ROI)
# ==============================
REVENUE_PER_CONVERSION = 50
COST_PER_CONTACT       = 5

thresholds = np.linspace(0, 1, 100)
profits    = []

for thresh in thresholds:
    predicted_positive = proba >= thresh
    true_positive      = predicted_positive & (y_test.values == 1)
    profit = (true_positive.sum() * REVENUE_PER_CONVERSION) - \
             (predicted_positive.sum() * COST_PER_CONTACT)
    profits.append(profit)

best_thresh = thresholds[np.argmax(profits)]
best_profit = max(profits)
print(f"\nOptimal decision threshold : {best_thresh:.2f}")
print(f"Estimated max profit       : ${best_profit:,.0f}")
print(f"At threshold {best_thresh:.2f}: target only customers with "
      f">{int(best_thresh*100)}% predicted conversion probability")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, profits, color='steelblue', lw=2, label='Estimated Profit')
ax.axvline(best_thresh, color='tomato', linestyle='--',
           label=f'Optimal threshold: {best_thresh:.2f}')
ax.axhline(best_profit, color='seagreen', linestyle='--',
           label=f'Max profit: ${best_profit:,.0f}')
ax.fill_between(thresholds, profits, alpha=0.08, color='steelblue')
ax.set_title('Profit Curve — Optimal Conversion Targeting Threshold\n'
             f'(Revenue=${REVENUE_PER_CONVERSION}/conversion, '
             f'Cost=${COST_PER_CONTACT}/contact)', fontsize=12)
ax.set_xlabel('Decision Threshold (min predicted conversion probability to target)')
ax.set_ylabel('Estimated Profit ($)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "profit_curve.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Profit curve            → profit_curve.png")
plt.show(); plt.close()

# ==============================
# Save Business Insight Summary → CSV
# ==============================
pd.DataFrame([{
    "top_conversion_driver":    mean_shap_df.iloc[0]["feature"],
    "second_conversion_driver": mean_shap_df.iloc[1]["feature"],
    "third_conversion_driver":  mean_shap_df.iloc[2]["feature"],
    "optimal_threshold":        round(best_thresh, 2),
    "estimated_max_profit":     round(best_profit, 2),
    "revenue_per_conversion":   REVENUE_PER_CONVERSION,
    "cost_per_contact":         COST_PER_CONTACT,
}]).to_csv(CSV_DIR / "business_insight_summary.csv", index=False)
print(f"[Saved] Business insight summary → business_insight_summary.csv")