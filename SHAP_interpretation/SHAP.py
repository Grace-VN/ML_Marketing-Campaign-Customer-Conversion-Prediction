# -*- coding: utf-8 -*-
import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

# LinearExplainer is correct for Logistic Regression
explainer   = shap.LinearExplainer(model, X_test_transformed)
shap_values = explainer.shap_values(X_test_transformed)

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

# ==============================
# Plot 1: SHAP Summary (Beeswarm)
# ==============================
shap.summary_plot(shap_values, X_test_transformed, show=False)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "shap_summary_plot.png", dpi=150, bbox_inches='tight')
print(f"[Saved] SHAP summary plot       → shap_summary_plot.png")
plt.show(); plt.close()

# ==============================
# Plot 2: SHAP Bar Plot (Mean |SHAP|)
# ==============================
shap.summary_plot(shap_values, X_test_transformed, plot_type='bar', show=False)
ax = plt.gca()
for bar in ax.patches:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "shap_bar_plot.png", dpi=150, bbox_inches='tight')
print(f"[Saved] SHAP bar plot           → shap_bar_plot.png")
plt.show(); plt.close()

# ==============================
# Plot 3: SHAP Dependence — Top Feature
# ==============================
top_feature = mean_shap_df.iloc[0]["feature"]   # auto-picks most important
fig, ax = plt.subplots()
shap.dependence_plot(top_feature, shap_values, X_test_transformed, ax=ax, show=False)
ax.set_title(f'SHAP Dependence — {top_feature}')
plt.tight_layout()
fig.savefig(IMAGE_DIR / f"shap_dependence_{top_feature}.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Dependence plot         → shap_dependence_{top_feature}.png")
plt.show(); plt.close()

# ==============================
# Plot 4: Waterfall — Single Customer Explanation
# ==============================
# Show one converted (y=1) and one non-converted (y=0) customer
for label, idx in [("converted", y_test.values.tolist().index(1)),
                   ("not_converted", y_test.values.tolist().index(0))]:
    shap.waterfall_plot(
        shap.Explanation(
            values         = shap_values[idx],
            base_values    = explainer.expected_value,
            data           = X_test_transformed.iloc[idx].values,
            feature_names  = feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches='tight')
    print(f"[Saved] Waterfall ({label})  → shap_waterfall_{label}.png")
    plt.show(); plt.close()

# ==============================
# Plot 5: Calibration Curve
# ==============================
from sklearn.calibration import calibration_curve

proba                = final_pipeline.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, marker='o', color='steelblue', label='Logistic Regression')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
ax.set_title('Calibration Curve — Logistic Regression', fontsize=13)
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "calibration_curve.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Calibration curve       → calibration_curve.png")
plt.show(); plt.close()

# ==============================
# Plot 6: Profit Curve (Business ROI)
# ==============================
REVENUE_PER_CONVERSION = 50    # $ revenue if customer converts
COST_PER_CONTACT       = 5     # $ cost to target one customer

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

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, profits, color='steelblue', lw=2)
ax.axvline(best_thresh, color='tomato', linestyle='--',
           label=f'Optimal threshold: {best_thresh:.2f}')
ax.axhline(best_profit, color='seagreen', linestyle='--',
           label=f'Max profit: ${best_profit:,.0f}')
ax.set_title('Profit Curve — Revenue vs Contact Cost by Threshold', fontsize=13)
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Estimated Profit ($)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "profit_curve.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Profit curve            → profit_curve.png")
plt.show(); plt.close()