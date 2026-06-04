# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_curve,
    roc_auc_score, RocCurveDisplay,
    confusion_matrix, ConfusionMatrixDisplay
)

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from input_processing.train_test_split import X_test, y_test, X_test, y_test
from input_processing.normalization_encoding import preprocessor
from model.fine_tuning import best_params

# ==============================
# Output Paths
# ==============================
CSV_DIR   = ROOT_DIR / 'output_storage' / 'csv_files'
IMAGE_DIR = ROOT_DIR / 'output_storage' / 'images'
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# Final Model Selection
# ==============================
try:
    from model.fine_tuning import best_params
    print(">>> Using Optimized Hyperparameters from Optuna Tuning")
    final_model = LogisticRegression(
        **best_params,
        solver='saga',
        class_weight='balanced',
        random_state=42
    )
except (ImportError, AttributeError, NameError):
    print(">>> Optuna tuning skipped. Using Baseline Logistic Regression parameters")
    final_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

# ==============================
# Final Fit on Full test Set
# ==============================
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   final_model)
])

final_pipeline.fit(X_test, y_test)

# ==============================
# Predictions on Held-Out Test Set
# ==============================
probs = final_pipeline.predict_proba(X_test)[:, 1]
preds = final_pipeline.predict(X_test)

# ==============================
# Evaluation
# ==============================
final_auc = roc_auc_score(y_test, probs)
print(f"\nFinal Test AUC-ROC: {final_auc:.5f}")

# --- Classification Report → CSV ---
report_dict = classification_report(y_test, preds, output_dict=True)
report_df   = pd.DataFrame(report_dict).transpose().round(4)
report_df.to_csv(CSV_DIR / "classification_report.csv")
print(f"\nClassification Report:\n{report_df}")
print(f"[Saved] Classification report  → classification_report.csv")

# --- Prediction Probabilities & Labels → CSV ---
preds_df = pd.DataFrame({
    "y_true":           y_test.values if hasattr(y_test, "values") else y_test,
    "y_pred":           preds,
    "prob_conversion":  probs,          # renamed from prob_churn
})
preds_df.to_csv(CSV_DIR / "test_predictions.csv", index=False)
print(f"[Saved] Test predictions        → test_predictions.csv")

# --- Summary Metrics → CSV ---
metrics_df = pd.DataFrame([{"metric": "AUC-ROC", "value": round(final_auc, 5)}])
metrics_df.to_csv(CSV_DIR / "summary_metrics.csv", index=False)
print(f"[Saved] Summary metrics         → summary_metrics.csv")

# --- ROC Curve Data → CSV ---
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
roc_df.to_csv(CSV_DIR / "roc_curve_data.csv", index=False)
print(f"[Saved] ROC curve data          → roc_curve_data.csv")

# ==============================
# Plot 1: ROC Curve
# ==============================
roc_display = RocCurveDisplay(
    fpr=fpr, tpr=tpr,
    roc_auc=final_auc,
    estimator_name='Optimized Logistic Regression'
)

fig, ax = plt.subplots(figsize=(8, 6))
roc_display.plot(ax=ax, color='steelblue', lw=2)
ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
ax.set_title('ROC Curve — Logistic Regression', fontsize=14)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "roc_curve.png", dpi=150, bbox_inches='tight')
print(f"[Saved] ROC curve               → roc_curve.png")
plt.show(); plt.close()

# ==============================
# Plot 2: Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['No Conversion', 'Conversion']).plot(
    ax=ax, colorbar=False, cmap='Blues'
)
ax.set_title('Confusion Matrix — Logistic Regression', fontsize=13)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Confusion matrix        → confusion_matrix.png")
plt.show(); plt.close()

# ==============================
# Plot 3: Precision-Recall by Threshold
# ==============================
from sklearn.metrics import precision_recall_curve

precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pr_thresholds, precision[:-1], color='steelblue', lw=2, label='Precision')
ax.plot(pr_thresholds, recall[:-1],    color='tomato',    lw=2, label='Recall')
ax.set_title('Precision & Recall vs Threshold — Logistic Regression', fontsize=13)
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "precision_recall_threshold.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Precision-recall curve  → precision_recall_threshold.png")
plt.show(); plt.close()