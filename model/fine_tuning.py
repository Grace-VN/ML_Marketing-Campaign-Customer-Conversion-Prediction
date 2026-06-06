# -*- coding: utf-8 -*-
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

warnings.filterwarnings('ignore', message='X does not have valid feature names')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from input_processing.train_test_split import X_temp, y_temp
from input_processing.normalization_encoding import preprocessor
from model.models import results as baseline_results, fold_rows as baseline_fold_rows

# ==============================
# Output Paths
# ==============================
CSV_DIR   = ROOT_DIR / 'output_storage' / 'csv_files'
IMAGE_DIR = ROOT_DIR / 'output_storage' / 'images'
os.makedirs(CSV_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==============================
# StratifiedKFold
# ==============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

METRICS = {
    'auc_roc':   'roc_auc',
    'accuracy':  'accuracy',
    'f1':        'f1',
    'precision': 'precision',
    'recall':    'recall',
}

METRIC_LABELS = {
    'auc_roc':   'AUC-ROC',
    'accuracy':  'Accuracy',
    'f1':        'F1 Score',
    'precision': 'Precision',
    'recall':    'Recall',
}

# ==============================
# Hyperparameter Grid for AdaBoost
# ==============================
param_grid = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__learning_rate': [0.1, 0.2, 0.5],
    'classifier__estimator__max_depth': [1, 3, 5],
    'classifier__estimator__min_samples_split': [2, 5, 10],
}

# ==============================
# Build AdaBoost Pipeline and GridSearchCV
# ==============================
base_estimator = DecisionTreeClassifier(random_state=42)
adaboost_model = AdaBoostClassifier(
    estimator=base_estimator,
    random_state=42
)

adaboost_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', adaboost_model)
])

grid_search = GridSearchCV(
    estimator=adaboost_pipe,
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    error_score='raise'
)

print("\n" + "="*80)
print("Starting GridSearchCV for AdaBoost...")
print("="*80)
grid_search.fit(X_temp, y_temp)

best_params = grid_search.best_params_
print("\n" + "="*80)
print("Best Hyperparameters found:")
print("="*80)
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest AUC-ROC (GridSearch): {grid_search.best_score_:.4f}")

# ==============================
# Re-evaluate Tuned AdaBoost Across ALL Metrics
# ==============================
# Reconstruct tuned model with best parameters
tuned_base_estimator = DecisionTreeClassifier(
    max_depth=best_params['classifier__estimator__max_depth'],
    min_samples_split=best_params['classifier__estimator__min_samples_split'],
    random_state=42
)

tuned_model = AdaBoostClassifier(
    estimator=tuned_base_estimator,
    n_estimators=best_params['classifier__n_estimators'],
    learning_rate=best_params['classifier__learning_rate'],
    random_state=42
)

tuned_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', tuned_model)])

print("\n--- Tuned AdaBoost Full Metric Evaluation ---\n")

tuned_metric_scores = {}
tuned_fold_rows     = []

for metric_key, sklearn_scorer in METRICS.items():
    scores = cross_val_score(tuned_pipe, X_temp, y_temp, cv=skf, scoring=sklearn_scorer, n_jobs=-1)
    tuned_metric_scores[metric_key] = scores

for i in range(skf.n_splits):
    row = {"model": "AdaBoost_Tuned", "fold": i + 1}
    for metric_key, scores in tuned_metric_scores.items():
        row[metric_key] = round(scores[i], 6)
    tuned_fold_rows.append(row)

print(f"{'Metric':<12} {'Mean':>8} {'Std':>8}")
print("-" * 30)
for metric_key, metric_lbl in METRIC_LABELS.items():
    mean = np.mean(tuned_metric_scores[metric_key])
    std  = np.std(tuned_metric_scores[metric_key])
    print(f"{metric_lbl:<12} {mean:>8.4f} {std:>8.4f}")

# ==============================
# Extract Baseline AdaBoost Stats
# ==============================
baseline_summary_df = pd.DataFrame(baseline_results)
baseline_fold_df    = pd.DataFrame(baseline_fold_rows)

adaboost_baseline_row   = baseline_summary_df[baseline_summary_df["model"] == "AdaBoost"].iloc[0]
adaboost_baseline_folds = baseline_fold_df[baseline_fold_df["model"] == "AdaBoost"].copy()

# ==============================
# Build Comparison DataFrames
# ==============================
comparison_rows = []
for label, source_row in [("AdaBoost_Baseline", adaboost_baseline_row), ("AdaBoost_Tuned", None)]:
    row = {"model": label}
    for metric_key in METRICS:
        if label == "AdaBoost_Baseline":
            mean = source_row[f"mean_{metric_key}"]
            std  = source_row[f"std_{metric_key}"]
        else:
            scores = tuned_metric_scores[metric_key]
            mean   = round(np.mean(scores), 6)
            std    = round(np.std(scores),  6)
        row[f"mean_{metric_key}"] = mean
        row[f"std_{metric_key}"]  = std
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)

# Append delta row
delta_row = {"model": "Delta (Tuned - Baseline)"}
for metric_key in METRICS:
    delta = (
        comparison_df.loc[comparison_df["model"] == "AdaBoost_Tuned",     f"mean_{metric_key}"].values[0]
      - comparison_df.loc[comparison_df["model"] == "AdaBoost_Baseline",  f"mean_{metric_key}"].values[0]
    )
    delta_row[f"mean_{metric_key}"] = round(delta, 6)
    delta_row[f"std_{metric_key}"]  = ""
comparison_df = pd.concat([comparison_df, pd.DataFrame([delta_row])], ignore_index=True)

# Per-fold comparison
adaboost_baseline_folds["model"] = "AdaBoost_Baseline"
tuned_fold_df        = pd.DataFrame(tuned_fold_rows)
fold_comparison_df   = pd.concat([adaboost_baseline_folds, tuned_fold_df], ignore_index=True)

# ==============================
# Save CSVs
# ==============================
best_params_df = pd.DataFrame(list(best_params.items()), columns=["parameter", "value"])
best_params_df.to_csv(CSV_DIR / "adaboost_best_hyperparameters.csv", index=False)
print(f"\n[Saved] Best hyperparameters         → adaboost_best_hyperparameters.csv")

cv_results_df = pd.DataFrame(grid_search.cv_results_).round(6)
cv_results_df.to_csv(CSV_DIR / "gridsearch_cv_results.csv", index=False)
print(f"[Saved] GridSearch CV results        → gridsearch_cv_results.csv")

search_summary_df = pd.DataFrame([{
    "best_auc_roc":    round(grid_search.best_score_, 6),
    "best_params":     str(best_params),
    "total_combinations": len(grid_search.cv_results_['params']),
    "cv_splits":       skf.n_splits
}])
search_summary_df.to_csv(CSV_DIR / "gridsearch_summary.csv", index=False)
print(f"[Saved] GridSearch summary           → gridsearch_summary.csv")

comparison_df.to_csv(CSV_DIR / "adaboost_baseline_vs_tuned_summary.csv", index=False)
print(f"[Saved] Comparison summary           → adaboost_baseline_vs_tuned_summary.csv")

fold_comparison_df.to_csv(CSV_DIR / "adaboost_baseline_vs_tuned_folds.csv", index=False)
print(f"[Saved] Per-fold comparison          → adaboost_baseline_vs_tuned_folds.csv")

# ==============================
# Plot Helpers
# ==============================
COLORS    = {"AdaBoost_Baseline": "steelblue", "AdaBoost_Tuned": "tomato"}
models_cmp = ["AdaBoost_Baseline", "AdaBoost_Tuned"]
metric_keys = list(METRIC_LABELS.keys())
metric_lbls = list(METRIC_LABELS.values())
fold_nums   = list(range(1, skf.n_splits + 1))

# ==============================
# Plot 1: GridSearch CV Results History
# ==============================
cv_results = grid_search.cv_results_
mean_scores = cv_results['mean_test_score']
rank_tests  = cv_results['rank_test_score']
iteration_nums = list(range(len(mean_scores)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(iteration_nums, mean_scores, c=rank_tests, cmap='RdYlGn', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.plot(iteration_nums, [np.max(mean_scores[:i+1]) for i in range(len(mean_scores))], 
        color='darkorange', lw=2, label='Best so far')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Rank (lower is better)', rotation=270, labelpad=15)
ax.set_title('GridSearchCV Results — AdaBoost', fontsize=14)
ax.set_xlabel('Parameter Combination Index')
ax.set_ylabel('Mean CV AUC-ROC Score')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "gridsearch_cv_results.png", dpi=150, bbox_inches='tight')
print(f"[Saved] GridSearch CV results        → gridsearch_cv_results.png")
plt.show(); plt.close()

# ==============================
# Plot 2: GridSearch Parameter Importance (Top 10 combinations)
# ==============================
cv_results_sorted = sorted(cv_results['params'], key=lambda x: grid_search.cv_results_['rank_test_score'][cv_results['params'].index(x)])
top_10_params = cv_results_sorted[:10]
top_10_scores = sorted(cv_results['mean_test_score'], reverse=True)[:10]

# Create parameter labels for top 10
param_labels = [f"Combo {i+1}" for i in range(10)]

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.barh(param_labels[::-1], top_10_scores[::-1], color='steelblue', alpha=0.85, edgecolor='black', linewidth=0.6)
for bar, score in zip(bars, top_10_scores[::-1]):
    width = bar.get_width()
    ax2.text(width - 0.005, bar.get_y() + bar.get_height() / 2,
             f"{width:.4f}", va='center', ha='right', fontsize=9, color='white', weight='bold')
ax2.set_title('Top 10 Parameter Combinations by AUC-ROC', fontsize=14)
ax2.set_xlabel('Mean CV AUC-ROC Score')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig2.savefig(IMAGE_DIR / "gridsearch_top_combinations.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Top parameter combinations   → gridsearch_top_combinations.png")
plt.show(); plt.close()

# ==============================
# Plot 3: Grouped Bar — Baseline vs Tuned
# ==============================
x         = np.arange(len(metric_keys))
bar_width = 0.32

fig3, ax3 = plt.subplots(figsize=(12, 6))
for idx, model_label in enumerate(models_cmp):
    row   = comparison_df[comparison_df["model"] == model_label].iloc[0]
    means = [row[f"mean_{m}"] for m in metric_keys]
    stds  = [row[f"std_{m}"]  for m in metric_keys]
    offset = (idx - 0.5) * bar_width
    bars  = ax3.bar(x + offset, means, bar_width,
                    yerr=stds, capsize=4,
                    label=model_label.replace("_", " "),
                    color=COLORS[model_label], alpha=0.85,
                    edgecolor='black', linewidth=0.6)
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.004,
                 f"{mean:.4f}", ha='center', va='bottom', fontsize=8)

all_means = [comparison_df.loc[comparison_df["model"].isin(models_cmp), f"mean_{m}"].astype(float).min()
             for m in metric_keys]
ax3.set_ylim(min(all_means) - 0.05, 1.05)
ax3.set_title('AdaBoost — Baseline vs Tuned (All Metrics)', fontsize=14)
ax3.set_xlabel('Metric')
ax3.set_ylabel('Score')
ax3.set_xticks(x)
ax3.set_xticklabels(metric_lbls)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig3.savefig(IMAGE_DIR / "adaboost_baseline_vs_tuned_bar.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Comparison bar chart         → adaboost_baseline_vs_tuned_bar.png")
plt.show(); plt.close()

# ==============================
# Plot 4: Per-Fold Line — Baseline vs Tuned
# ==============================
fig4, axes4 = plt.subplots(2, 3, figsize=(16, 9))
axes4 = axes4.flatten()

for ax_idx, (metric_key, metric_lbl) in enumerate(METRIC_LABELS.items()):
    ax = axes4[ax_idx]
    for model_label in models_cmp:
        scores = fold_comparison_df[fold_comparison_df["model"] == model_label][metric_key].tolist()
        ax.plot(fold_nums, scores, marker='o',
                label=model_label.replace("_", " "),
                color=COLORS[model_label], lw=2)
    ax.set_title(metric_lbl, fontsize=12)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_xticks(fold_nums)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes4[-1].set_visible(False)
fig4.suptitle('AdaBoost — Baseline vs Tuned per Fold', fontsize=14, y=1.01)
plt.tight_layout()
fig4.savefig(IMAGE_DIR / "adaboost_baseline_vs_tuned_fold_lines.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Per-fold comparison lines    → adaboost_baseline_vs_tuned_fold_lines.png")
plt.show(); plt.close()

# ==============================
# Plot 5: Delta Bar — Improvement per Metric
# ==============================
delta_row_data = comparison_df[comparison_df["model"] == "Delta (Tuned - Baseline)"].iloc[0]
deltas     = [float(delta_row_data[f"mean_{m}"]) for m in metric_keys]
bar_colors = ["seagreen" if d >= 0 else "tomato" for d in deltas]

fig5, ax5 = plt.subplots(figsize=(9, 5))
bars5 = ax5.bar(metric_lbls, deltas, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.6)
ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax5.set_title('AdaBoost — Tuning Improvement per Metric (Tuned − Baseline)', fontsize=13)
ax5.set_xlabel('Metric')
ax5.set_ylabel('Delta Score')
ax5.grid(axis='y', alpha=0.3)
for bar, delta in zip(bars5, deltas):
    va  = 'bottom' if delta >= 0 else 'top'
    off = 0.0005   if delta >= 0 else -0.0005
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + off,
             f"{delta:+.4f}", ha='center', va=va, fontsize=9)
plt.tight_layout()
fig5.savefig(IMAGE_DIR / "adaboost_tuning_delta.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Delta improvement chart      → adaboost_tuning_delta.png")
plt.show(); plt.close()

# ==============================
# Plot 6: Radar — Baseline vs Tuned
# ==============================
angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
angles += angles[:1]

fig6, ax6 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for model_label in models_cmp:
    row    = comparison_df[comparison_df["model"] == model_label].iloc[0]
    values = [float(row[f"mean_{m}"]) for m in metric_keys]
    values += values[:1]
    ax6.plot(angles, values, color=COLORS[model_label], lw=2,
             label=model_label.replace("_", " "))
    ax6.fill(angles, values, color=COLORS[model_label], alpha=0.15)

ax6.set_thetagrids(np.degrees(angles[:-1]), metric_lbls, fontsize=11)
ax6.set_ylim(0, 1)
ax6.set_title('AdaBoost — Baseline vs Tuned Radar', fontsize=13, pad=18)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax6.grid(alpha=0.3)
plt.tight_layout()
fig6.savefig(IMAGE_DIR / "adaboost_baseline_vs_tuned_radar.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Radar comparison             → adaboost_baseline_vs_tuned_radar.png")
plt.show(); plt.close()