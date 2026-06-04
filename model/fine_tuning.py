# -*- coding: utf-8 -*-
import os
import sys
import warnings
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore', message='X does not have valid feature names')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from input_processing.train_test_split import X_train, y_train
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
# Optuna Objective (AUC-ROC only for tuning)
# ==============================
def objective(trial):
    params = {
        'C':            trial.suggest_float('C', 1e-4, 100.0, log=True),
        'penalty':      trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
        'solver':       'saga',      # saga supports all penalties
        'max_iter':     trial.suggest_int('max_iter', 500, 3000),
        'class_weight': 'balanced',
        'random_state': 42,
    }
    # l1_ratio only applies to elasticnet
    if params['penalty'] == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    model = LogisticRegression(**params)
    pipe  = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    score = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    return score

# ==============================
# Run Optuna Study
# ==============================
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Hyperparameters found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest AUC-ROC (tuning): {study.best_value:.4f}")

# ==============================
# Re-evaluate Tuned LogReg Across ALL Metrics
# ==============================
tuned_model = LogisticRegression(
    **best_params,
    solver='saga',
    class_weight='balanced',
    random_state=42
)
tuned_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', tuned_model)])

print("\n--- Tuned Logistic Regression Full Metric Evaluation ---\n")

tuned_metric_scores = {}
tuned_fold_rows     = []

for metric_key, sklearn_scorer in METRICS.items():
    scores = cross_val_score(tuned_pipe, X_train, y_train, cv=skf, scoring=sklearn_scorer, n_jobs=-1)
    tuned_metric_scores[metric_key] = scores

for i in range(skf.n_splits):
    row = {"model": "LogReg_Tuned", "fold": i + 1}
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
# Extract Baseline LogReg Stats
# ==============================
baseline_summary_df = pd.DataFrame(baseline_results)
baseline_fold_df    = pd.DataFrame(baseline_fold_rows)

logreg_baseline_row   = baseline_summary_df[baseline_summary_df["model"] == "LogisticRegression"].iloc[0]
logreg_baseline_folds = baseline_fold_df[baseline_fold_df["model"] == "LogisticRegression"].copy()

# ==============================
# Build Comparison DataFrames
# ==============================
comparison_rows = []
for label, source_row in [("LogReg_Baseline", logreg_baseline_row), ("LogReg_Tuned", None)]:
    row = {"model": label}
    for metric_key in METRICS:
        if label == "LogReg_Baseline":
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
        comparison_df.loc[comparison_df["model"] == "LogReg_Tuned",     f"mean_{metric_key}"].values[0]
      - comparison_df.loc[comparison_df["model"] == "LogReg_Baseline",  f"mean_{metric_key}"].values[0]
    )
    delta_row[f"mean_{metric_key}"] = round(delta, 6)
    delta_row[f"std_{metric_key}"]  = ""
comparison_df = pd.concat([comparison_df, pd.DataFrame([delta_row])], ignore_index=True)

# Per-fold comparison
logreg_baseline_folds["model"] = "LogReg_Baseline"
tuned_fold_df        = pd.DataFrame(tuned_fold_rows)
fold_comparison_df   = pd.concat([logreg_baseline_folds, tuned_fold_df], ignore_index=True)

# ==============================
# Save CSVs
# ==============================
best_params_df = pd.DataFrame(list(best_params.items()), columns=["parameter", "value"])
best_params_df.to_csv(CSV_DIR / "best_hyperparameters.csv", index=False)
print(f"\n[Saved] Best hyperparameters         → best_hyperparameters.csv")

trials_df = study.trials_dataframe().round(6)
trials_df.to_csv(CSV_DIR / "optuna_trials.csv", index=False)
print(f"[Saved] All trial results            → optuna_trials.csv")

study_summary_df = pd.DataFrame([{
    "metric":     "best_auc_roc",
    "value":      round(study.best_value, 6),
    "best_trial": study.best_trial.number,
    "n_trials":   len(study.trials)
}])
study_summary_df.to_csv(CSV_DIR / "optuna_study_summary.csv", index=False)
print(f"[Saved] Study summary                → optuna_study_summary.csv")

comparison_df.to_csv(CSV_DIR / "logreg_baseline_vs_tuned_summary.csv", index=False)
print(f"[Saved] Comparison summary           → logreg_baseline_vs_tuned_summary.csv")

fold_comparison_df.to_csv(CSV_DIR / "logreg_baseline_vs_tuned_folds.csv", index=False)
print(f"[Saved] Per-fold comparison          → logreg_baseline_vs_tuned_folds.csv")

# ==============================
# Plot Helpers
# ==============================
COLORS    = {"LogReg_Baseline": "steelblue", "LogReg_Tuned": "tomato"}
models_cmp = ["LogReg_Baseline", "LogReg_Tuned"]
metric_keys = list(METRIC_LABELS.keys())
metric_lbls = list(METRIC_LABELS.values())
fold_nums   = list(range(1, skf.n_splits + 1))

# ==============================
# Plot 1: Optuna Optimization History
# ==============================
trial_numbers = [t.number for t in study.trials if t.value is not None]
trial_values  = [t.value  for t in study.trials if t.value is not None]
best_so_far   = [max(trial_values[:i+1]) for i in range(len(trial_values))]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(trial_numbers, trial_values, color='steelblue', alpha=0.5, s=25, label='Trial AUC-ROC')
ax.plot(trial_numbers, best_so_far, color='darkorange', lw=2, label='Best so far')
ax.set_title('Optuna Optimization History — Logistic Regression', fontsize=14)
ax.set_xlabel('Trial Number')
ax.set_ylabel('AUC-ROC')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGE_DIR / "optuna_optimization_history.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Optimization history         → optuna_optimization_history.png")
plt.show(); plt.close()

# ==============================
# Plot 2: Optuna Hyperparameter Importance
# ==============================
importances = optuna.importance.get_param_importances(study)
fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.barh(
    list(importances.keys())[::-1],
    list(importances.values())[::-1],
    color='steelblue'
)
for bar in bars:
    width = bar.get_width()
    ax2.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{width:.3f}", va='center', fontsize=9)
ax2.set_title('Hyperparameter Importance — Optuna (Logistic Regression)', fontsize=14)
ax2.set_xlabel('Importance Score')
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig2.savefig(IMAGE_DIR / "optuna_hyperparameter_importance.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Hyperparameter importance    → optuna_hyperparameter_importance.png")
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
ax3.set_title('Logistic Regression — Baseline vs Tuned (All Metrics)', fontsize=14)
ax3.set_xlabel('Metric')
ax3.set_ylabel('Score')
ax3.set_xticks(x)
ax3.set_xticklabels(metric_lbls)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig3.savefig(IMAGE_DIR / "logreg_baseline_vs_tuned_bar.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Comparison bar chart         → logreg_baseline_vs_tuned_bar.png")
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
fig4.suptitle('Logistic Regression — Baseline vs Tuned per Fold', fontsize=14, y=1.01)
plt.tight_layout()
fig4.savefig(IMAGE_DIR / "logreg_baseline_vs_tuned_fold_lines.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Per-fold comparison lines    → logreg_baseline_vs_tuned_fold_lines.png")
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
ax5.set_title('Logistic Regression — Tuning Improvement per Metric (Tuned − Baseline)', fontsize=13)
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
fig5.savefig(IMAGE_DIR / "logreg_tuning_delta.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Delta improvement chart      → logreg_tuning_delta.png")
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
ax6.set_title('Logistic Regression — Baseline vs Tuned Radar', fontsize=13, pad=18)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax6.grid(alpha=0.3)
plt.tight_layout()
fig6.savefig(IMAGE_DIR / "logreg_baseline_vs_tuned_radar.png", dpi=150, bbox_inches='tight')
print(f"[Saved] Radar comparison             → logreg_baseline_vs_tuned_radar.png")
plt.show(); plt.close()