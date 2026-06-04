# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ROOT_DIR = Path(__file__).parent.parent


def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path()

# Import df globally so it's available everywhere in this file
from input_processing.data_loading import df


# ── Chart 1: Conversion Rate Distribution ────────────────────────────────────
def plot_conversion_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of ConversionRate
    axes[0].hist(df['ConversionRate'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_title('Conversion Rate Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Conversion Rate')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df['ConversionRate'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["ConversionRate"].mean():.3f}')
    axes[0].legend()

    # Conversion Rate by Campaign Type
    df.boxplot(column='ConversionRate', by='CampaignType', ax=axes[1], patch_artist=True)
    axes[1].set_title('Conversion Rate by Campaign Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Campaign Type')
    axes[1].set_ylabel('Conversion Rate')
    plt.suptitle('')

    plt.tight_layout()
    output_path_0 = ROOT_DIR / 'output_storage' / 'images' / 'chart_conversion_distribution.png'
    plt.savefig(output_path_0, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path_0


def print_summary(df, chart_path):
    print(f"Saved chart: {chart_path}")
    print(f"\nOverall Conversion Rate (binary): {df['Conversion'].mean():.3f}")
    print(f"Avg ConversionRate (continuous): {df['ConversionRate'].mean():.4f}")
    print(f"Max ConversionRate: {df['ConversionRate'].max():.4f}")
    print(f"Min ConversionRate: {df['ConversionRate'].min():.4f}")


# ── Run Chart 1 ───────────────────────────────────────────────────────────────
chart_path_0 = plot_conversion_distribution(df)
print_summary(df, chart_path_0)

# ── Chart 2: Conversion by Channel & Campaign Type ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Average conversion rate per channel
channel_conv = df.groupby('CampaignChannel')['Conversion'].mean().sort_values(ascending=False)
bars = axes[0].bar(channel_conv.index, channel_conv.values,
                   color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'],
                   edgecolor='white')
axes[0].set_title('Avg Conversion Rate by Channel', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Conversion Rate')
axes[0].set_xlabel('Campaign Channel')
for bar, val in zip(bars, channel_conv.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

# Conversion rate by Campaign Type
type_conv = df.groupby('CampaignType')['Conversion'].mean().sort_values(ascending=False)
colors = ['#1565C0', '#1976D2', '#42A5F5', '#90CAF9']
axes[1].bar(type_conv.index, type_conv.values, color=colors, edgecolor='white')
axes[1].set_title('Avg Conversion Rate by Campaign Type', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Conversion Rate')
axes[1].set_xlabel('Campaign Type')
for i, val in enumerate(type_conv.values):
    axes[1].text(i, val + 0.002, f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
output_path_1 = ROOT_DIR / 'output_storage' / 'images' / 'chart_channel_campaigntype.png'
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)


# ── Chart 3: Correlation Heatmap ──────────────────────────────────────────────
# Shows how strongly each numeric feature relates to Conversion
# (Closer to 1 or -1 = stronger relationship)

numeric_cols = [
    'Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
    'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
    'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints', 'Conversion'
]

plt.figure(figsize=(13, 9))
corr_matrix = df[numeric_cols].corr()

sns.heatmap(df.corr(numeric_only=True), cmap='viridis', annot=True)

plt.title('Correlation Heatmap\n(How closely each feature relates to Conversion)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
output_path_2 = ROOT_DIR / 'output_storage' / 'images' / 'chart_heatmap.png'
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')
plt.show()

# Print top correlations with Conversion
print("\nHow each feature correlates with CONVERSION:")
print(corr_matrix['Conversion'].sort_values(ascending=False).to_string())