# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path()

from input_processing.data_loading import df
import pandas as pd

print(df.head())

# ── Engagement Efficiency Features ────────────────────────────────────────────
# How much revenue are we spending per website visit?
df['CostPerVisit']         = df['AdSpend'] / (df['WebsiteVisits'] + 1)

# How much does each click cost us?
df['CostPerClick']         = df['AdSpend'] / (df['EmailClicks'] + 1)

# Email engagement rate: of those who opened, how many clicked?
df['EmailEngagementRate']  = df['EmailClicks'] / (df['EmailOpens'] + 1)

# ── Customer Quality Features ─────────────────────────────────────────────────
# High loyalty + repeat purchases = strong retention signal
df['CustomerValue']        = df['LoyaltyPoints'] * df['PreviousPurchases']

# Income relative to ad spend: can the customer afford what we're selling?
df['IncomeToAdSpend']      = df['Income'] / (df['AdSpend'] + 1)

# ── Behavioural Engagement Score ──────────────────────────────────────────────
# Composite site engagement: deeper visits + time = more intent
df['SiteEngagementScore']  = df['PagesPerVisit'] * df['TimeOnSite']

# Virality signal: social shares relative to visits
df['SocialAmplification']  = df['SocialShares'] / (df['WebsiteVisits'] + 1)

# ── Click-Through Quality ─────────────────────────────────────────────────────
# CTR tells reach efficiency; did that reach actually engage on-site?
df['CTR_x_PagesPerVisit']  = df['ClickThroughRate'] * df['PagesPerVisit']

# ── Age Bucketing ─────────────────────────────────────────────────────────────
# Segment customers into life-stage groups for campaign targeting
df['AgeBand'] = pd.cut(df['Age'],
                        bins=[0, 25, 35, 45, 60, 100],
                        labels=['Gen Z', 'Young Adult', 'Mid Career', 'Senior', 'Retired'])

print("✅ New features created:")
new_cols = [
    'CostPerVisit', 'CostPerClick', 'EmailEngagementRate',
    'CustomerValue', 'IncomeToAdSpend', 'SiteEngagementScore',
    'SocialAmplification', 'CTR_x_PagesPerVisit', 'AgeBand'
]
print(df[new_cols].head(3))