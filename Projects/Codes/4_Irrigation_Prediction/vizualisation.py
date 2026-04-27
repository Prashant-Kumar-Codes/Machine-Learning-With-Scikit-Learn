"""
=============================================================
  IRRIGATION DATASET — FULL EDA VISUALIZATION PLAYBOOK
=============================================================
Target variable : Irrigation_Need  (Low / Medium / High)
Task            : Multi-class Classification
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F8F6',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linestyle':   '--',
    'font.size':        11,
})

PALETTE = {
    'Low':    '#1D9E75',   # green
    'Medium': '#BA7517',   # amber
    'High':   '#E24B4A',   # red
}
ORDER = ['Low', 'Medium', 'High']

# ── Load your data ────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(os.getcwd()))
print(base_dir, '\n\n')
df = pd.read_csv(r'D:\Codes\Artificial_Intelligence\Machine_Learning\Machine-Learning-With-Scikit-Learn\Projects\Datasets\predicting_irrigation_need\train.csv')
#df = pd.read_csv(fr'{base_dir}/Datasets/predicting_irrigation_need/train.csv')   # ← replace with your path

# ── Column groups ─────────────────────────────────────────
NUMERIC = [
    'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
    'Electrical_Conductivity', 'Temperature_C', 'Humidity',
    'Rainfall_mm', 'Sunlight_Hours', 'Field_Area_hectare',
    'Previous_Irrigation_mm'
]
CATEGORICAL = [
    'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
    'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
]
TARGET = 'Irrigation_Need'


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 1 — Target distribution                           ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  First thing always. Know what you are predicting.
  Multi-class imbalance affects which metric to use
  (accuracy vs macro-F1) and whether to use class_weight='balanced'.

WHAT TO LOOK FOR:
  - Any class < 15% of total → imbalanced, handle it
  - All roughly equal → safe to use accuracy
  - Note the order (Low < Medium < High) — this is ordinal,
    which could be exploited with ordinal encoding or
    an ordinal regression approach
"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Plot 1 — Target Variable: Irrigation_Need', fontsize=13, fontweight='bold', y=1.01)

counts = df[TARGET].value_counts()[ORDER]
pcts   = df[TARGET].value_counts(normalize=True)[ORDER] * 100

axes[0].bar(ORDER, counts, color=[PALETTE[c] for c in ORDER], width=0.5, edgecolor='white')
for i, (n, p) in enumerate(zip(counts, pcts)):
    axes[0].text(i, n + 5, f'{n}\n({p:.1f}%)', ha='center', fontsize=10)
axes[0].set_title('Class counts')
axes[0].set_ylabel('Count')

axes[1].pie(counts, labels=ORDER, colors=[PALETTE[c] for c in ORDER],
            autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
axes[1].set_title('Class proportions')

plt.tight_layout()
plt.savefig('plot1_target.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 2 — Numeric feature distributions (histograms)    ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Reveals skewness, bimodal peaks, impossible values,
  and whether scaling or log-transforms are needed.

WHAT TO LOOK FOR:
  - Rainfall_mm / Previous_Irrigation_mm → likely right-skewed
    → apply np.log1p() before feeding to linear models / KNN
  - Soil_Moisture, Humidity → likely bounded [0,100] — check
    if any values exceed 100 (data error)
  - Electrical_Conductivity → often very right-skewed in soil data
  - Any spike at 0 or 999 → missing value sentinel
"""
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
fig.suptitle('Plot 2 — Numeric Feature Distributions', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(NUMERIC):
    sk = skew(df[col].dropna())
    color = '#E24B4A' if abs(sk) > 1 else '#378ADD'
    axes[i].hist(df[col].dropna(), bins=35, color=color, alpha=0.8, edgecolor='white')
    axes[i].set_title(f'{col}\nskew={sk:.2f}', fontsize=9)
    axes[i].tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('plot2_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Quick print — which features need log transform
print("\n── Features with |skew| > 1 (log-transform candidates) ──")
for col in NUMERIC:
    s = skew(df[col].dropna())
    if abs(s) > 1:
        print(f"  {col:35s}  skew = {s:+.2f}")


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 3 — Distribution colored by target class          ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  The MOST important numeric visualization for classification.
  When class distributions are separated for a feature,
  that feature has discriminating power. When they overlap
  completely, the feature is weak (by itself).

WHAT TO LOOK FOR:
  - Rainfall_mm: expect High irrigation need where rainfall is LOW
  - Soil_Moisture: High need where moisture is LOW
  - Temperature_C: High need in hot weather
  - If three curves completely overlap → feature may be weak
  - If curves are well separated → feature is very predictive
"""
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Plot 3 — Feature Distributions by Irrigation_Need Class', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(NUMERIC):
    for cls in ORDER:
        vals = df[df[TARGET] == cls][col].dropna()
        axes[i].hist(vals, bins=25, alpha=0.55, label=cls,
                     color=PALETTE[cls], edgecolor='white', density=True)
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(labelsize=8)
    if i == 0:
        axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig('plot3_class_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 4 — Box plots (outlier detection)                 ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Box plots show median, IQR (25th–75th percentile), and
  outlier points beyond 1.5×IQR. Also great for comparing
  feature spread across the three irrigation classes.

WHAT TO LOOK FOR:
  - Rainfall_mm / Previous_Irrigation_mm: likely have extreme outliers
  - If median shifts clearly across Low → Medium → High for a feature,
    that feature is highly predictive of irrigation need
  - Very long whiskers = high variance; consider capping at 1st/99th percentile
"""
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Plot 4 — Box Plots per Feature × Irrigation Class', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(NUMERIC):
    data_by_class = [df[df[TARGET] == cls][col].dropna() for cls in ORDER]
    bp = axes[i].boxplot(data_by_class, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, cls in zip(bp['boxes'], ORDER):
        patch.set_facecolor(PALETTE[cls])
        patch.set_alpha(0.7)
    axes[i].set_xticklabels(ORDER, fontsize=8)
    axes[i].set_title(col, fontsize=9)

plt.tight_layout()
plt.savefig('plot4_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# Print outlier summary
print("\n── Outlier summary (IQR method) ──")
for col in NUMERIC:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    pct = n_out / len(df) * 100
    print(f"  {col:35s}  outliers = {n_out:4d}  ({pct:.1f}%)")


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 5 — Correlation heatmap                           ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Reveals multicollinearity between features.
  Also shows which numeric features correlate most with
  an encoded version of the target.

WHAT TO LOOK FOR:
  - Humidity ↔ Rainfall_mm: likely correlated (both about wetness)
  - Temperature_C ↔ Sunlight_Hours: likely correlated
  - If two features have |corr| > 0.85 → consider dropping one
  - Last row/col (target encoded): high |corr| = predictive feature
"""
df_enc = df[NUMERIC + [TARGET]].copy()
df_enc[TARGET] = df_enc[TARGET].map({'Low': 0, 'Medium': 1, 'High': 2})

fig, ax = plt.subplots(figsize=(13, 10))
fig.suptitle('Plot 5 — Correlation Heatmap (numeric features + encoded target)', fontsize=13, fontweight='bold')

corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle only
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 9}, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('plot5_correlation.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 6 — Categorical feature counts                    ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Shows class balance within each categorical feature and
  reveals rare categories that might cause issues with
  one-hot encoding (a category appearing only 5 times
  will give a near-zero variance column).

WHAT TO LOOK FOR:
  - Very rare categories (<2% of data) → group into 'Other'
  - Mulching_Used (Yes/No) → check if one dominates heavily
  - Irrigation_Type might be highly correlated with target —
    some irrigation types are only used for high-need crops
"""
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle('Plot 6 — Categorical Feature Value Counts', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL):
    vc = df[col].value_counts()
    axes[i].barh(vc.index, vc.values, color='#378ADD', alpha=0.8, edgecolor='white')
    for j, (idx, val) in enumerate(zip(vc.index, vc.values)):
        axes[i].text(val + 0.5, j, str(val), va='center', fontsize=8)
    axes[i].set_title(col, fontsize=10)
    axes[i].invert_yaxis()

plt.tight_layout()
plt.savefig('plot6_categorical.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 7 — Categorical features × target (stacked bar)   ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  For classification, you need to know which categories
  are associated with which irrigation need level.
  A stacked bar with percentages is the clearest way to
  see this relationship.

WHAT TO LOOK FOR:
  - Season: Zaid (summer) → expect more High irrigation
  - Soil_Type: Sandy soil → dries faster → more irrigation
  - Mulching_Used=Yes → expect lower irrigation need
  - Crop_Type: Sugarcane → expect High, Wheat → Low/Medium
  - If proportions are the same across all categories of a
    feature → that feature has little predictive power for target
"""
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Plot 7 — Target Distribution within Each Categorical Feature', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL):
    ct = df.groupby([col, TARGET]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=ORDER, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='bar', stacked=True, ax=axes[i],
                color=[PALETTE[c] for c in ORDER], edgecolor='white', width=0.6)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('% share')
    axes[i].tick_params(axis='x', rotation=30, labelsize=8)
    axes[i].legend(ORDER, fontsize=8, loc='upper right')
    axes[i].set_ylim(0, 115)

plt.tight_layout()
plt.savefig('plot7_cat_vs_target.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 8 — Missing value analysis                        ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Missing data is not random. Its pattern carries signal.
  You must see it before imputing.

WHAT TO LOOK FOR:
  - Any column > 30% missing → consider dropping or using
    a model that handles NaN natively (LightGBM, CatBoost)
  - Columns always missing together → correlated missingness,
    a single indicator flag covers both
  - If this is soil/agriculture data, Electrical_Conductivity
    and Organic_Carbon often have missing entries (lab tests)
"""
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_df = missing_pct[missing_pct > 0]

if len(missing_df) == 0:
    print("\n✓ No missing values found in dataset.")
else:
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.5)))
    fig.suptitle('Plot 8 — Missing Value Analysis', fontsize=13, fontweight='bold')
    colors = ['#E24B4A' if p > 30 else '#BA7517' if p > 10 else '#1D9E75'
              for p in missing_df.values]
    ax.barh(missing_df.index, missing_df.values, color=colors, edgecolor='white')
    ax.axvline(30, color='#E24B4A', linestyle='--', alpha=0.7, label='30% threshold')
    ax.axvline(10, color='#BA7517', linestyle='--', alpha=0.7, label='10% threshold')
    ax.set_xlabel('% Missing')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plot8_missing.png', dpi=150, bbox_inches='tight')
    plt.show()

# Always print the full table
print("\n── Missing value summary ──")
print(missing_pct.to_string())


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 9 — Pair plot (top 5 numeric features)            ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Shows pairwise scatter plots between the most important
  features colored by class. Reveals interaction effects,
  clusters, and non-linear decision boundaries that
  a model needs to learn.

WHAT TO LOOK FOR:
  - If two features together create a visible cluster per class
    → that feature pair is very powerful together
  - Diagonal: KDE per class (shape of distribution per class)
  - Rainfall_mm vs Soil_Moisture: should show an interaction
    (high rainfall → high moisture → low irrigation need)
"""
# Use top 5 correlated features with target for clarity
df_enc_tmp = df[NUMERIC].copy()
target_enc = df[TARGET].map({'Low': 0, 'Medium': 1, 'High': 2})
correlations_with_target = df_enc_tmp.corrwith(target_enc).abs().sort_values(ascending=False)
top5 = correlations_with_target.head(5).index.tolist()

pair_df = df[top5 + [TARGET]].copy()
g = sns.pairplot(pair_df, hue=TARGET, hue_order=ORDER,
                 palette=PALETTE, diag_kind='kde',
                 plot_kws={'alpha': 0.5, 's': 20},
                 diag_kws={'fill': True, 'alpha': 0.4})
g.fig.suptitle('Plot 9 — Pair Plot (top 5 correlated features × class)', y=1.01, fontsize=13, fontweight='bold')
plt.savefig('plot9_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 10 — Violin plots                                 ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Violin plots = box plot + KDE. Shows the full shape of the
  distribution per class. Better than a box plot when
  distributions are bimodal or asymmetric per class.

WHAT TO LOOK FOR:
  - A feature where the three violins are clearly at different
    positions = strong feature
  - Rainfall_mm: expect the 'High' irrigation class to have
    a violin concentrated at LOW rainfall values
  - Soil_Moisture: expect 'Low' need class to have violin
    concentrated at HIGH moisture values
"""
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle('Plot 10 — Violin Plots: Feature Distribution per Irrigation Class', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(NUMERIC):
    sns.violinplot(data=df, x=TARGET, y=col, order=ORDER,
                   palette=PALETTE, ax=axes[i], inner='box',
                   cut=0, linewidth=0.8)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=20, labelsize=8)

plt.tight_layout()
plt.savefig('plot10_violin.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 11 — Quick feature importance (Random Forest)     ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Gives a data-driven ranking of ALL features (numeric +
  encoded categorical) in ~30 seconds. Tells you where to
  focus your feature engineering effort.

WHAT TO LOOK FOR:
  - Rainfall_mm, Soil_Moisture, Temperature_C → expect these
    near the top (domain knowledge confirms this)
  - Irrigation_Type being high → might be target leakage!
    Check if irrigation type is decided BEFORE or AFTER need
  - Very low importance features (< 0.01) → safe to drop,
    reduce noise and overfitting
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df_rf = df.copy()
for col in CATEGORICAL:
    df_rf[col] = LabelEncoder().fit_transform(df_rf[col].astype(str))
df_rf[TARGET] = LabelEncoder().fit_transform(df_rf[TARGET])
df_rf = df_rf.fillna(df_rf.median(numeric_only=True))

X = df_rf.drop(columns=[TARGET])
y = df_rf[TARGET]

rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X, y)

fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(fi) * 0.35)))
fig.suptitle('Plot 11 — Random Forest Feature Importances', fontsize=13, fontweight='bold')

colors = ['#E24B4A' if v > 0.08 else '#378ADD' if v > 0.03 else '#888780' for v in fi.values]
ax.barh(fi.index, fi.values, color=colors, edgecolor='white')
ax.axvline(0.03, color='#888780', linestyle='--', alpha=0.6, label='0.03 threshold')
ax.set_xlabel('Importance score')
ax.legend()

plt.tight_layout()
plt.savefig('plot11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n── Feature importances (descending) ──")
print(fi.sort_values(ascending=False).to_string())


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 12 — Numeric feature correlations (with target)   ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  A clean ranked bar chart of each feature's Pearson
  correlation with the encoded target. Quick signal check —
  helps you see which features have a monotonic relationship
  with irrigation need.

WHAT TO LOOK FOR:
  - Negative correlation: Rainfall_mm, Soil_Moisture, Humidity
    → as these increase, irrigation NEED decreases (makes sense!)
  - Positive correlation: Temperature_C, Sunlight_Hours
    → as heat/sun increases, more irrigation is needed
  - Feature near zero → consider non-linear importance (plot 11)
    before dropping, it might still matter for tree models
"""
target_enc = df[TARGET].map({'Low': 0, 'Medium': 1, 'High': 2})
corrs = df[NUMERIC].corrwith(target_enc).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Plot 12 — Feature Correlation with Irrigation_Need (encoded)', fontsize=13, fontweight='bold')

colors = ['#E24B4A' if v > 0 else '#1D9E75' for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson correlation with target (Low=0, Med=1, High=2)')

plt.tight_layout()
plt.savefig('plot12_target_correlation.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 13 — Region × Season × Irrigation Need heatmap   ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Interaction plots reveal when two categorical features
  TOGETHER are more predictive than each alone.
  Region + Season is a classic interaction in crop datasets.

WHAT TO LOOK FOR:
  - North + Zaid (summer) → likely very High irrigation need
  - South + Rabi (winter) → likely Low
  - If all cells are the same color → no interaction, features
    are independent of each other w.r.t. the target
"""
target_enc_series = df[TARGET].map({'Low': 0, 'Medium': 1, 'High': 2})
pivot = df.copy()
pivot['target_enc'] = target_enc_series
heat = pivot.groupby(['Region', 'Season'])['target_enc'].mean().unstack()

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Plot 13 — Interaction: Region × Season → Avg Irrigation Need', fontsize=13, fontweight='bold')
sns.heatmap(heat, annot=True, fmt='.2f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, vmin=0, vmax=2)
ax.set_xlabel('Season')
ax.set_ylabel('Region')
plt.tight_layout()
plt.savefig('plot13_interaction_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 14 — Soil Type × Crop Type × Irrigation Need      ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  Another critical interaction: what soil a crop is grown in
  determines how fast moisture drains and thus how much
  irrigation is needed. This combination might be more
  predictive than either feature alone.

WHAT TO LOOK FOR:
  - Sandy soil + Sugarcane → expect very High need
  - Loamy soil + Wheat → moderate to Low
  - If you see a strong pattern here → create an interaction
    feature: df['soil_crop'] = df['Soil_Type'] + '_' + df['Crop_Type']
"""
pivot2 = df.copy()
pivot2['target_enc'] = target_enc_series
heat2 = pivot2.groupby(['Soil_Type', 'Crop_Type'])['target_enc'].mean().unstack()

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Plot 14 — Interaction: Soil Type × Crop Type → Avg Irrigation Need', fontsize=13, fontweight='bold')
sns.heatmap(heat2, annot=True, fmt='.2f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, vmin=0, vmax=2)
ax.set_xlabel('Crop Type')
ax.set_ylabel('Soil Type')
plt.tight_layout()
plt.savefig('plot14_soil_crop_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  PLOT 15 — SHAP summary (post-model explainability)     ║
# ╚══════════════════════════════════════════════════════════╝
"""
WHY:
  SHAP goes beyond feature importance — it shows HOW each
  feature pushes predictions toward each class. Each dot is
  one sample, color = feature value, x-axis = impact.

WHAT TO LOOK FOR:
  - High rainfall (blue dot) pushing LEFT → decreasing irrigation need
  - Low soil moisture (red dot) pushing RIGHT → increasing need
  - Any feature where dots don't separate → weak feature
  - Features near top → most impactful overall

NOTE: Run this AFTER you have a trained model.
      pip install shap
"""
import shap

# Use the random forest from Plot 11 (or replace with your best model)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# For multi-class, shap_values is a list [class0, class1, class2]
# Plot for class 'High' (index 2 after LabelEncoding)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values[2], X, plot_type='dot',
                  show=False, max_display=15)
plt.title('Plot 15 — SHAP Summary (class: High irrigation need)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot15_shap.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  SUMMARY PRINT                                          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "="*60)
print("  EDA COMPLETE — KEY DECISIONS CHECKLIST")
print("="*60)
print(f"  Rows          : {len(df):,}")
print(f"  Columns       : {df.shape[1]}")
print(f"  Numeric feats : {len(NUMERIC)}")
print(f"  Categorical   : {len(CATEGORICAL)}")
print(f"\n  Target distribution:")
for cls in ORDER:
    n = (df[TARGET] == cls).sum()
    print(f"    {cls:8s}: {n:5d}  ({n/len(df)*100:.1f}%)")
print(f"\n  Missing values: {df.isnull().sum().sum()} total cells")
print(f"\n  High-skew features (|skew|>1):")
for col in NUMERIC:
    s = skew(df[col].dropna())
    if abs(s) > 1:
        print(f"    {col} → log1p transform recommended")
print("\n" + "="*60)