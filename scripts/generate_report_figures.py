"""Generate all report-quality figures for the academic DOCX report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'report_figures'
OUT.mkdir(parents=True, exist_ok=True)

# Load data
model = pd.read_parquet(ROOT / 'data/processed/eurocontrol_model_table.parquet')
model['date'] = pd.to_datetime(model['date'])
model['year'] = model['date'].dt.year
preds = pd.read_csv(ROOT / 'outputs/tables/test_predictions_eurocontrol.csv')
preds['date'] = pd.to_datetime(preds['date'])
feat_imp = pd.read_csv(ROOT / 'outputs/tables/feature_importance_eurocontrol.csv')
print('Data loaded.')

# ===== FIG 1: Daily movements time series =====
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(model['date'], model['total_movements'], alpha=0.25, color='#2563eb')
ax.plot(model['date'], model['total_movements'], linewidth=0.5, color='#2563eb')
ax.annotate('COVID-19\nLockdown', xy=(pd.Timestamp('2020-03-15'), 200),
            xytext=(pd.Timestamp('2020-09-01'), 700),
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
            fontsize=9, color='#dc2626', fontweight='bold')
for d, lbl in [(pd.Timestamp('2023-06-08'), 'Train | Valid'),
               (pd.Timestamp('2024-10-18'), 'Valid | Test')]:
    ax.axvline(d, color='#6b7280', ls='--', lw=0.8, alpha=0.7)
    ax.text(d, 1500, lbl, fontsize=7, ha='center', color='#6b7280',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d1d5db'))
ax.set_ylabel('Daily IFR Movements')
ax.set_xlabel('Date')
ax.set_title('Figure 1. Daily IFR Flight Movements at Madrid-Barajas (LEMD), 2017--2026')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(OUT / 'fig01_timeseries.png')
plt.close()
print('  Fig 1')

# ===== FIG 2: Yearly averages =====
yearly = model.groupby('year')['total_movements'].mean()
fig, ax = plt.subplots(figsize=(8, 3.2))
colors = ['#dc2626' if y == 2020 else '#f59e0b' if y == 2021 else '#2563eb' for y in yearly.index]
bars = ax.bar(yearly.index, yearly.values, color=colors, width=0.7, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, yearly.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Avg. Daily Movements')
ax.set_xlabel('Year')
ax.set_title('Figure 2. Average Daily IFR Movements by Year')
ax.set_ylim(0, 1350)
plt.savefig(OUT / 'fig02_yearly.png')
plt.close()
print('  Fig 2')

# ===== FIG 3: Day-of-week + Monthly seasonality =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
dow_avg = model.groupby('dow')['total_movements'].mean()
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
colors_dow = ['#2563eb'] * 7
colors_dow[5] = '#dc2626'
colors_dow[4] = '#059669'
ax1.bar(dow_names, dow_avg.values, color=colors_dow, width=0.6)
ax1.set_ylabel('Avg. Daily Movements')
ax1.set_title('(a) Day-of-Week Pattern')
ax1.set_ylim(850, 1060)

recent = model[model['year'].isin([2022, 2023, 2024, 2025])]
monthly = recent.groupby(recent['date'].dt.month)['total_movements'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.plot(month_names, monthly.values, 'o-', color='#2563eb', linewidth=2, markersize=5)
ax2.fill_between(range(12), monthly.values, alpha=0.12, color='#2563eb')
ax2.set_ylabel('Avg. Daily Movements')
ax2.set_title('(b) Monthly Seasonality (2022--2025)')
ax2.set_ylim(900, 1200)
ax2.tick_params(axis='x', rotation=45)
fig.suptitle('Figure 3. Temporal Patterns in Flight Traffic', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'fig03_temporal_patterns.png')
plt.close()
print('  Fig 3')

# ===== FIG 4: ACPS distribution =====
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(model['acps'], bins=40, color='#2563eb', alpha=0.7, edgecolor='white', linewidth=0.5)
p60 = np.percentile(model['acps'], 60)
p85 = np.percentile(model['acps'], 85)
ax.axvline(p60, color='#f59e0b', ls='--', lw=1.5, label=f'60th pctl = {p60:.1f} (Low|Medium)')
ax.axvline(p85, color='#dc2626', ls='--', lw=1.5, label=f'85th pctl = {p85:.1f} (Medium|High)')
ax.legend()
ax.set_xlabel('ACPS Score (0--100)')
ax.set_ylabel('Number of Days')
ax.set_title('Figure 4. Distribution of Airport Congestion Pressure Score')
plt.savefig(OUT / 'fig04_acps_distribution.png')
plt.close()
print('  Fig 4')

# ===== FIG 5: Feature importance =====
top15 = feat_imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.2))
colors_fi = ['#2563eb' if v > 0.01 else '#94a3b8' for v in top15['importance']]
ax.barh(range(len(top15)), top15['importance'], color=colors_fi, height=0.6)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'], fontsize=9)
ax.set_xlabel('Permutation Importance')
ax.set_title('Figure 5. Top 15 Features by Permutation Importance')
plt.savefig(OUT / 'fig05_feature_importance.png')
plt.close()
print('  Fig 5')

# ===== FIG 6: Actual vs Predicted =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[3, 1])
ax1.plot(preds['date'], preds['acps'], linewidth=1.2, color='#2563eb', label='Actual ACPS')
ax1.plot(preds['date'], preds['acps_predicted'], linewidth=1.2, color='#059669',
         linestyle='--', label='Predicted ACPS', alpha=0.85)
ax1.set_ylabel('ACPS Score')
ax1.set_title('Figure 6. Regression: Actual vs Predicted ACPS (Test Set, Oct 2024 -- Feb 2026)')
ax1.legend(loc='lower right')
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
residuals = preds['acps'] - preds['acps_predicted']
ax2.bar(preds['date'], residuals, width=1,
        color=np.where(residuals >= 0, '#2563eb', '#dc2626'), alpha=0.6)
ax2.axhline(0, color='black', lw=0.5)
ax2.set_ylabel('Residual')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig(OUT / 'fig06_predictions.png')
plt.close()
print('  Fig 6')

# ===== FIG 7: Confusion matrix =====
cm = confusion_matrix(preds['congestion_class'], preds['congestion_predicted'],
                      labels=['Low', 'Medium', 'High'])
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
labels = ['Low', 'Medium', 'High']
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Figure 7. Confusion Matrix (Test Set)')
for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                fontweight='bold', color=color)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(OUT / 'fig07_confusion_matrix.png')
plt.close()
print('  Fig 7')

# ===== FIG 8: Scatter actual vs predicted =====
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(preds['acps'], preds['acps_predicted'], s=12, alpha=0.4, color='#2563eb')
lims = [min(preds['acps'].min(), preds['acps_predicted'].min()) - 2,
        max(preds['acps'].max(), preds['acps_predicted'].max()) + 2]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfect prediction')
ax.set_xlabel('Actual ACPS')
ax.set_ylabel('Predicted ACPS')
ax.set_title('Figure 8. Prediction Scatter (R\u00b2 = 0.96)')
ax.legend()
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
plt.savefig(OUT / 'fig08_scatter.png')
plt.close()
print('  Fig 8')

# ===== FIG 9: Weather correlation =====
weather_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m',
                'surface_pressure', 'cloud_cover', 'relative_humidity_2m']
existing = [c for c in weather_cols if c in model.columns]
corrs = model[existing + ['acps']].corr()['acps'].drop('acps').sort_values()
fig, ax = plt.subplots(figsize=(7, 3))
colors_wc = ['#dc2626' if v < 0 else '#059669' for v in corrs.values]
ax.barh(range(len(corrs)), corrs.values, color=colors_wc, height=0.5)
ax.set_yticks(range(len(corrs)))
names = [c.replace('_2m', '').replace('_10m', '').replace('_', ' ').title() for c in corrs.index]
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Pearson Correlation with ACPS')
ax.set_title('Figure 9. Weather Variables Correlation with Congestion')
ax.axvline(0, color='black', lw=0.5)
plt.savefig(OUT / 'fig09_weather_correlation.png')
plt.close()
print('  Fig 9')

# ===== FIG 10: COVID recovery =====
fig, ax = plt.subplots(figsize=(8, 3.5))
cmap = {2019: '#2563eb', 2020: '#dc2626', 2021: '#f59e0b',
        2022: '#94a3b8', 2023: '#94a3b8', 2024: '#059669'}
for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
    subset = model[model['year'] == yr].copy()
    subset['doy'] = subset['date'].dt.dayofyear
    weekly = subset.groupby(subset['doy'] // 7)['total_movements'].mean()
    lw = 2 if yr in [2019, 2020, 2024] else 1
    alpha = 1.0 if yr in [2019, 2020, 2024] else 0.45
    ax.plot(weekly.index * 7, weekly.values, label=str(yr),
            color=cmap[yr], linewidth=lw, alpha=alpha)
ax.set_xlabel('Day of Year')
ax.set_ylabel('Avg. Daily Movements')
ax.set_title('Figure 10. COVID-19 Impact and Traffic Recovery')
ax.legend(ncol=3, fontsize=8)
ax.set_xlim(0, 365)
plt.savefig(OUT / 'fig10_covid_recovery.png')
plt.close()
print('  Fig 10')

print('All 10 figures generated.')
