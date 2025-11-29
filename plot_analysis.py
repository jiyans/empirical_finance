"""
Data Visualization for Empirical Finance Repository
====================================================

Generates plots for:
1. Factor cumulative returns over time
2. Factor correlation heatmap
3. 25 Portfolio returns heatmap
4. NYSE vs NASDAQ comparison
5. Factor performance by decade
6. Rolling Sharpe ratios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to use science plots style
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except:
    plt.style.use('seaborn-v0_8-whitegrid')

# Create output directory
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

print("Loading data...")

# Load data
factors_df = pd.read_csv("homework/homework_1/factors_monthly.csv")
factors_df['date_parsed'] = pd.to_datetime(factors_df['date'].astype(str), format='%Y%m%d')
factors_df = factors_df.set_index('date_parsed')

portfolios_df = pd.read_csv("homework/homework_1/portfolios25.csv")
nyse_df = pd.read_csv("homework/homework_2/df_long_NYSE.csv")
nasdaq_df = pd.read_csv("homework/homework_2/df_long_NASDAQ.csv")

print("Generating plots...")

# ============================================================================
# PLOT 1: Cumulative Factor Returns Over Time
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

for factor, label, color in [('mktrf', 'Market', '#1f77b4'),
                              ('smb', 'SMB (Size)', '#ff7f0e'),
                              ('hml', 'HML (Value)', '#2ca02c'),
                              ('umd', 'UMD (Momentum)', '#d62728')]:
    cumulative = (1 + factors_df[factor].fillna(0)).cumprod()
    ax.plot(cumulative.index, cumulative.values, label=label, color=color, linewidth=1.5)

ax.set_yscale('log')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative Return (log scale)', fontsize=12)
ax.set_title('Fama-French Factor Cumulative Returns (1926-2025)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_cumulative_factor_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_cumulative_factor_returns.png")

# ============================================================================
# PLOT 2: Factor Correlation Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

corr_matrix = factors_df[['mktrf', 'smb', 'hml', 'umd']].dropna().corr()
labels = ['Market', 'SMB', 'HML', 'UMD']

im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)

# Add correlation values
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                       ha='center', va='center', fontsize=12,
                       color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black')

ax.set_title('Factor Correlation Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)

plt.tight_layout()
plt.savefig(output_dir / '02_factor_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_factor_correlation_heatmap.png")

# ============================================================================
# PLOT 3: 25 Portfolio Returns Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

vwret_cols = [col for col in portfolios_df.columns if 'vwret' in col.lower()]
avg_returns = portfolios_df[vwret_cols].mean() * 12 * 100  # Annualized %

# Reshape to 5x5 matrix
returns_matrix = avg_returns.values.reshape(5, 5)
size_labels = ['Small', '2', '3', '4', 'Big']
bm_labels = ['Growth', '2', '3', '4', 'Value']

im = ax.imshow(returns_matrix, cmap='RdYlGn', vmin=8, vmax=20)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(bm_labels, fontsize=11)
ax.set_yticklabels(size_labels, fontsize=11)
ax.set_xlabel('Book-to-Market', fontsize=12)
ax.set_ylabel('Size', fontsize=12)

# Add values
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{returns_matrix[i, j]:.1f}%',
                       ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_title('Average Annual Returns: 25 Size/BM Portfolios (1926-2025)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Annual Return (%)', shrink=0.8)

plt.tight_layout()
plt.savefig(output_dir / '03_portfolio_25_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_portfolio_25_heatmap.png")

# ============================================================================
# PLOT 4: NYSE vs NASDAQ Return Distribution
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calculate returns
nyse_df['Date'] = pd.to_datetime(nyse_df['Date'])
nyse_df = nyse_df.sort_values(['Symbol', 'Date'])
nyse_df['Return'] = nyse_df.groupby('Symbol')['Close'].pct_change()

nasdaq_df['Date'] = pd.to_datetime(nasdaq_df['Date'])
nasdaq_df = nasdaq_df.sort_values(['Symbol', 'Date'])
nasdaq_df['Return'] = nasdaq_df.groupby('Symbol')['Close'].pct_change()

# Filter extreme values for better visualization
nyse_returns = nyse_df['Return'].dropna()
nyse_returns = nyse_returns[(nyse_returns > -0.5) & (nyse_returns < 0.5)]

nasdaq_returns = nasdaq_df['Return'].dropna()
nasdaq_returns = nasdaq_returns[(nasdaq_returns > -0.5) & (nasdaq_returns < 0.5)]

# Histogram comparison
axes[0].hist(nyse_returns, bins=100, alpha=0.7, label='NYSE', color='#1f77b4', density=True)
axes[0].hist(nasdaq_returns, bins=100, alpha=0.7, label='NASDAQ', color='#ff7f0e', density=True)
axes[0].set_xlabel('Monthly Return', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Return Distribution: NYSE vs NASDAQ', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_xlim(-0.4, 0.4)

# Box plot
box_data = [nyse_returns.values, nasdaq_returns.values]
bp = axes[1].boxplot(box_data, labels=['NYSE', 'NASDAQ'], patch_artist=True)
bp['boxes'][0].set_facecolor('#1f77b4')
bp['boxes'][1].set_facecolor('#ff7f0e')
axes[1].set_ylabel('Monthly Return', fontsize=12)
axes[1].set_title('Return Dispersion: NYSE vs NASDAQ', fontsize=14, fontweight='bold')
axes[1].set_ylim(-0.3, 0.3)

plt.tight_layout()
plt.savefig(output_dir / '04_nyse_nasdaq_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_nyse_nasdaq_comparison.png")

# ============================================================================
# PLOT 5: Factor Performance by Decade
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

factors_df['decade'] = (factors_df.index.year // 10) * 10
decade_means = factors_df.groupby('decade')[['mktrf', 'smb', 'hml', 'umd']].mean() * 12 * 100

x = np.arange(len(decade_means))
width = 0.2

bars1 = ax.bar(x - 1.5*width, decade_means['mktrf'], width, label='Market', color='#1f77b4')
bars2 = ax.bar(x - 0.5*width, decade_means['smb'], width, label='SMB', color='#ff7f0e')
bars3 = ax.bar(x + 0.5*width, decade_means['hml'], width, label='HML', color='#2ca02c')
bars4 = ax.bar(x + 1.5*width, decade_means['umd'], width, label='UMD', color='#d62728')

ax.set_xlabel('Decade', fontsize=12)
ax.set_ylabel('Annualized Return (%)', fontsize=12)
ax.set_title('Factor Performance by Decade', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{int(d)}s" for d in decade_means.index], fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '05_factor_performance_by_decade.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_factor_performance_by_decade.png")

# ============================================================================
# PLOT 6: Rolling 5-Year Factor Returns
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

window = 60  # 5 years monthly

for factor, label, color in [('mktrf', 'Market', '#1f77b4'),
                              ('smb', 'SMB', '#ff7f0e'),
                              ('hml', 'HML', '#2ca02c'),
                              ('umd', 'UMD', '#d62728')]:
    rolling_mean = factors_df[factor].rolling(window=window).mean() * 12 * 100
    ax.plot(rolling_mean.index, rolling_mean.values, label=label, color=color, linewidth=1.2)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Rolling 5-Year Annualized Return (%)', fontsize=12)
ax.set_title('Rolling 5-Year Factor Returns', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '06_rolling_factor_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_rolling_factor_returns.png")

# ============================================================================
# PLOT 7: Annual Market Returns Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 6))

annual_returns = factors_df['mktrf'].resample('Y').sum() * 100
colors = ['#2ca02c' if r > 0 else '#d62728' for r in annual_returns.values]

ax.bar(annual_returns.index.year, annual_returns.values, color=colors, alpha=0.8)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Annual Market Return (%)', fontsize=12)
ax.set_title('Annual Market Excess Returns (1926-2025)', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Rotate x labels for readability
plt.xticks(rotation=45, ha='right')
ax.set_xticks(ax.get_xticks()[::5])  # Show every 5th year

plt.tight_layout()
plt.savefig(output_dir / '07_annual_market_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_annual_market_returns.png")

# ============================================================================
# PLOT 8: Value vs Growth Over Time
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get value and growth portfolio returns - reload to avoid index issues
portfolios_df2 = pd.read_csv("homework/homework_1/portfolios25.csv")
vwret_cols = [col for col in portfolios_df2.columns if 'vwret' in col.lower()]
growth_cols = [col for col in vwret_cols if 'b1' in col]  # Low BM = Growth
value_cols = [col for col in vwret_cols if 'b5' in col]   # High BM = Value

# Use factors_df index since they have the same number of rows
growth_return = portfolios_df2[growth_cols].mean(axis=1)
value_return = portfolios_df2[value_cols].mean(axis=1)

# Assign the factors date index
growth_return.index = factors_df.index
value_return.index = factors_df.index

# Cumulative returns
growth_cum = (1 + growth_return).cumprod()
value_cum = (1 + value_return).cumprod()

ax.plot(growth_cum.index, growth_cum.values, label='Growth (Low B/M)', color='#1f77b4', linewidth=1.5)
ax.plot(value_cum.index, value_cum.values, label='Value (High B/M)', color='#d62728', linewidth=1.5)

ax.set_yscale('log')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative Return (log scale)', fontsize=12)
ax.set_title('Value vs Growth: Long-Term Performance', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '08_value_vs_growth.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08_value_vs_growth.png")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 60)
print("All plots saved to the 'plots/' directory:")
print("=" * 60)
for f in sorted(output_dir.glob('*.png')):
    print(f"  - {f.name}")
print()
print("Done!")
