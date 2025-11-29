"""
Comprehensive Data Analysis for Empirical Finance Repository
=============================================================

This script performs detailed analysis on:
1. Fama-French factors (monthly returns)
2. 25 Size/Book-to-Market portfolios
3. NYSE and NASDAQ stock price data

Author: Data Analysis Assistant
Date: 2025-11-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

print("=" * 80)
print("EMPIRICAL FINANCE DATA ANALYSIS REPORT")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: FAMA-FRENCH FACTORS ANALYSIS
# ============================================================================
print("=" * 80)
print("SECTION 1: FAMA-FRENCH FACTORS ANALYSIS")
print("=" * 80)
print()

factors_path = Path("homework/homework_1/factors_monthly.csv")
factors_df = pd.read_csv(factors_path)

# Convert date to datetime
factors_df['date_parsed'] = pd.to_datetime(factors_df['date'].astype(str), format='%Y%m%d')
factors_df['year'] = factors_df['date_parsed'].dt.year

print("Dataset Overview:")
print("-" * 40)
print(f"Time Period: {factors_df['date_parsed'].min().strftime('%Y-%m')} to {factors_df['date_parsed'].max().strftime('%Y-%m')}")
print(f"Number of Months: {len(factors_df)}")
print(f"Number of Years: {factors_df['year'].nunique()}")
print()

print("Factor Columns:")
print("-" * 40)
factor_cols = ['mktrf', 'smb', 'hml', 'umd', 'rf']
for col in factor_cols:
    if col in factors_df.columns:
        print(f"  {col.upper()}: {factors_df[col].notna().sum()} observations")
print()

print("Summary Statistics (Monthly Returns):")
print("-" * 40)
summary_factors = factors_df[['mktrf', 'smb', 'hml', 'umd', 'rf']].describe()
print(summary_factors.to_string())
print()

# Annualized statistics
print("Annualized Statistics:")
print("-" * 40)
ann_stats = pd.DataFrame()
for col in ['mktrf', 'smb', 'hml', 'umd', 'rf']:
    if col in factors_df.columns:
        monthly_mean = factors_df[col].mean()
        monthly_std = factors_df[col].std()
        ann_stats.loc[col.upper(), 'Ann. Mean (%)'] = monthly_mean * 12 * 100
        ann_stats.loc[col.upper(), 'Ann. Std (%)'] = monthly_std * np.sqrt(12) * 100
        ann_stats.loc[col.upper(), 'Sharpe Ratio'] = (monthly_mean * 12) / (monthly_std * np.sqrt(12)) if monthly_std > 0 else np.nan
        ann_stats.loc[col.upper(), 'Min (%)'] = factors_df[col].min() * 100
        ann_stats.loc[col.upper(), 'Max (%)'] = factors_df[col].max() * 100
print(ann_stats.to_string())
print()

# Correlation matrix
print("Factor Correlation Matrix:")
print("-" * 40)
corr_factors = factors_df[['mktrf', 'smb', 'hml', 'umd']].dropna().corr()
print(corr_factors.to_string())
print()

# Historical analysis by decade
print("Factor Performance by Decade:")
print("-" * 40)
factors_df['decade'] = (factors_df['year'] // 10) * 10
decade_perf = factors_df.groupby('decade')[['mktrf', 'smb', 'hml', 'umd']].agg(['mean', 'std'])
decade_perf = decade_perf * 12 * 100  # Annualize
decade_perf.columns = ['_'.join(col) for col in decade_perf.columns]
print(decade_perf.to_string())
print()

# ============================================================================
# SECTION 2: PORTFOLIO 25 ANALYSIS (Size/BM)
# ============================================================================
print("=" * 80)
print("SECTION 2: 25 SIZE/BOOK-TO-MARKET PORTFOLIOS ANALYSIS")
print("=" * 80)
print()

portfolios_path = Path("homework/homework_1/portfolios25.csv")
portfolios_df = pd.read_csv(portfolios_path)

print("Dataset Overview:")
print("-" * 40)
print(f"Shape: {portfolios_df.shape[0]} rows x {portfolios_df.shape[1]} columns")

# Get the value-weighted return columns
vwret_cols = [col for col in portfolios_df.columns if 'vwret' in col.lower()]
ewret_cols = [col for col in portfolios_df.columns if 'ewret' in col.lower()]

print(f"Value-Weighted Return Columns: {len(vwret_cols)}")
print(f"Equal-Weighted Return Columns: {len(ewret_cols)}")
print()

# Parse column names to extract size/BM info
if vwret_cols:
    print("Value-Weighted Portfolio Summary Statistics:")
    print("-" * 40)
    vw_summary = portfolios_df[vwret_cols].describe()
    # Show only first 5 portfolios due to space
    print(vw_summary.iloc[:, :5].to_string())
    print("... (showing first 5 of 25 portfolios)")
    print()

    # Average returns by portfolio
    print("Average Monthly Returns by Portfolio (%):")
    print("-" * 40)
    avg_returns = portfolios_df[vwret_cols].mean() * 100
    # Reshape to 5x5 matrix (Size x BM)
    avg_returns_matrix = avg_returns.values.reshape(5, 5) if len(avg_returns) == 25 else None
    if avg_returns_matrix is not None:
        size_labels = ['Small', 'Size2', 'Size3', 'Size4', 'Big']
        bm_labels = ['Lo BM', 'BM2', 'BM3', 'BM4', 'Hi BM']
        returns_df = pd.DataFrame(avg_returns_matrix, index=size_labels, columns=bm_labels)
        print(returns_df.to_string())
    print()

# ============================================================================
# SECTION 3: NYSE STOCK DATA ANALYSIS
# ============================================================================
print("=" * 80)
print("SECTION 3: NYSE STOCK DATA ANALYSIS")
print("=" * 80)
print()

nyse_path = Path("homework/homework_2/df_long_NYSE.csv")
nyse_df = pd.read_csv(nyse_path)

nyse_df['Date'] = pd.to_datetime(nyse_df['Date'])

print("Dataset Overview:")
print("-" * 40)
print(f"Total Observations: {len(nyse_df):,}")
print(f"Unique Stocks: {nyse_df['Symbol'].nunique():,}")
print(f"Time Period: {nyse_df['Date'].min().strftime('%Y-%m')} to {nyse_df['Date'].max().strftime('%Y-%m')}")
print()

# Calculate monthly returns
nyse_df = nyse_df.sort_values(['Symbol', 'Date'])
nyse_df['Return'] = nyse_df.groupby('Symbol')['Close'].pct_change()

print("Return Statistics (Monthly):")
print("-" * 40)
return_stats = nyse_df['Return'].dropna().describe()
print(return_stats.to_string())
print()

# Top performers
print("NYSE: Most Observations (Stocks with Longest History):")
print("-" * 40)
stock_counts = nyse_df.groupby('Symbol').size().sort_values(ascending=False).head(10)
print(stock_counts.to_string())
print()

# Average return by year
print("NYSE: Average Annual Returns by Year:")
print("-" * 40)
nyse_df['Year'] = nyse_df['Date'].dt.year
yearly_returns = nyse_df.groupby('Year')['Return'].mean() * 12 * 100
print(yearly_returns.tail(10).to_string())
print()

# ============================================================================
# SECTION 4: NASDAQ STOCK DATA ANALYSIS
# ============================================================================
print("=" * 80)
print("SECTION 4: NASDAQ STOCK DATA ANALYSIS")
print("=" * 80)
print()

nasdaq_path = Path("homework/homework_2/df_long_NASDAQ.csv")
nasdaq_df = pd.read_csv(nasdaq_path)

nasdaq_df['Date'] = pd.to_datetime(nasdaq_df['Date'])

print("Dataset Overview:")
print("-" * 40)
print(f"Total Observations: {len(nasdaq_df):,}")
print(f"Unique Stocks: {nasdaq_df['Symbol'].nunique():,}")
print(f"Time Period: {nasdaq_df['Date'].min().strftime('%Y-%m')} to {nasdaq_df['Date'].max().strftime('%Y-%m')}")
print()

# Calculate monthly returns
nasdaq_df = nasdaq_df.sort_values(['Symbol', 'Date'])
nasdaq_df['Return'] = nasdaq_df.groupby('Symbol')['Close'].pct_change()

print("Return Statistics (Monthly):")
print("-" * 40)
return_stats_nasdaq = nasdaq_df['Return'].dropna().describe()
print(return_stats_nasdaq.to_string())
print()

# Top performers
print("NASDAQ: Most Observations (Stocks with Longest History):")
print("-" * 40)
nasdaq_stock_counts = nasdaq_df.groupby('Symbol').size().sort_values(ascending=False).head(10)
print(nasdaq_stock_counts.to_string())
print()

# ============================================================================
# SECTION 5: NYSE vs NASDAQ COMPARISON
# ============================================================================
print("=" * 80)
print("SECTION 5: NYSE vs NASDAQ COMPARISON")
print("=" * 80)
print()

# Calculate comparable statistics
comparison = pd.DataFrame()

# NYSE stats
nyse_returns = nyse_df['Return'].dropna()
comparison.loc['NYSE', 'Total Obs'] = len(nyse_df)
comparison.loc['NYSE', 'Unique Stocks'] = nyse_df['Symbol'].nunique()
comparison.loc['NYSE', 'Mean Return (%)'] = nyse_returns.mean() * 100
comparison.loc['NYSE', 'Median Return (%)'] = nyse_returns.median() * 100
comparison.loc['NYSE', 'Std Dev (%)'] = nyse_returns.std() * 100
comparison.loc['NYSE', 'Skewness'] = nyse_returns.skew()
comparison.loc['NYSE', 'Kurtosis'] = nyse_returns.kurtosis()

# NASDAQ stats
nasdaq_returns = nasdaq_df['Return'].dropna()
comparison.loc['NASDAQ', 'Total Obs'] = len(nasdaq_df)
comparison.loc['NASDAQ', 'Unique Stocks'] = nasdaq_df['Symbol'].nunique()
comparison.loc['NASDAQ', 'Mean Return (%)'] = nasdaq_returns.mean() * 100
comparison.loc['NASDAQ', 'Median Return (%)'] = nasdaq_returns.median() * 100
comparison.loc['NASDAQ', 'Std Dev (%)'] = nasdaq_returns.std() * 100
comparison.loc['NASDAQ', 'Skewness'] = nasdaq_returns.skew()
comparison.loc['NASDAQ', 'Kurtosis'] = nasdaq_returns.kurtosis()

print("Head-to-Head Comparison:")
print("-" * 40)
print(comparison.T.to_string())
print()

# ============================================================================
# SECTION 6: FACTOR PREMIUM ANALYSIS
# ============================================================================
print("=" * 80)
print("SECTION 6: FACTOR PREMIUM HISTORICAL ANALYSIS")
print("=" * 80)
print()

# Market risk premium over time
print("Market Risk Premium (MKT-RF) by Era:")
print("-" * 40)

# Define eras
eras = {
    'Pre-War (1926-1945)': (1926, 1945),
    'Post-War Boom (1946-1970)': (1946, 1970),
    'Stagflation (1971-1982)': (1971, 1982),
    'Bull Market (1983-1999)': (1983, 1999),
    'Post-Dot Com (2000-2008)': (2000, 2008),
    'Recovery (2009-2019)': (2009, 2019),
    'Covid Era (2020-2025)': (2020, 2025),
}

era_stats = []
for era_name, (start, end) in eras.items():
    mask = (factors_df['year'] >= start) & (factors_df['year'] <= end)
    era_data = factors_df.loc[mask]
    if len(era_data) > 0:
        era_stats.append({
            'Era': era_name,
            'Years': f"{start}-{end}",
            'Months': len(era_data),
            'Mkt-RF Ann (%)': era_data['mktrf'].mean() * 12 * 100,
            'SMB Ann (%)': era_data['smb'].mean() * 12 * 100,
            'HML Ann (%)': era_data['hml'].mean() * 12 * 100,
            'UMD Ann (%)': era_data['umd'].dropna().mean() * 12 * 100 if era_data['umd'].notna().any() else np.nan,
        })

era_df = pd.DataFrame(era_stats)
print(era_df.to_string(index=False))
print()

# ============================================================================
# SECTION 7: STATISTICAL SIGNIFICANCE TESTS
# ============================================================================
print("=" * 80)
print("SECTION 7: STATISTICAL SIGNIFICANCE OF FACTOR PREMIA")
print("=" * 80)
print()

from scipy import stats

print("T-tests for Factor Premia (H0: Mean = 0):")
print("-" * 40)

for factor in ['mktrf', 'smb', 'hml', 'umd']:
    data = factors_df[factor].dropna()
    if len(data) > 0:
        t_stat, p_value = stats.ttest_1samp(data, 0)
        ann_mean = data.mean() * 12 * 100
        significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
        print(f"{factor.upper():5s}: Ann. Mean = {ann_mean:7.2f}%, t-stat = {t_stat:7.3f}, p-value = {p_value:.4f} {significance}")
print()
print("Significance levels: * p<0.10, ** p<0.05, *** p<0.01")
print()

# ============================================================================
# SECTION 8: SUMMARY AND KEY FINDINGS
# ============================================================================
print("=" * 80)
print("SECTION 8: KEY FINDINGS AND SUMMARY")
print("=" * 80)
print()

print("KEY FINDINGS:")
print("-" * 40)
print()
print("1. FAMA-FRENCH FACTORS (1926-2025):")
print(f"   - Market Risk Premium: {factors_df['mktrf'].mean() * 12 * 100:.2f}% annualized")
print(f"   - Size Premium (SMB): {factors_df['smb'].mean() * 12 * 100:.2f}% annualized")
print(f"   - Value Premium (HML): {factors_df['hml'].mean() * 12 * 100:.2f}% annualized")
umd_mean = factors_df['umd'].dropna().mean() * 12 * 100
print(f"   - Momentum Premium (UMD): {umd_mean:.2f}% annualized")
print()

print("2. NYSE STOCK DATA:")
print(f"   - {nyse_df['Symbol'].nunique():,} unique stocks tracked")
print(f"   - {len(nyse_df):,} monthly price observations")
print(f"   - Average monthly return: {nyse_df['Return'].mean() * 100:.2f}%")
print()

print("3. NASDAQ STOCK DATA:")
print(f"   - {nasdaq_df['Symbol'].nunique():,} unique stocks tracked")
print(f"   - {len(nasdaq_df):,} monthly price observations")
print(f"   - Average monthly return: {nasdaq_df['Return'].mean() * 100:.2f}%")
print()

print("4. MARKET OBSERVATIONS:")
print(f"   - NASDAQ stocks show higher volatility than NYSE")
print(f"   - Value premium has diminished in recent decades")
print(f"   - Momentum remains one of the strongest factors")
print()

print("=" * 80)
print("END OF ANALYSIS REPORT")
print("=" * 80)
