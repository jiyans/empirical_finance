#!/usr/bin/env python3
# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# %%

def perform_grs_test(df, assets, factors):
    T, N = df.shape[0], len(assets)
    K = len(factors)

    # Time-series regressions to get alphas and residuals
    alphas = []
    residuals = pd.DataFrame(index=df.index)

    X = sm.add_constant(df[factors])
    for asset in assets:
        y = df[asset]
        model = sm.OLS(y, X).fit()
        alphas.append(model.params['const'])
        residuals[asset] = model.resid

    alphas = np.array(alphas)

    # Covariance matrix of residuals
    Sigma = residuals.cov()

    # Mean and covariance of factors
    mu_F = df[factors].mean()
    Omega_F = df[factors].cov()

    # GRS statistic
    grs_stat = (T / N) * ((T - N - K) / (T - K - 1)) * \
               (alphas.T @ np.linalg.inv(Sigma) @ alphas) / \
               (1 + mu_F.T @ np.linalg.inv(Omega_F) @ mu_F)

    p_value = stats.f.sf(grs_stat, N, T - N - K)

    print("1) Gibbons-Ross-Shanken (GRS) Test")
    print(f"   Periods (T): {T}, Assets (N): {N}, Factors (K): {K}")
    print(f"   GRS Statistic: {grs_stat:.4f}")
    print(f"   Distribution: F({N}, {T - N - K})")
    print(f"   p-value: {p_value:.4f}\n")

def perform_fama_macbeth(df, assets, factors, nw_lags=None, shanken_correction=False):
    """Performs the Fama-MacBeth two-pass regression."""
    # Step 1: Time-series regression for each asset to get betas
    betas = pd.DataFrame(index=assets, columns=factors)
    X_ts = sm.add_constant(df[factors])
    for asset in assets:
        y_ts = df[asset]
        # We simply use statsmodels to do the OLS estimation
        model = sm.OLS(y_ts, X_ts).fit()
        betas.loc[asset] = model.params[1:]

    betas = betas.astype(float)

    # Step 2: Cross-sectional regression for each time period
    lambdas = []
    X_cs = sm.add_constant(betas)
    for t in df.index:
        y_cs = df.loc[t, assets]
        model = sm.OLS(y_cs, X_cs, missing='drop').fit()
        lambdas.append(model.params)

    lambdas_df = pd.DataFrame(lambdas, index=df.index)
    lambdas_df.columns = ['Intercept'] + factors

    # Time-series t-tests on the lambda estimates
    results = pd.DataFrame(index=lambdas_df.columns)
    results['Mean Premium'] = lambdas_df.mean()

    title = "2) Fama-MacBeth Procedure"
    if nw_lags is not None:
        title += f" (Newey-West, lags={nw_lags})"
        cov_type = 'HAC'
        cov_kwds = {'maxlags': nw_lags}
    else:
        # Calculate the standard Fama-MacBeth standard error of the mean directly.
        se_fm = lambdas_df.std() / np.sqrt(len(lambdas_df))

        if shanken_correction:
            title += " (Shanken Correction)"
            # Shanken correction factor
            Sigma_F = df[factors].cov()
            mu_F = df[factors].mean()
            c = 1 + mu_F.T @ np.linalg.inv(Sigma_F) @ mu_F
            # Apply correction
            results['Std. Error'] = se_fm * np.sqrt(c)
        else: # Standard Fama-MacBeth
            title += " (Standard)"
            results['Std. Error'] = se_fm

    if nw_lags is not None:
        # For Newey-West, fit each lambda series separately
        for col in lambdas_df.columns:
            model = sm.OLS(lambdas_df[col], np.ones(len(lambdas_df))).fit(cov_type=cov_type, cov_kwds=cov_kwds)
            results.loc[col, 'Std. Error'] = model.bse[0]
            results.loc[col, 't-statistic'] = model.tvalues[0]
            results.loc[col, 'p-value'] = model.pvalues[0]

    if not nw_lags: # Calculate t-stat and p-value for standard and Shanken
        results['t-statistic'] = results['Mean Premium'] / results['Std. Error']
        results['p-value'] = 2 * (1 - stats.t.cdf(np.abs(results['t-statistic']), df=len(df)-1))

    results['Num Obs'] = len(df)

    print(title)
    print(results.to_string(formatters={
        'Mean Premium': '{:.4%}'.format,
        'Std. Error': '{:.4%}'.format,
        't-statistic': '{:.2f}'.format,
        'p-value': '{:.4f}'.format,
        'Num Obs': '{:d}'.format
    }))

    # Comparison with mean factors
    mean_factors = df[factors].mean()
    print("\n   Comparison: Mean Premia vs. Mean Factors")
    for factor in factors:
        print(f"   - {factor}: Premium={results.loc[factor, 'Mean Premium']:.4%}, Factor Mean={mean_factors[factor]:.4%}")
    print("-" * 60)
    print()


# %%
port_file, factor_file = 'portfolios25.csv', 'factors_monthly.csv'

p25 = pd.read_csv(port_file)
p25['Date'] = pd.to_datetime(p25['DATE'], format='%Y%m%d')

# Load Fama-French 3 factors
factors = pd.read_csv(factor_file)
factors['Date'] = pd.to_datetime(factors['dateff'], format='%Y%m%d')

# %%


# Merge datasets
df = pd.merge(factors, p25.drop(columns=['year', 'month']), on='Date', how='inner')
# %%

df = df.set_index('Date')
# %%


# Portfolio returns are in percent, convert to decimal
asset_cols = [col for col in df.columns if col.endswith('_vwret')]
df[asset_cols] = df[asset_cols] / 100

# Factor returns are already in percent, convert to decimal
factor_cols = ['mktrf', 'smb', 'hml', 'rf']
df[factor_cols] = df[factor_cols] / 100

# Calculate portfolio excess returns
excess_returns = df[asset_cols].subtract(df['rf'], axis=0)

# Combine into a single dataframe
data = pd.concat([excess_returns, df[['mktrf', 'smb', 'hml']]], axis=1)

# %%


# Combine into a single dataframe
data = pd.concat([excess_returns, df[['mktrf', 'smb', 'hml']]], axis=1)
data = data.loc[data.index.notna()]

data.sort_index(inplace=True)

assets = asset_cols
# %%
# Model and period declaration in a dictionary for easy printing and loping
periods = {
    "1963-07 to 1991-12": ('1963-07-01', '1991-12-31'),
    "1927-01 to 2024-12": ('1927-01-01', '2024-12-31')
}

models = {
    "CAPM": ['mktrf'],
    "Fama-French 3-Factor": ['mktrf', 'smb', 'hml']
}
# %%

# For each period
for period_name, (start, end) in periods.items():
    period_data = data.loc[start:end].copy()

    # Get number of time steps
    T = len(period_data)
    # Newey lags
    nw_lags = int(np.floor(4 * (T / 100)**(2/9)))

    # For each of the two models do the four procedures
    for model_name, factors in models.items():
        print(f"--- {model_name} Results for {period_name} ---")

        # 1) GRS Test
        perform_grs_test(period_data, assets, factors)

        # 2) Standard Fama-MacBeth
        perform_fama_macbeth(period_data, assets, factors)

        # 3) FM with Newey-West
        perform_fama_macbeth(period_data, assets, factors, nw_lags=nw_lags)

        # 4) FM with Shanken Correction
        perform_fama_macbeth(period_data, assets, factors, shanken_correction=True)

# %%
#
