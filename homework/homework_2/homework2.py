#!/usr/bin/env python3
# %%
import pandas as pd
import altair as alt
from scipy import stats
alt.renderers.enable("browser")

def load_and_prep(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])
    # Compute monthly returns
    df['ret'] = df.groupby('Symbol')['Close'].pct_change()
    return df

df_nyse = load_and_prep('df_long_NYSE.csv')
df_nasdaq = load_and_prep('df_long_NASDAQ.csv')

# %%
df_nyse


# %%
def get_momentum_portfolios(df):
    # Calculate the (12,1) momentum signal
    df['mom_signal'] = df.groupby('Symbol')['Close'].shift(2) / \
                       df.groupby('Symbol')['Close'].shift(13) - 1

    # Remove rows where signal or return is missing
    df_filtered = df.dropna(subset=['ret', 'mom_signal'])

    # For each month, rank stocks into 10 deciles based on the signal
    df_filtered['decile'] = df_filtered.groupby('Date')['mom_signal'] \
        .transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Calculate equally-weighted portfolio returns for each decile
    port_returns = df_filtered.groupby(['Date', 'decile'])['ret'].mean().unstack()

    # Calculate average number of stocks per decile
    # Modified this line to properly compute average stocks per decile
    avg_stocks = df_filtered.groupby('decile').size()

    return port_returns, avg_stocks

# %%

nyse_ports, nyse_avg_stocks = get_momentum_portfolios(df_nyse)
nasdaq_ports, nasdaq_avg_stocks = get_momentum_portfolios(df_nasdaq)

# %%

def analyze_portfolios(port_returns, avg_stocks):
    means = port_returns.mean()
    # Perform a one-sample t-test against 0 for each decile
    ttest_results = stats.ttest_1samp(port_returns.dropna(), 0)

    results = pd.DataFrame({
        'Mean Return': means,
        't-statistic': ttest_results.statistic,
        'p-value': ttest_results.pvalue,
        'Avg Stocks': avg_stocks
    })
    return results


print("--- NYSE Results ---")
analyze_portfolios(nyse_ports, nyse_avg_stocks)
print("\n--- NASDAQ Results ---")
analyze_portfolios(nasdaq_ports, nasdaq_avg_stocks)

# %%
print("--- NYSE Results ---")
res_nyse = analyze_portfolios(nyse_ports, nyse_avg_stocks)

nyse_returns_deciles = alt.Chart(res_nyse.reset_index()).encode(
    y='Mean Return',
    x="decile:O"
).mark_bar()
# %%
nyse_returns_deciles.show()
# %%
print("\n--- NASDAQ Results ---")
res_nasdaq = analyze_portfolios(nasdaq_ports, nasdaq_avg_stocks)

nasdaq_returns_deciles = alt.Chart(res_nasdaq.reset_index()).encode(
    y='Mean Return',
    x="decile:O"
).mark_bar()
nasdaq_returns_deciles.show()

# %%

port_diff = nyse_ports - nasdaq_ports
diff_ttest = stats.ttest_1samp(port_diff.dropna(), 0)

diff_results = pd.DataFrame({
    'Mean Difference (NYSE-NASDAQ)': port_diff.mean(),
    't-statistic': diff_ttest.statistic,
    'p-value': diff_ttest.pvalue
})
print("\n--- NYSE vs NASDAQ Difference ---")
print(diff_results)
