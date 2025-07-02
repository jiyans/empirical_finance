# a1_download_prices.py 6/20/25
# Downloads prices of NYSE and NASDAQ stocks.
# -Downloads data from Yahoo! Finance.
# -Done in 2 mins on UA fuji.
# -To run on fuji:
#   & G:\shared\python\venv\ml\Scripts\activate.ps1  # Go on venv ml
#   ipython
#   exec(open('a1_download_prices.py').read())
#
# O/P (under temp/): df_long_NYSE.csv, df_long_NASDAQ.csv
#
# a1_download_prices.py
# 6/20/25-6/19/25 Initial version.

import yfinance as yf
import pandas as pd
import datetime, requests
from io import StringIO
from pathlib import Path
Path("temp").mkdir(parents=True, exist_ok=True) # Create O/P directory silently
staDateTime = datetime.datetime.now()
print(f"Started {staDateTime}, reading data.\n")


############################# Get Symbols #############################
def getSymbols(venue):
# Downloads ticker symbols from www.nasdaqtrader.com.
# I/P:
# venue: 'NYSE' uses NYSE stocks, anything else (e.g., 'NASDAQ') for NASDAQ

  # Get ticker symbols
  if venue=='NYSE':
    url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"  # NYSE and other exchanges
  else:
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"  # NASDAQ

  response = requests.get(url)
  response.raise_for_status()

  # Remove the footer line(s) starting with "File Creation Time:"
  lines = response.text.splitlines()
  lines = [line for line in lines if not line.startswith("File Creation Time")]
  data_str = "\n".join(lines)

  # Read into DataFrame (pipe-separated)
  df = pd.read_csv(StringIO(data_str), sep='|')
  print(df.head())

  # Select common stocks, non-test issues
  # -See https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
  if venue=='NYSE':
    Symbols = df[(df['ETF']=='N') & (df['Test Issue']=='N')]['ACT Symbol']  # 3160 non-test-issue stocks on NYSE 6/20/25
    #df.shape # 6549 non-NASDAQ (other) stocks and ETFs incl. NYSE as of 6/20/25
    #df[df['ACT Symbol']=='IBM']  # IBM
  else:
    Symbols = df[(df['Market Category']=='Q') & (df['ETF']=='N') & (df['Test Issue']=='N')]['Symbol']  # 1502 stocks on NASDAQ Global Select 6/20/25
    #df[df['Symbol']=='MSFT']  # MSFT

  return df, Symbols

(df_NYSE  , Symbols_NYSE)   = getSymbols('NYSE')
(df_NASDAQ, Symbols_NASDAQ) = getSymbols('NASDAQ')

# Sanity check:
#Symbols_NYSE.is_unique # True, so symbols in Symbols_NYSE are unique
#Symbols_NASDAQ.is_unique # True, so symbols in Symbols_NASDAQ are unique
#set(Symbols_NYSE) & set(Symbols_NASDAQ) # Empty, so there is no overlap b/w Symbols_NYSE and Symbols_NASDAQ


############################# Download prices #############################
def downloadPrices(Symbols, venue):
# Download monthly prices

  df2 = yf.download(Symbols.values.tolist(), period='max', interval='1mo', group_by='Symbol')
  df2_close = df2.xs('Close', axis=1, level=1)  # Extract closing prices

  # Convert df2_close to long format, sort by Symbol then Date
  df_long = df2_close.stack().reset_index()
  df_long.columns = ['Date', 'Symbol', 'Close']
  df_long = df_long.sort_values(['Symbol', 'Date']).reset_index(drop=True)

  df_long.to_csv(f"temp/df_long_{venue}.csv", index=False)

  return df2, df_long

(df2_NYSE  , df_long_NYSE  ) = downloadPrices(Symbols_NYSE  , 'NYSE')
(df2_NASDAQ, df_long_NASDAQ) = downloadPrices(Symbols_NASDAQ, 'NASDAQ')

# df_long_NYSE.isna().sum()  # All 0, no value of any variable is missing
# df_long_NASDAQ.isna().sum()  # Ditto

print(f"\nNote (a1_download_prices) a1_download_prices.py started {staDateTime}, finished {datetime.datetime.now()}, took {(datetime.datetime.now() - staDateTime).total_seconds()/60} minutes.")
