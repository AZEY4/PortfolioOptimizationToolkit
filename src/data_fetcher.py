import yfinance as yf
import pandas as pd

def fetch_data(tickers, start='2022-01-01', end='2025-01-01'):
    # Ensure auto_adjust=True to get adjusted prices
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    
    # If multiple tickers, select 'Close' column (auto-adjusted)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']  # 'Close' now represents auto-adjusted prices
    else:
        data = data['Close']
    
    return data.dropna()

def save_to_csv(data, filename):
    data.to_csv(filename)
