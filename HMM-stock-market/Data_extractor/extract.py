import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

TICKERS = ["MARUTI.NS", "TCS.NS", "ITC.NS"]


def fetch_data(tickers, period="5y"):
    t = yf.Tickers(" ".join(tickers))
    return {sym: t.tickers[sym].history(period=period) for sym in tickers}


def drop_zero_volume(stocks_map):
    for sym, df in stocks_map.items():
        n_before = len(df)
        df = df[df["Volume"] != 0].reset_index(drop=False)
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped} zero-volume rows from {sym}")
        stocks_map[sym] = df
    return stocks_map

def zscore_rolling(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def add_momentum(stocks_map, length=40):
    for sym, df in stocks_map.items():
        ret = np.log(df["Close"]).diff()
        drift = ret.ewm(span=length, adjust=False).mean()
        rvol = ret.ewm(span=length, adjust=False).std()
        df["obs_mom"] = drift / rvol * np.sqrt(length)
    return stocks_map

def add_atr(stocks_map, length=20):
    for sym, df in stocks_map.items():
        df["Prev_Close"] = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Prev_Close"]).abs()
        tr3 = (df["Low"] - df["Prev_Close"]).abs()
        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_pct = df["TR"].ewm(alpha=1 / length, adjust=False).mean() / df["Close"]
        df["obs_vol"] = zscore_rolling(np.log(atr_pct), 250)
    return stocks_map

def data_writer(stocks_map):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for sym, df in stocks_map.items():
        filepath = os.path.join(script_dir, f"{sym}.csv")
        df.to_csv(filepath, encoding="utf-8", index=False)
        print(f"Saved {filepath}")

def clean_na(stocks_map):
    for sym, df in stocks_map.items():
        n_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped} NA rows from {sym}")
        stocks_map[sym] = df
    return stocks_map

def add_er(stocks_map, length=20):
    for sym, df in stocks_map.items():
        net = (df["Close"] - df["Close"].shift(length)).abs()
        path = df["Close"].diff().abs().rolling(length).sum()
        df["obs_er"] = net / path
    return stocks_map

if __name__ == "__main__":
    stocks = fetch_data(TICKERS)
    stocks = drop_zero_volume(stocks)
    stocks = add_momentum(stocks)
    stocks = add_atr(stocks)
    stocks = add_er(stocks)
    stocks = clean_na(stocks)
    for sym, df in stocks.items():
        print(sym)
        print(df.head(15))

    data_writer(stocks)

