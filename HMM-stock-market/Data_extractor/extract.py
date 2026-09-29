import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

TICKERS = ["MARUTI.NS", "TCS.NS", "ITC.NS"]


def fetch_data(tickers, period="2y"):
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

def add_momentum(stocks_map, length=20):
    for sym, df in stocks_map.items():
        mom_raw = df["Close"].pct_change(1) * 100
        mom_smooth = mom_raw.ewm(span=length, adjust=False).mean()
        df["obs_mom"] = zscore_rolling(mom_smooth, length)
    return stocks_map

def add_atr(stocks_map, length=20):
    for sym, df in stocks_map.items():
        df["Prev_Close"] = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Prev_Close"]).abs()
        tr3 = (df["Low"] - df["Prev_Close"]).abs()
        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        vol_raw = df["TR"].ewm(alpha=1 / length, adjust=False).mean()
        df["obs_vol"] = zscore_rolling(vol_raw, length)
    return stocks_map

def data_writer(stocks_map):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for sym, df in stocks_map.items():
        filepath = os.path.join(script_dir, f"{sym}.csv")
        df.to_csv(filepath, encoding="utf-8")
        print(f"Saved {filepath}")

def clean_na(stocks_map):
    for sym, df in stocks_map.items():
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped} NA rows from {sym}")
        stocks_map[sym] = df
    return stocks_map


if __name__ == "__main__":
    stocks = fetch_data(TICKERS)
    stocks = drop_zero_volume(stocks)
    stocks = add_momentum(stocks)
    stocks = add_atr(stocks)
    stocks = clean_na(stocks)
    for sym, df in stocks.items():
        print(sym)
        print(df.head(15))

    data_writer(stocks)