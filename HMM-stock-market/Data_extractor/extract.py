import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
import csv
import os

ts = "MARUTI.NS TCS.NS ITC.NS"
tickers = ts.split(" ")

t = yf.Tickers(ts)

m_hist = t.tickers["MARUTI.NS"].history(period = "2y")
t_hist = t.tickers["TCS.NS"].history(period = "2y")
i_hist = t.tickers["ITC.NS"].history(period = "2y")

#print(m_hist.head())
#print(t_hist.head())
#print(i_hist.head())

mdf = pd.DataFrame(m_hist)
tdf = pd.DataFrame(t_hist)
idf = pd.DataFrame(i_hist)

print(mdf.head())
#Momentum calc
# Momentum(M) = Closing Price - closing price yesterday

def momentum(stocks, period = 1):
    for i in stocks:
        i['Momentum'] = i['Close'].diff(periods = period)
        i['sm_mom'] = zscore(i['Momentum'], nan_policy='omit')
    return stocks

def ATR(stocks, period = 14):
    for i in stocks:
        i['Prev_Close'] = i['Close'].shift(1)
        tr1 = i['High'] - i['Low']
        tr2 = (i["High"] - i["Prev_Close"]).abs()
        tr3 = (i["Low"] - i["Prev_Close"]).abs()
        i['TR'] = pd.concat([tr1,tr2,tr3], axis=1).max(axis = 1)
        i['ATR'] = i['TR'].rolling(window = period).mean()
    return stocks

stocks = [mdf, tdf, idf]

momentum(stocks)
ATR(stocks)
print("checking")
print(mdf.head(n=15))

sn = dict(zip(tickers, stocks))
print(sn)

def data_writer(stocks_map):
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    for symbol, i in stocks_map.items():
        filepath = os.path.join(script_dir, f"{symbol}.csv") 
        i.to_csv(filepath, encoding="utf-8")  # Keeping index for Date
        print(f"Saved {filepath}")

data_writer(sn)
#onto matlabs now