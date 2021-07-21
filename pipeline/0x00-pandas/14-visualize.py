#!/usr/bin/env python3
"""Visualize"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.pop('Weighted_Price')
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df.set_index('Date', inplace=True)
df['Close'].fillna(method="ffill", inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)',
                                                'Volume_(Currency)']].fillna(0)

print(df)
df = df[df.index.year >= 2017]
df['High'].groupby(pd.Grouper(freq='d')).max()
df['Low'].groupby(pd.Grouper(freq='d')).min()
df['Open'].groupby(pd.Grouper(freq='d')).mean()
df['Close'].groupby(pd.Grouper(freq='d')).mean()
df['Volume_(BTC)'].groupby(pd.Grouper(freq='d')).sum()
df['Volume_(Currency)'].groupby(pd.Grouper(freq='d')).sum()
df = df[::1440]
# df.reset_index('Date', inplace=True)
# idx = df.index.date[::1440]
# print(idx)


df.plot()
plt.show()
