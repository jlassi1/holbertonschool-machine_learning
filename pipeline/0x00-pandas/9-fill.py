#!/usr/bin/env python3
"""Fill """
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.pop('Weighted_Price')
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)',
                                                'Volume_(Currency)']].fillna(0)

df['Close'].fillna(method="ffill", inplace=True)
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

print(df.head())
print(df.tail())
