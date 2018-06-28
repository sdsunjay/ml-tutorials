from collections import Counter
import numpy as np
import pandas as pd
import pickle

# groups of companies are likely to move together
# someone is going to be a first mover and someone is going to be a lager
# if price went up by 2% - Buy
# if price down by 2% - Sell
# if price did NOT go up by 2% and did NOT go down by 2% - Hold

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col='Date')
    df.drop(df.index[0], inplace=True)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    # applymap has also been suggest instead of .astype(float)
    for i in range(1, hm_days+1):
        # price in i days from now - old
        #  calculate percent change for the future
        # pct_change = (df[ticker].shift(-i).astype(float) - df[ticker].astype(float))/ df[ticker].astype(float)
        # df['{}_{}f'.format(ticker, i)] = pct_change
        # shameless stolen from
        # https://stackoverflow.com/questions/48743556/python-pandas-percent-change-with-columns-of-dataframe
        temp = (df[ticker].astype(float) / df[ticker].shift(i).astype(float) - 1)
        df['{}_{}d'.format(ticker, i)] = temp
    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02 # 2%
    for c in cols:
        # print('Col: ' + str(c))
        if c > requirement:
            return 1 # Buy
        if c < -requirement:
            return -1 # Sell
    return 0 # Hold

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    # print(df[ticker].head(7))
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
      df['{}_1d'.format(ticker)],
      df['{}_2d'.format(ticker)],
      df['{}_3d'.format(ticker)],
      df['{}_4d'.format(ticker)],
      df['{}_5d'.format(ticker)],
      df['{}_6d'.format(ticker)],
      df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values
    # print('Vals: ' + vals)
    str_vals = [str(i) for i in vals]
    print('Data Spread: ', Counter(str_vals))
    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan )
    df.dropna(inplace=True)
    # tickers = ['AAPL', 'TSLA','BABA']
    for tick in tickers:
        # print(df[ticker])
        df_vals = df[tick].astype(float).pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    # X is the featuresets
    # y is labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    print(df.head(7))

    return X, y, df

if __name__ == '__main__':
    extract_featuresets('BABA')
