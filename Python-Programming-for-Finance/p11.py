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
    # if(df[ticker].iloc[0].isdigit()):
    tickers = df.columns.values.tolist()
    # print(df[ticker].iloc[0])
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        # price in two days from now - old
        #  calculate percent change for the future
        temp = float(df[ticker].shift(-i).iloc[0]) - float(df[ticker].iloc[0])
        # print(temp)
        temp_again = float(temp) / float(df[ticker].iloc[0])
        print(temp_again)
        df['{}_{}d'.format(ticker, i)] = float(temp_again)
    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02 # 2%
    for col in cols:
        if col > requirement:
            return 1 # Buy
        if col < -requirement:
            return -1 # Sell
    return 0 # Hold

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    # print(df)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
      df['{}_1d'.format(ticker)],
      df['{}_2d'.format(ticker)],
      df['{}_3d'.format(ticker)],
      df['{}_4d'.format(ticker)],
      df['{}_5d'.format(ticker)],
      df['{}_6d'.format(ticker)],
      df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data Spread: ', Counter(str_vals))
    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan )
    df.dropna(inplace=True)
    print(df.head())
    # percent change from yesterday
    # TODO fix this
    # this currently does not work
    # df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    # X is the featuresets
    # y is labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

if __name__ == '__main__':
    extract_featuresets('TSLA')
