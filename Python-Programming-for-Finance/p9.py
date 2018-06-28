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
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        # price in two days from now - old
        #  calculate percent change for the future
        # print(df[ticker])
        df['{}_{}d'.format(ticker, i)] = (float(df[ticker].shift(-i)) -
                float(df[ticker])) / float(df[ticker])
    df.fillna(0, inplace=True)
    return tickers, df

process_data_for_labels('TSLA')


