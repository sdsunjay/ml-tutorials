from functools import reduce
import bs4 as bs
# new imports
import datetime as dt
from datetime import date
import os
import pandas as pd
import pandas_datareader.data as web
import time
import matplotlib.pyplot as plt
from matplotlib import style

import pickle
import requests

style.use('ggplot')

def save_sp500_tickers():
    #scrape data
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class' : 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    # print(tickers)

    return tickers

def sleeper():
    #  while True:
    # Get user input
    num = input('How many seconds to wait: ')

    # Try to convert it to a float
    try:
        num = float(num)
    except ValueError:
        print('Please enter in a number.\n')
        # continue

	# Run our time.sleep() command,
	# and show the before and after time
    print('Before: %s' % time.ctime())
    time.sleep(num)
    print('After: %s\n' % time.ctime())

def get_tickers(dir_name, symbols):
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.today().strftime('%Y-%m-%d')
    # end = dt.datetime.now()
    # end = dt.datetime.now(dt.timezone.utc)
    print(start)
    print(end)

    for symbol in symbols:
        fullPath = dir_name + '/{}.csv'.format(symbol)
        if not os.path.exists(fullPath):
            print(fullPath)
            time.sleep(1)
            df = web.DataReader(symbol, 'morningstar', start, end)
            df.to_csv(fullPath)
        else:
            print('Already have ' + fullPath)

def get_data_from_morningstar(reload_sp500=False):
    # store ALL the data locally
    dir_name = 'stock_dfs'
    if(reload_sp500):
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # get the csvs for all the stocks in s&p 500
    # get_tickers(dir_name, tickers)

def compile_date(flag):
    if(flag):
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    else:
        with open("tickers.csv", "r") as f:
            tickers = f.read().split(',')
    main_df = pd.DataFrame()
    list_of_dfs = []

    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), parse_dates = True, index_col = 'Date', header = None, names = ["Symbol","Date","Close","High","Low","Open","Volume"], na_values = ['nan', '0'])

        df1 = df.rename(columns = {'Close':ticker})
        df2 = df1.drop(['Symbol', 'Open', 'High', 'Low', 'Volume'], 1,inplace=False)
        list_of_dfs.append(df2)
        if count % 10 == 0:
            print(count)
    # https://stackoverflow.com/questions/23668427/pandas-joining-multiple-dataframes-on-columns
    #  Combine a list of dataframes, on top of each other
    main_df = reduce(lambda left,right: pd.merge(left,right,on='Date'), list_of_dfs)
    # print(main_df.tail())
    main_df.to_csv('sp500_joined_closes.csv')

def plot_data(df, title="Stock Prices"):
    '''Plot stock prices'''
    df.set_index(['Date'],inplace=True)
    df = df.astype(float)
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # get_data_from_morningstar()
    compile_date(False)
    # visualize_data()
