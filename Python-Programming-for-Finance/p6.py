
import bs4 as bs
# new imports
import datetime as dt
from datetime import date
import os
import pandas as pd
import pandas_datareader.data as web
import time

import pickle
import requests

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
    get_tickers(dir_name, tickers)
    final_end = dt.datetime.now(dt.timezone.utc)
    print(final_end)

if __name__ == '__main__':
    get_data_from_morningstar()
