import glob
import pandas as pd
import argparse
import time
import os

import datetime
import pandas_datareader.data as web
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
def get_max_close(symbol):
    """Return the maximum closing value for the stock indicated by the symbol.

    Note: Data for all stocks in Nasdaq is stored in file: data/<date>.txt
    """
    return df['Close'].max() # compute and return max

def print_head_tail(df):
    print('Tail: ')
    print(df.tail())
    print('\nhead: ')
    print(df.head())
    #print(df[10:100]) # rows between index 10 and 20

def read_from_web(symbol):
    style.use('fivethirtyeight')

    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()

    df = web.DataReader(symbol, "morningstar", start, end)

    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    df = df.drop("Symbol", axis=1)

    print(df.head())

    df['High'].plot()
    plt.legend()
    plt.show()

def read_file(fullpath):


    df = pd.read_csv(fullpath, index_col=None, header=None)

    df.columns = ["symbol", "date", "open", "high", "low", "closing","volume"]
    df2 = df.set_index("symbol", drop = False)
    # [2832 rows x 7 columns]
    # print(df2)
    print(df2.loc["AAPL", : ])

    # print(df2['close'].max())
    # print(df2.iloc[0:3,0:4])
    #df2.loc["Alaska":"Arkansas","2005":"2007"]
    # ZNGA
    # print_head_tail(df)
	#for symbol in ['AAPL', 'WDAY']:
    #    print('Max close for year')
    #    print symbol, get_max_close(symbol)

def plot_data(df, title="Stock Prices"):
    '''Plot stock prices'''
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    plt.legend()
    plt.show()


def read_files(db, files, datetime, root, database_flag, output_flag, verbose_flag):
    combined_df = pd.DataFrame()
    list_of_dfs = []
    for fle in files:
        fullpath = os.path.join(root, fle)
        if not fle.endswith('.txt'):
            if output_flag:
                print('Skipping ' + fle)
            continue
        if verbose_flag and output_flag:
            print('Path: ' + fullpath)
        if database_flag:
            print('Database feature not yet implemented')
			# store_in_db(db, fullpath, datetime, output_flag)
            # del db
        # if output_flag and verbose_flag:
        # read_file(fullpath)
        df = pd.read_csv(fullpath, parse_dates = True, index_col = "date", header = None, names =
                ["symbol", "date", "open", "high", "low", "closing", "volume"],
                na_values = ['nan', '0'])

        # df.columns = ['symbol', 'date', 'open', 'high', 'low', 'closing', 'volume']
        # df2 = df.set_index("date", drop = False)
        df = df[df.open.notnull()]
        df = df[df.high.notnull()]
        df = df[df.low.notnull()]
        # try to drop places where volume is 0 or null
        # not working
        df = df[df.volume.notnull()]
        # Use `iloc[]` to select row `0`
        #print(df2.iloc[0])

        # Use `loc[]` to select column `'A'`
        # print(df2.loc["AAPL", :])
       # fillna is great. It will take every NaN and replace it with some data.
       # For example, let’s say you had a bunch of null values and you wanted to replace them with the word “Unknown”
       # Use inplace=True to save it back to the dataframe
       # df['height'].fillna("Unknown", inplace=True)
        list_of_dfs.append(df)
    # frame = pd.concat(list_, sort=False, ignore_index=True)
    # Combine a list of dataframes, on top of each other
    combined_df = pd.concat(list_of_dfs)
    plot_data(combined_df['high'].loc[combined_df['symbol'] == 'AAPL'])
    # print(combined_df.loc[combined_df['symbol'] == 'AAPL'])
    # print(combined_df)
    # [0].loc["AAPL", : ])
    # Find the rows where age isn't null
	# And save them into your new dataframe
    # print(frame)
        # output_to_screen(tree)

# Takes in a given xml file and returns the tree of failures in a nice
# printed format
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sunjays stock project')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-d", "--directory", type=str, nargs=1, help="read all files in a directory")
    parser.add_argument("-db", "--database", help="utilize the database for storing failures", action="store_true")
    parser.add_argument('-t', "--text_file", nargs='*', type=str, help='txt files to parse for raw errors')
    parser.add_argument('-w', "--web", nargs=1, type=str, help='read from web')
    parser.add_argument('-o', "--output", help='output to stdout', action="store_true")
    parser.add_argument('filenames', nargs='*', type=str, help='files to parse')
    args = parser.parse_args()
    db = None
    # datetime = time.strftime('%Y-%m-%d %H:%M:%S')
    if args.database:
        print('Database feature is not implemented yet')
		# db = Database()
        # Sometimes throws warnings when the tables are already created
        # db.create()  # create all the tables
    elif args.web:
        read_from_web(args.web[0])
    elif args.directory:
        for root, dirs, files in os.walk(args.directory[0]):
            read_files(db, files, False, root, args.database, args.output, args.verbose)
    elif args.text_file:
        for filename in args.text_file:
            if filename.endswith('.txt'):
                read_file(filename)
                         # read_from_txt_file(filename, db, args.output)
            #else:
            #    break_up(db, args.filenames, datetime, './', args.database, args.output, args.verbose)
    else:
        print("Usage: python test.py [-v] [-d directory_name] [-db] [-t filename.txt] [-o]")
