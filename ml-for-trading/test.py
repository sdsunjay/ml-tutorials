import glob
import pandas as pd
import argparse
import time
import os

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

def read_file(fullpath):

# AAAP,20170302,37.85,38.27,36.5,37.52,63500
# AAL,20170302,46.67,46.74,45.64,45.72,7590600
# AAME,20170302,3.75,3.75,3.7,3.75,4500
# AAOI,20170302,46.97,49.25,46.69,49.23,1533600

    df = pd.read_csv(fullpath, index_col=None, header=None)

    df.columns = ["symbol", "date", "open", "high", "low", "closing",
    "volume"]
    df2 = df.set_index("symbol", drop = False)
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



def read_files(db, files, datetime, root, database_flag, output_flag, verbose_flag):
    frame = pd.DataFrame()
    list_ = []
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
        df = pd.read_csv(fullpath, index_col=None, header=0)
        # list_.append(df)
    # frame = pd.concat(list_, sort=False)
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
    parser.add_argument('-o', "--output", help='output to stdout', action="store_true")
    parser.add_argument('filenames', nargs='*', type=str, help='files to parse')
    args = parser.parse_args()
    db = None
    datetime = time.strftime('%Y-%m-%d %H:%M:%S')
    if args.database:
        print('Database feature is not implemented yet')
		# db = Database()
        # Sometimes throws warnings when the tables are already created
        # db.create()  # create all the tables
    elif args.directory:
        for root, dirs, files in os.walk(args.directory[0]):
            read_files(db, files, datetime, root, args.database, args.output, args.verbose)
    elif args.text_file:
        for filename in args.text_file:
            if filename.endswith('.txt'):
                read_file(filename)
                         # read_from_txt_file(filename, db, args.output)
            #else:
            #    break_up(db, args.filenames, datetime, './', args.database, args.output, args.verbose)
    else:
        print("Usage: python test.py [-v] [-d directory_name] [-db] [-t filename.txt] [-o]")
