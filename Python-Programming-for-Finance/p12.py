from collections import Counter
import numpy as np
import pandas as pd
import pickle

# new import
from sklearn.model_selection import train_test_split
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import  VotingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
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
        # shameless stolen from
        # https://stackoverflow.com/questions/48743556/python-pandas-percent-change-with-columns-of-dataframe
        temp = (df[ticker].astype(float) / df[ticker].shift(i).astype(float) - 1)
        df['{}_{}d'.format(ticker, i)] = temp
        days = 10 + i
        df['{}_{}MA'.format(ticker, days)] = df[ticker].rolling(days).mean()
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

def do_ml(ticker):

    X, y, df = extract_featuresets(ticker)

    # Add noisy features
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.25, random_state=random_state)
    # clf = neighbors.KNeighborsClassifier()

    clf = VotingClassifier([('lsvc', svm.SVC(kernel='linear', C=1 )), ('knn',
        neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier()) ])

	# Run classifier
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	# X_train - all the companies
    # y_train - our labels (0,1,-1)
    # X_train = X_train.reshape(-1,1)
   #  y_train = y_train.reshape(-1,1)
    # clf.fit(X_train, y_train)
    # confidence = clf.score(X_test, y_test)
    # print('Accuracy: ' + confidence)
    # you would pickle the classifier if you don't want to train again
    # predictions = clf.predict(X_test)
    # print('Predicted spread: ', Counter(predictions))
    # return confidence
    return 0
if __name__ == '__main__':
    # extract_featuresets('BABA')
    do_ml('BABA')
