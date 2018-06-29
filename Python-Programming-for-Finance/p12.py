from collections import Counter
import numpy as np
import pandas as pd
import pickle

# new import
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import  VotingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

import warnings
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
    requirement = 0.025 # 2.5%
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
        df_vals = df[tick].astype(float).pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    # X is the featuresets
    # y is labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    # print(df.head(7))

    return X, y, df

def plot_precision_recall():

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test,
            y_score, average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})' ''.format(average_precision["micro"]))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label='Precision-recall curve of class {0} (area = {1:0.2f})'''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def do_ml(ticker):

    X, y, df = extract_featuresets(ticker)

    # Add noisy features
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    X = X.reshape(-1, 1)
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.25, random_state=random_state)
    # clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.SVC(kernel='linear', C=1 )), ('knn',
        neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier()) ])

	# Run classifier
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    # print('Y score: ' + str(y_score))
	# X_train - all the companies
    # y_train - our labels (0,1,-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy: ' + str(confidence))
    # you would pickle the classifier if you don't want to train again
    predictions = clf.predict(X_test)
    print('Predicted spread: ', Counter(predictions))
    return confidence

if __name__ == '__main__':
    # extract_featuresets('BABA')
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    do_ml('BABA')
