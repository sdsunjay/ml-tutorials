from collections import Counter
import numpy as np
import pandas as pd
import pickle

# new import
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import  VotingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

import warnings

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             accuracy_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# groups of companies are likely to move together
# someone is going to be a first mover and someone is going to be a lager
# if price went up by 2% - Buy
# if price down by 2% - Sell
# if price did NOT go up by 2% and did NOT go down by 2% - Hold

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col='Date')
    # df = pd.read_csv('sp500_joined_closes1.csv', index_col='Date')
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

def buy_sell(*args):
    cols = [c for c in args]
    requirement = 0.050 # 5.0%
    for c in cols:
        # print('Col: ' + str(c))
        if c > requirement:
            return 1 # Buy
        if c < -requirement:
            return 0 # Sell
    return 0 # Sell

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    # print(df[ticker].head(7))
    df['{}_target'.format(ticker)] = list(map(buy_sell,
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
    # print(df.tail(7))

    return X, y, df

def plot_precision_recall(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    print('Precision: ' + precision)
    print('Recall: ' + recall)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

def plot_precision_recall1(y_test, y_score, n_classes):

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

def do_ml1(ticker):

    X, y, df = extract_featuresets(ticker)
    # Add noisy features
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    n  = X.shape
    # print(n)
    X = X.reshape(-1, 1)

    # Use label_binarize to be multi-label like settings
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
	# Run classifier
    clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    # average_precision = average_precision_score(X, y, average='macro')
    average_precision = average_precision_score(y_test, y_score)
    precision = precision_score(y_test, y_score.round(), average='macro')
    # accuracy = accuracy_score(y_test, y_pred.round(), normalize=False)
    accuracy = accuracy_score(y_test, y_score.round())
   #  accuracy = accuracy_score(y_test, y_pred)
    confidence = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    recall = recall_score(y_test, y_score.round(), average='macro')
    name = 'One Vs Rest Classifier'
    print("%s:" % name)
    print('\t Accuracy: %1.3f' % confidence)
    print("\t Precision (micro): %1.3f" % precision_score(y_test, y_score.round(), average='micro'))
    # print("\t Precision: %1.3f" % precision_score(y_test, y_pred, average='micro'))
    # recall = recall_score(y_test, y_pred, average='micro')
    average_precision = average_precision_score(y_test, y_score)
    # print('\t Average precision score: {0:0.2f}'.format(average_precision))
    # print('\t Recall: %1.3f' % recall_score(y_test, y_score.round(), average="macro"))
    print('\t Precision score (macro): {0:0.2f}'.format(precision))
    # print('\t Accuracy score: {0:0.2f}'.format(accuracy))
    print('\t Recall score: {0:0.2f}\n'.format(recall))
    # plot_precision_recall(y_test, y_score)
    # plot_precision_recall1(y_test, y_score, n_classes)
    return 0

def do_ml(ticker):

    X, y, df = extract_featuresets(ticker)
    # confidence = 0
    # Add noisy features
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    X = X.reshape(-1, 1)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.25, random_state=random_state)

    # classifier
    clf = VotingClassifier([('lsvc', svm.SVC(kernel='linear', C=1, probability=True, random_state=random_state )), ('rbf',
        svm.SVC(kernel='rbf', C=1, gamma=0.10000000000000001, probability=True, random_state=random_state)), ('knn',
         neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())
        ], voting='hard')
	# X_train - all the companies
    # y_train - our labels (0,1,-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Predicted spread: ', Counter(y_pred))
    name = 'Voting Classifier'
    print("%s:" % name)
    # you would pickle the classifier if you don't want to train again
   #  predictions = clf.predict(X_test)
    # predicted2 = clf.predict_proba(X_test)
    confidence = clf.score(X_test, y_test)
    print('\t Accuracy: %1.3f' % confidence)
    # print('\t Precision: %1.3f' % precision_score(y_test, y_pred, average="macro"))
    print('\t Recall: %1.3f' % recall_score(y_test, y_pred, average="macro"))
    print("\t Precision: %1.3f" % precision_score(y_test, y_pred, average='micro'))
    print("\t Recall: %1.3f" % recall_score(y_test, y_pred, average='micro'))
    # print('\t F1: %1.3f\n' % f1_score(y_test, y_pred, average="macro"))
    print("\t F1: %1.3f\n" % f1_score(y_test, y_pred, average='micro'))
    return confidence

def start_plot_calibration_curve(ticker):

    X, y, df = extract_featuresets(ticker)
    # confidence = 0
    # Add noisy features
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    X = X.reshape(-1, 1)

    # y = label_binarize(y, classes=[0, 1, 2])
    # n_classes = y.shape[1]
    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.25, random_state=random_state)

    # Plot calibration curve for Gaussian Naive Bayes
    plot_calibration_curve(GaussianNB(), "Naive Bayes", 1, X_train, X_test,
            y_train, y_test, X, y)

    # Plot calibration curve for Linear SVC
    # TODO - figure out why this does not work
    # plot_calibration_curve(LinearSVC(), "SVC", 2, X_train, X_test, y_train,
    # y_test, X, y)

    # plt.show()


def plot_calibration_curve(est, name, fig_index, X_train, X_test, y_train,
        y_test, X, y):

    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred, average='micro'))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred, average='micro'))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred, average='micro'))

        # TODO - figure out why this does not work
        # fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=1)

        # ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
        #         label="%s (%1.3f)" % (name, clf_score))

        # ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
        #         histtype="step", lw=2)

    # ax1.set_ylabel("Fraction of positives")
    # ax1.set_ylim([-0.05, 1.05])
    # ax1.legend(loc="lower right")
    # ax1.set_title('Calibration plots  (reliability curve)')

    # ax2.set_xlabel("Mean predicted value")
    # ax2.set_ylabel("Count")
    # ax2.legend(loc="upper center", ncol=2)

    # plt.tight_layout()

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    ticker = 'BABA'
    print('Ticker: ' + ticker)
    do_ml(ticker)
    # TODO - figure out what to do with this
    # do_ml1(ticker)
    start_plot_calibration_curve(ticker)
