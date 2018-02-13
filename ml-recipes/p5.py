# Writing Our First Classifier - Machine Learning Recipes #5

# Welcome back! It's time to write our first classifier. This is a milestone if you’re new to
# machine learning. We'll start with our code from episode #4 and comment out the classifier we
# imported. Then, we'll code up a simple replacement - using a scrappy version of k-Nearest
# Neighbors. 
# https://www.youtube.com/watch?v=AoeEHqVSNOw
import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class RandomKnn():
   # save a copy of the data in the class
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

class ScrappyKnn():

    # save a copy of the data in the class
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            # measure distance between two points
            label = self.closest(row)
            predictions.append(label)
        return predictions

    # find the closest neighbor
    def closest(self, row):
        # k = 1
        # We only find the very closest neighbor
        best_dist = euc(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# import iris dataset
from sklearn import datasets
iris = datasets.load_iris()

# features
X = iris.data
# labels
y = iris.target

# split data into test and training sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# create decision tree classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

# show our accuracy for decision tree
from sklearn.metrics import accuracy_score
print("Decision Tree Classifier accuracy")
print(accuracy_score(y_test, predictions))

# Previously, we just imported knn classifier from sklearn
# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier = KNeighborsClassifier()

# create random knn classifier
random_knn_classifier = RandomKnn()
random_knn_classifier.fit(X_train, y_train)
random_kpredictions = random_knn_classifier.predict(X_test)

# create knn classifier
knn_classifier = ScrappyKnn()
knn_classifier.fit(X_train, y_train)
kpredictions = knn_classifier.predict(X_test)

# show our accuracy for random knn classifier
from sklearn.metrics import accuracy_score
print("Random KNN Classifier accuracy")
print(accuracy_score(y_test, random_kpredictions))

# show our accuracy for knn classifier
print("KNN Classifier accuracy")
print(accuracy_score(y_test, kpredictions))
