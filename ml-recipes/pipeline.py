# Let’s Write a Pipeline - Machine Learning Recipes 4

# In this episode, we’ll write a basic pipeline for supervised learning with just 12 lines of code.
# Along the way, we'll talk about training and testing data. Then, we’ll work on our intuition for
# what it means to “learn” from data. 
# https://www.youtube.com/watch?v=84gqSbLcBFE

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

# create knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
kpredictions = knn_classifier.predict(X_test)

# show our accuracy for knn classifier
from sklearn.metrics import accuracy_score
print("KNN Classifier accuracy")
print(accuracy_score(y_test, kpredictions))

