# Six lines of Python is all it takes to write your first machine learning program! In this episode,
# we'll briefly introduce what machine learning is and why it's important. Then, we'll follow a
# recipe for supervised learning (a technique to create a classifier from examples) and code it up.
# https://www.youtube.com/watch?v=cKxRvEZd3Mw

from sklearn import tree
# 1 is apple
# 0 is orange
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))
 
