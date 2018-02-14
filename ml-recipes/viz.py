# Visualizing a Decision Tree - Machine Learning Recipes 2

# Last episode, we treated our Decision Tree as a blackbox. In this episode, we'll build one on a
# real dataset, add code to visualize it, and practice reading it - so you can see how it works
# under the hood. And hey -- I may have gone a little fast through some parts. Just let me know,
# I'll slow down. Also: we'll do a Q&A episode down the road, so if anything is unclear, just ask!
# https://www.youtube.com/watch?v=tNa99PG8hR8

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print(iris.data[0])
print(iris.target[0])

# seperate test from training data
test_idx = [0,50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# Visualization Code
from sklearn.externals.six import StringIO
import pydot
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, impurity=False, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

print(test_data[1], test_target[1])
print(iris.feature_names)
print(iris.target_names)
