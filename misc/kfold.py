from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Configuring the KFold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)

model = LogisticRegression(max_iter=10000)
# Lists to store the results of accuracy scores
accuracy_scores = []

for train_index, test_index in kf.split(X):
    # Splitting the dataset into training and testing sets for each fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions
    predictions = model.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

# Average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
print(f"Average Accuracy: {average_accuracy}")

