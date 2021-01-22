
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load in the data
from sklearn.datasets import load_breast_cancer

# call the function
data = load_breast_cancer()

print(type(data))

# sklearn.utils.Bunch

print(data.keys)

# print(data['feature_names'])
print(data.feature_names)

print(data.data)
# it has 569 samples, 30 features

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head)
# inputs and targets
data.target

print(data.target_names)

data.target.shape

# you can also determine the meaning of each feature
data.feature_names

# Split te data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
X, D = X_train.shape

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now all the fun PyTorch stuff
# Build the model
model = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid()
    )

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Convert data into torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

# train the model
n_epochs = 1000

# Stuff to store
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)


# Train the model
for it in range(n_epochs):
  # zero the parameter gradients
  optimizer.zero_grad()

  # forward pass
  outputs = model(X_train) # model is an object of type nn.Sequential
  loss = criterion(outputs, y_train)

  # keep the loss so we can plot it later
  # losses.append(loss.item())

  # backward and optimize
  loss.backward()
  optimizer.step() # do one step of gradient descent

  # Get the test loss
  outputs_test = model(X_test)
  loss_test = criterion(outputs_test, y_test)

  # Save losses
  train_losses[it] = loss.item()
  test_losses[it] = loss_test.item()

  if(it +1) % 50 == 0:
    print(f'Epoch {it+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}')

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()



# Get accuracy
with torch.no_grad():
  p_train = model(X_train)
  p_train = np.round(p_train.numpy())
  # true evaluates to 1
  # false evaluates to 0
  train_acc = np.mean(y_train.numpy() == p_train)


  p_test = model(X_test)
  p_test = np.round(p_test.numpy())
  test_acc = np.mean(y_test.numpy() == p_test)

print(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}" )

# Look at the state dict
model.state_dict()

# Save the model
torch.save(model.state_dict(), 'mymodel.pt')

# Load the model
model2 = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid()
    )

model2.load_state_dict(torch.load('mymodel.pt'))

# Get accuracy
with torch.no_grad():
  p_train = model2(X_train)
  p_train = np.round(p_train.numpy())
  # true evaluates to 1
  # false evaluates to 0
  train_acc2 = np.mean(y_train.numpy() == p_train)


  p_test = model2(X_test)
  p_test = np.round(p_test.numpy())
  test_acc2 = np.mean(y_test.numpy() == p_test)

print(f"Train accuracy 2: {train_acc2:.4f}, Test accuracy 2: {test_acc2:.4f}" )

# Download the model
from google.colab import files
files.download('mymodel.pt')
