"""PyTorch CIFAR10.ipynb"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import sys

# root is where to download the data to
# train = true, return train dataset
# transform does some useful preprocessing
# download the data
# train dataset
train_dataset = torchvision.datasets.CIFAR10(root='.', train=True, transform=transforms.ToTensor(), download=True)
# test dataset
test_dataset = torchvision.datasets.CIFAR10(root='.', train=False, transform=transforms.ToTensor(), download=True)

# train_dataset.data.shape


# number of classes
K = len(set(train_dataset.targets))
print("Number of classes: ", K)

# Data load
# Useful because it automatically generates batches in the training loop
# and takes care of the shuffling

batch_size = 128
# shuffle training data, but not test data
# we dont want correlations between the data
# no need to shuffle test data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
tmp_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

# for x,y in tmp_loader:
#  print(x)
#  print(x.shape)
#  break

# Define the model
class CNN(nn.Module):
  def __init__(self, K):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 1024)
    self.fc2 = nn.Linear(1024, K)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 128 * 3 * 3)
    x = F.dropout(x, p=0.5)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.2)
    x = self.fc2(x)
    return x

# Instantiate the model
model = CNN(K)
if torch.cuda.is_available():
    model.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # we use this because we have multiple categories
optimizer = torch.optim.Adam(model.parameters()) # we use Adam jus cuz

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):

  # Stuff to store
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)

  for it in range(epochs):
    start = timer()
    train_loss = []
    for inputs, targets in train_loader:
      # Move data to GPU if device is GPU
      inputs, targets = inputs.to(device), targets.to(device)

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      # Backward and Optimize
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())


    # Get the train loss and test loss
    train_loss = np.mean(train_loss)  # a little misleading

    # test loss
    # same as train loop, except for backward and optimizer steps

    test_loss = []
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())

    test_loss = np.mean(test_loss)

    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss

    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    end = timer()
    duration = "{:.2f}".format(end - start)
    print('Duration: ' + str(duration)) # Time in seconds, e.g. 5.38091952400282
  return train_losses, test_losses

train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=15)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# Calculate the accuracy

# train accuracy
n_correct = 0
n_total = 0
for inputs, targets in train_loader:
  # move data to GPU
  inputs, targets = inputs.to(device), targets.to(device)

  # forward pass
  outputs = model(inputs)

  # get the prediction
  # torch.max returns both max and argmax
  _, predictions = torch.max(outputs, 1)

  # update counts
  n_correct += (predictions == targets).sum().item()
  n_total += targets.shape[0]

train_acc = n_correct / n_total

# test accuracy
n_correct = 0
n_total = 0
for inputs, targets in test_loader:
  # move data to GPU
  inputs, targets = inputs.to(device), targets.to(device)

  # forward pass
  outputs = model(inputs)

  # get the prediction
  # torch.max returns both max and argmax
  _, predictions = torch.max(outputs, 1)

  # update counts
  n_correct += (predictions == targets).sum().item()
  n_total += targets.shape[0]

test_acc = n_correct / n_total


print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
