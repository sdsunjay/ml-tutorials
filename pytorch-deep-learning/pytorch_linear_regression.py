import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# we would like to generate 20 data points
N = 20

# random data on the x-axis in (-5, +5)
X = np.random.random(N)*10 - 5

# a line plus some noise
Y = 0.5 * X - 1 + np.random.randn(N)

# you'll have to take my "in-depth" series to understand
# why this is the CORRECT model to use with our MSE loss

# Plot the data
plt.scatter(X, Y)

# Create the linear regression model
model = nn.Linear(1,1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# in ML we want out data to be of shape:
# (num_samples x num_dimensions)
X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

# PyTorch use float32 by default
# Numpy creates float64 by default
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

type(inputs)
# torch.Tensor

# Train the model
# Main training loop

n_epochs = 30 # 30 iterations is what it takes for us to converge
losses = []
for it in range(n_epochs):
  # zero the parameter gradients
  optimizer.zero_grad()

  # forward pass
  outputs = model(inputs) # model is an object of type nn.Linear (from earlier), but we use it like a function
  loss = criterion(outputs, targets)

  # keep the loss so we can plot it later
  losses.append(loss.item())

  # backward and optimize
  loss.backward()
  optimizer.step() # do one step of gradient descent

  # print(f'Epoch {it+1}/{n_epochs}, Loss: {loss.item():.4f}')
  # plt.plot(losses) # should go to 0

predicted = model(inputs).detach().numpy()
plt.scatter(X, Y, label='Orginal Data')
plt.plot(X, predicted, label='Fitted line')
plt.legend()
plt.show()

# with torch.no_grad():
#  out = model(inputs).numpy()
# out

# important!
# In order to test the efficacy of our model, synthetic data is useful
# Why?
# Because ***we know the answer***
# True values of (w, b) are (0.5, -1)
w = model.weight.data.numpy() # model parameters
b = model.bias.data.numpy() # model parameters
print(w, b)
