# Building our Neutral Network - Deep Learning and Neutral Networks with Python and PyTorch p3.ipynb
# https://youtu.be/ixathu7U-LQ

### copied from p2.py
import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# batch_size is how many we want to pass to our model (eventually we will have too many samples that will fit in memory)
# usually 8 - 64 is batch_size
# you do want a larger batch_size because generally this will effect how quickly you can train through all of your data
# You definitely want to shuffle
# mnist is hand drawn digits, all the ones are together, all the twos are together
# shuffle does exactly as it sounds, shuffles your data
# number of neurons per layer is trial and error
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

### end copy

import torch.nn as nn
import torch.nn.functional as F

input_size = 28 * 28 # 784 is the number of pixels when the image is flattened
output_size = 64

class Net(nn.Module):
  def __init__(self):
    super().__init__() # run initialization of nn.Module()
    self.fc1 = nn.Linear(input_size, output_size)
    self.fc2 = nn.Linear(output_size, output_size)
    self.fc3 = nn.Linear(output_size, output_size)
    self.fc4 = nn.Linear(output_size, 10) # we have 10 classes, we need 10 outputs

  # x is the data
  # simple neutral network is a feed forward network
  def forward(self, x):
    # ReLU is our activation function
    # Remember activation function keeps outputs between [0,1]
    # whether or not the neutron is firing
    # activation function runs on the output
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))

     # you could do if  weather == sunny
     # do something fancy
     # you can add your own logic

     x = F.relu(self.fc2(x))
     x = F.relu(self.fc3(x))
     x = self.fc4(x) # dont want to run on the 10 categories
     # we want a probability distribution on our output
     # apply softmax
     return F.log_softmax(x, dim=1)
     # dimension = 1
     # dimenson is the number of axis

net = Net()
X = torch.rand((28, 28))
X = X.view(-1,28*28) # -1 means input is unknown shape
output = net(X)

print(output)
