# pip install torchvision

# Data - Deep Learning and Neutral Networks with Python and PyTorch - p2
# https://youtu.be/i2yPxY2rOzs

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

for data in trainset:
  print(data)
  break

x,y = data[0][0], data[1][0]

print(y)

# pip install matplotlin
import matplotlib.pyplot as plt
# plt.imshow(data[0][0])
# issue with this line because its 1,28,28. instead of 28,28

# this shows us its a 1,28,28
print(data[0][0].shape)

plt.imshow(data[0][0].view(28,28))
plt.show()


# what is a balancing
# the model is trying to decrease loss
# it doesn't know how good we can get
# it'll just try to decrease loss as easy and best as possible
# we gotta make sure our data is balanced, evenly distributed
# if not, we'd need to fix it

total = 0
counter_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
for data in trainset:
  Xs, ys = data
  for y in ys:
    counter_dict[int(y)] += 1
    total += 1

print(counter_dict)

print("Percentage distribution: ")
for i in counter_dict:
  print(f"{i}: {counter_dict[i]/total*100}")
