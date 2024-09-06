import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import utils
import datasets
import models

args = utils.add_args()

epochs = int(args.epoch)
batch_size = int(args.batch_size)
lr = float(args.lr)

#Load Data
train_data, test_data = datasets.load_data(args.dataset)
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Load Model
args.model = args.model.lower()
if args.model == "lenet":
  net = models.LeNet()
  
if torch.cuda.is_available():
  net.to('cuda')
  
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
  
for epoch in range(epochs):
	running_loss = 0.0
	for i, data in enumerate(trainLoader, 0):
		inputs, labels = data
		if torch.cuda.is_available():
			inputs = inputs.to('cuda')
			labels = labels.to('cuda')

		optimizer.zero_grad()
  
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
  
		running_loss += loss.item()
		if i % 2000 == 1999:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
			running_loss = 0.0

print("Fininsed Training")