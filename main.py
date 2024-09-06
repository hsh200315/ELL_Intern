import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import utils
import datasets
import models

PATH = './models/cifar_LeNet.pth'

args = utils.add_args()
epochs = int(args.epoch)
batch_size = int(args.batch_size)
lr = float(args.lr)

writer = SummaryWriter()

#Load Data
train_data, test_data, classes = datasets.load_data(args.dataset)
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

#Load Model
args.model = args.model.lower()
if args.model == "lenet":
  net = models.LeNet()
  
net.to('cuda')
  
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
  
for epoch in range(epochs):
	running_loss = 0.0
	for i, data in enumerate(trainLoader, 0):
		inputs, labels = data[0].to('cuda'), data[1].to('cuda')

		optimizer.zero_grad()
  
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
  
		running_loss += loss.item()
	print(f'epoch: {epoch} loss: {running_loss/len(trainLoader):.3f}')
	writer.add_scalar("Loss/train", running_loss/len(trainLoader), epoch)

writer.flush()
writer.close()
print("Fininsed Training")
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].to('cuda'), data[1].to('cuda')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].to('cuda'), data[1].to('cuda')
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')