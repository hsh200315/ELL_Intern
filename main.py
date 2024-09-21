import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import utils
import datasets
import models

args = utils.add_args()
epochs = int(args.epoch)
batch_size = int(args.batch_size)
lr = float(args.lr)

#Trained Model Save Path
model_name = f'{args.dataset}_{args.model}'
PATH = f'./models/{model_name}.pth'
PATH_FOR_LOG = f'./runs/{model_name}'

#Load Network
args.model = args.model.lower()
if args.model == "lenet":
    net = models.LeNet()
elif args.model[:6] == "resnet":
    layer_num = args.layer; block = args.block
    net = models.ResNet(64, layer_num, block)
elif args.model[:12] == "preactresnet":
    layer_num = args.layer; block = args.block
    net = models.PreActResNet(64, layer_num, block)
elif args.model[:8] == "densenet":
    layer_num = args.layer; growth_rate = args.growth_rate; theta = args.theta
    net = models.DenseNet(64, layer_num, growth_rate, theta)
elif args.model == "fractalnet":
    net = models.FractalNet(4)

net.to('cuda')

#Load Data
train_data, test_data, classes = datasets.load_data(args.dataset)
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.00001, momentum=0.9)
milestones = [int(epochs*0.5), int(epochs*0.75)]
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

writer = SummaryWriter(PATH_FOR_LOG)

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
    scheduler.step()
    correct = 0
    total = 0
    with torch.no_grad(): 
        #torch는 autograd기능을 제공하고, default로 x.requires_grad=True/x의 기울기를 담은 tensor를 갖고 있는다
        #no_grad를 사용하면, gradient 계산 기능을 끄고(requires_grad=False) 메모리 소비를 줄인다. inference 할 때 유용하다
        for data in testLoader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    writer.add_scalar("Loss/train", running_loss/len(trainLoader), epoch)
    writer.add_scalar("Accuracy", 100 * correct // total, epoch)
    print(f'epoch: {epoch} loss: {running_loss/len(trainLoader):.3f} accuracy: {100 * correct // total}')
    

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

writer.flush() #disk에 저장함으로써, 기록 확실히 함
writer.close()

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