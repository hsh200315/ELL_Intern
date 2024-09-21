import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, downsampling=False):
        super(PreActBasicBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        
        self.downsampling = downsampling
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride)
        
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        save = x
        c1 = self.conv1(self.relu(self.bn1(x)))
        c2 = self.conv2(self.relu(self.bn2(c1)))
        
        s = self.shortcut(save) if self.downsampling else save
        
        result = c2+s
        return result

class PreActBottleNeckBlock(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, downsampling=False, projection=False):
        super(PreActBottleNeckBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        
        self.projection = projection
        self.conv1 = nn.Conv2d(input_channel, middle_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(middle_channel, middle_channel, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(middle_channel, output_channel, kernel_size=1, padding=0, stride=1)
        self.shortcut =  nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride)
        
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.bn2 = nn.BatchNorm2d(middle_channel)
        self.bn3 = nn.BatchNorm2d(middle_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        save = x
        c1 = self.conv1(self.relu(self.bn1(x)))
        c2 = self.conv2(self.relu(self.bn2(c1)))
        c3 = self.conv3(self.relu(self.bn3(c2)))
        
        s = self.shortcut(save) if self.projection else save
        
        result = c3+s
        return result

class PreActResNet(nn.Module):
    def __init__(self, start_channel, layer_num, block):
        super(PreActResNet, self).__init__()
        self.conv = nn.ModuleList()

        self.conv.append(nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)))
        for i in range(4):
            self.conv.append(nn.Sequential())

        if block == "basic":
            for i in range(layer_num[0]):
                self.conv[1].append(PreActBasicBlock(start_channel, start_channel))
            for i in range(3): #conv3,4,5
                self.conv[i+2].append(PreActBasicBlock(start_channel*pow(2,i), start_channel*pow(2,i+1), True))
                for j in range(layer_num[i+1]-1):
                    self.conv[i+2].append(PreActBasicBlock(start_channel*pow(2,i+1), start_channel*pow(2,i+1)))
            self.bn = nn.BatchNorm2d(start_channel*8)
            self.fc1 = nn.Linear(start_channel*8, 10)
        elif block == "bottleneck":
            self.conv[1].append(PreActBottleNeckBlock(start_channel, start_channel, start_channel*4, False, True)) #conv2
            for i in range(layer_num[0]-1):
                self.conv[1].append(PreActBottleNeckBlock(start_channel*4, start_channel, start_channel*4))
            
            for i in range(3): #conv3,4,5
                self.conv[i+2].append(PreActBottleNeckBlock(start_channel*pow(2,i+2), start_channel*pow(2,i+1), start_channel*pow(2,i+3), True, True))
                for j in range(layer_num[i+1]-1):
                    self.conv[i+2].append(PreActBottleNeckBlock(start_channel*pow(2,i+3), start_channel*pow(2,i+1), start_channel*pow(2,i+3)))
            self.bn = nn.BatchNorm2d(start_channel*32)
            self.fc1 = nn.Linear(start_channel*32, 10)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        for i in range(5):
            for layer in self.conv[i]:
                x = layer(x)
        
        x = F.relu(self.bn(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x