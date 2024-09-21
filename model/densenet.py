import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class DenseTransitionLayer(nn.Module):
    def __init__(self, input_channel, theta=0.5):
        super(DenseTransitionLayer, self).__init__()
        
        self.bn = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channel, int(theta*input_channel), kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d((2,2), 2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.avgpool(x)
        
        return x

class DenseBottleNeck(nn.Module):
    def __init__(self, input_channel, growth_rate):
        super(DenseBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 4*growth_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))
        
        return x

class DenseBlock(nn.Module):
    def __init__(self, input_channel, growth_rate, num_layer):
        super(DenseBlock, self).__init__()
        self.num_layer = num_layer
        self.bottlenecks = nn.ModuleList()
        
        for i in range(num_layer):
            self.bottlenecks.append(DenseBottleNeck(input_channel+i*growth_rate, growth_rate))

    def forward(self, x):
        feature_map = [x]
        for i in range(self.num_layer):
            x = self.bottlenecks[i](torch.cat(feature_map, 1))
            feature_map.append(x)
        return torch.cat(feature_map, 1)

class DenseNet(nn.Module):
    def __init__(self, start_channel, layer_num, growth_rate, theta=0.5):
        super(DenseNet, self).__init__()
        growth_rate = int(growth_rate); theta = float(theta)
        input_channel = [start_channel]
        
        self.conv1 = nn.Conv2d(3, input_channel[0], kernel_size=3, stride=1, padding=1)
        
        self.dense_block = nn.ModuleList()
        self.transition_layer = nn.ModuleList()
        for i in range(2):
            self.dense_block.append(DenseBlock(input_channel[i], growth_rate, layer_num[i]))
            self.transition_layer.append(DenseTransitionLayer(input_channel[i]+growth_rate*layer_num[i], theta))
            input_channel.append(int((input_channel[i]+growth_rate*layer_num[i])*theta))
        self.dense_block.append(DenseBlock(input_channel[2], growth_rate, layer_num[2]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        num = input_channel[2] + growth_rate*layer_num[2]
        self.bn = nn.BatchNorm2d(num)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num, 10)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(2):
            x = self.dense_block[i](x)
            x = self.transition_layer[i](x)
        x = self.dense_block[2](x)
        
        x = self.avgpool(self.relu(self.bn(x)))
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x