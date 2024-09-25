import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class JoinLayer(nn.Module):
    def __init__(self):
        super(JoinLayer, self).__init__()
    
    def forward(self, input1, input2):
        return (input1+input2)/2

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, depth):
        super(FractalBlock, self).__init__()
        self.depth = depth
        if depth == 1:
            self.block = ConvLayer(input_channel, output_channel)
        else:
            self.short_path = ConvLayer(input_channel, output_channel)
            self.block1 = FractalBlock(input_channel, output_channel, depth-1)
            self.block2 = FractalBlock(output_channel, output_channel, depth-1)
            self.join = JoinLayer()
            
    def forward(self, x):
        if self.depth == 1:
            x = self.block(x)
        else:
            result1 = self.short_path(x)
            result2 = self.block2(self.block1(x))
            x = self.join(result1, result2)
        return x


class FractalNet(nn.Module):
    def __init__(self, start_channel, B, C):
        super(FractalNet, self).__init__()
        self.B = B
        self.blocks = nn.ModuleList()
        self.blocks.append(FractalBlock(3, start_channel, C))
        for i in range(B-1):
            self.blocks.append(FractalBlock(start_channel*pow(2,i), start_channel*pow(2,i+1), C))
        
        self.maxpool = nn.MaxPool2d((2,2))
        self.fc = nn.Linear(start_channel*pow(2,B-1), 10)

    def forward(self, x):      
        for i in range(self.B):
            x = self.maxpool(self.blocks[i](x))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x