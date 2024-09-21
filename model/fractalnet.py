import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class FractalBasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FractalBasicBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, depth):
        super(FractalBlock, self).__init__()
        if depth == 1:
            self.block = FractalBasicBlock(input_channel, output_channel)
        else:
            self.fractal1 = FractalBlock(input_channel, output_channel, depth-1)
            self.fractal2 = FractalBlock(output_channel, output_channel, depth-1)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        if utils.is_true():
            result1 = self.fractal1(x)
        else:
            result1 = 0
        
        if result1 == 0:
            result2 = self.fractal2(self.fractal1(x))
        elif utils.is_true():
            result2 = self.fractal2(self.fractal1(x))
        else:
            result2 = 0
            
        return self.avgpool(result1 + result2)  #Local drop-path 구현

class FractalNet(nn.Module):
    def __init__(self, C):
        super(FractalNet, self).__init__()
        self.block1 = FractalBlock(3, 64, C)
        self.block2 = FractalBlock(64, 128, C)
        self.block3 = FractalBlock(128, 256, C)
        self.block4 = FractalBlock(256, 512, C)
        self.block5 = FractalBlock(512, 512, C)
        
        self.maxpool = nn.MaxPool2d((2,2))

    def forward(self, x):        
        x = self.maxpool(self.block1(x))
        x = self.maxpool(self.block2(x))
        x = self.maxpool(self.block3(x))
        x = self.maxpool(self.block4(x))
        x = self.maxpool(self.block5(x))
        
        return x