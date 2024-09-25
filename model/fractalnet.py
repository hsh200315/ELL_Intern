import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, drop_out_prob):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_out_prob)
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class JoinLayer(nn.Module):
    def __init__(self):
        super(JoinLayer, self).__init__()
    
    def forward(self, input1, input2):
        return (input1+input2)/2

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, depth, local_prob, drop_out_prob):
        super(FractalBlock, self).__init__()
        self.depth = depth
        self.local_prob = local_prob
        if depth == 1:
            self.block = ConvLayer(input_channel, output_channel, drop_out_prob)
        else:
            self.short_path = ConvLayer(input_channel, output_channel)
            self.block1 = FractalBlock(input_channel, output_channel, depth-1, local_prob, drop_out_prob)
            self.block2 = FractalBlock(output_channel, output_channel, depth-1, local_prob, drop_out_prob)
            self.join = JoinLayer()
            
    def forward(self, x, epoch):
        if epoch%2 == 0: #local drop
            if self.depth == 1:
                x = self.block(x)
            else:
                if random.random() >= self.local_prob:
                    result1 = self.short_path(x)
                    if random.random() >= self.local_prob:
                        result2 = self.block2(self.block1(x))
                        x = self.join(result1, result2)
                    else:
                        x = result1
                else:
                    x = self.block2(self.block1(x))
                    
        else: #global drop
            x = x
            
        return x


class FractalNet(nn.Module):
    def __init__(self, start_channel, B, C):
        super(FractalNet, self).__init__()
        self.B = B
        self.blocks = nn.ModuleList()
        self.blocks.append(FractalBlock(3, start_channel, C, 0.5, 0))
        for i in range(B-1):
            self.blocks.append(FractalBlock(start_channel*pow(2,i), start_channel*pow(2,i+1), C, 0.5, 0.1*(i+1)))
        
        self.maxpool = nn.MaxPool2d((2,2))
        self.fc = nn.Linear(start_channel*pow(2,B-1), 10)

    def forward(self, x, epoch):      
        for i in range(self.B):
            x = self.maxpool(self.blocks[i](x, epoch))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x