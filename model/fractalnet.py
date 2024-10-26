import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)
random.seed(42)

#Conv 검토 완료
class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, drop_out_prob):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_out_prob)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.bn(self.conv(x))))
        return x
    
class JoinLayer(nn.Module):
    def __init__(self, depth, lprob):
        super(JoinLayer, self).__init__()
        self.depth = depth
        self.lprob = lprob
        
    def forward(self, inputs, is_global, path_num):
        selected = []
        local_drop = []
        if self.training:
            if is_global: #global drop path
                if self.depth <= path_num:
                    selected.append(inputs[0])
                if path_num == 1 or self.depth > path_num:
                    selected.append(inputs[self.depth-path_num]) #global drop path 오류
            else:
                for i in range(self.depth):#local drop path
                    local_drop.append(1)
                    if random.random() < self.lprob:
                        local_drop[i] = 0
                if sum(local_drop) == 0:
                    local_drop[random.randint(0, self.depth-1)] = 1
                for i in range(self.depth):
                    if local_drop[i] == 1:
                        selected.append(inputs[i]) # local drop path
        else:
            selected = inputs
            
        if len(selected) == 1:
            return selected[0]
        x = torch.stack(selected, dim=0).mean(dim=0)
        return x

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, depth, lprob, drop_out_prob):
        super(FractalBlock, self).__init__()
        self.depth = depth
        self.block = ConvLayer(input_channel, output_channel, drop_out_prob)
        if depth > 1:                
            self.fractal1 = FractalBlock(input_channel, output_channel, depth-1, lprob, drop_out_prob)
            self.fractal2 = FractalBlock(output_channel, output_channel, depth-1, lprob, drop_out_prob)
            self.join = JoinLayer(depth-1, lprob)
        
    def forward(self, x, is_global, path_num):
        output = []
        if self.depth == 1:
            output.append(self.block(x))
        else:
            x1 = self.block(x)
            x2 = self.join(self.fractal1(x, is_global, path_num), is_global, path_num)
            x2 = self.fractal2(x2, is_global, path_num)
            
            output.append(x1)
            for i in x2:
                output.append(i)
        return output
    
class FractalNet(nn.Module):
    def __init__(self, start_channel, end_channel, B, C):
        super(FractalNet, self).__init__()
        self.B = B
        self.C = C
        self.block_list = nn.ModuleList()
        self.block_list.append(FractalBlock(3, start_channel, C, 0.15, 0.0))
        self.join = JoinLayer(C, 0.15)
        for i in range(B-1):
            if start_channel*(2**i+1) <= end_channel:
                self.block_list.append(FractalBlock(start_channel*(2**i), start_channel*(2**(i+1)), C, 0.15, 0.1*(i+1)))
            else:
                self.block_list.append(FractalBlock(end_channel, end_channel, C, 0.15, 0.1*(i+1)))
                
        self.maxpool = nn.MaxPool2d((2, 2))
        self.fc = nn.Linear(start_channel*8, 10)
        
    def forward(self, x, is_global):
        for i in range(self.B):
            path_num = random.randint(1, self.C)
            x = self.block_list[i](x, is_global, path_num)
            x = self.join(x, is_global, path_num)
            x = self.maxpool(x)
        
        x = x.flatten(1)
        x = self.fc(x)
        return x