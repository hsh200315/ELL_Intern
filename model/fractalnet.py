import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)
random.seed(42)

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
        input_list = []
        selected = []
        local_drop = [] 
        if self.training:
            if is_global: #global drop path
                if self.depth <= path_num:
                    selected.append(inputs[0])
                if path_num == 1 or self.depth > path_num:
                    selected.append(inputs[1]) #global drop path
            else:
                for i in range(len(inputs)):#local drop path
                    local_drop.append(1)
                    if random.random() < self.lprob:
                        local_drop[i] = 0
                if sum(local_drop) == 0:
                    local_drop[random.randint(0, 1)] = 1
                for i in range(len(inputs)):
                    if local_drop[i] == 1:
                        selected.append(inputs[i]) # local drop path
        else:
            selected = inputs
        for i in range(len(selected)):
            if selected[i].dim() != 4:
                for j in range(selected[i].size(0)):
                    input_list.append(selected[i][j])
            else:
                input_list.append(selected[0])
        
        x = torch.stack(input_list, dim=0).mean(dim=0)
        return x

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, depth, lprob, drop_out_prob, use_join):
        super(FractalBlock, self).__init__()
        self.depth = depth
        self.use_join = use_join
        self.block = ConvLayer(input_channel, output_channel, drop_out_prob)
        if depth > 1:                
            self.fractal1 = FractalBlock(input_channel, output_channel, depth-1, lprob, drop_out_prob, True)
            self.fractal2 = FractalBlock(output_channel, output_channel, depth-1, lprob, drop_out_prob, False)
            if use_join:
                self.join = JoinLayer(depth, lprob)
        
    def forward(self, x, is_global, path_num):
        if self.depth == 1:
            return self.block(x)
        else:
            x1 = self.block(x)
            x2 = self.fractal2(self.fractal1(x, is_global, path_num), is_global, path_num)
            
            if self.use_join:
                return self.join([x1, x2], is_global, path_num)
            else:
                input_list = [x1]
                if x2.dim() != 4:
                    for i in range(x2.size(0)):
                        input_list.append(x2[i])
                return torch.stack(input_list, dim=0)
    
class FractalNet(nn.Module):
    def __init__(self, start_channel, end_channel, B, C):
        super(FractalNet, self).__init__()
        self.B = B
        self.C = C
        self.block_list = nn.ModuleList()
        self.block_list.append(FractalBlock(3, start_channel, C, 0.15, 0.0, True))
        for i in range(B-1):
            if start_channel*(2**i+1) <= end_channel:
                self.block_list.append(FractalBlock(start_channel*(2**i), start_channel*(2**(i+1)), C, 0.15, 0.1*(i+1), True))
            else:
                self.block_list.append(FractalBlock(end_channel, end_channel, C, 0.15, 0.1*(i+1), True))
                
        self.maxpool = nn.MaxPool2d((2, 2))
        self.fc = nn.Linear(start_channel*8, 10)
        
    def forward(self, x, is_global):
        for i in range(self.B):
            x = self.block_list[i](x, is_global, random.randint(1, self.C))
            x = self.maxpool(x)
        
        x = x.flatten(1)
        x = self.fc(x)
        return x