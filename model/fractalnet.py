import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)
random.seed(42)

fractal_list = nn.ModuleList()
num_input_join = 0

class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, drop_out_prob):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_out_prob)
        
    def forward(self, x):
        if self.training:
            x = self.dropout(self.relu(self.bn(self.conv(x))))
        else:
            x = self.relu(self.bn(self.conv(x)))
        return x
    
class JoinLayer(nn.Module):
    def __init__(self, depth, lprob):
        super(JoinLayer, self).__init__()
        self.depth = depth
        self.lprob = lprob
        
    def forward(self, *inputs):
        input_list = []
        selected = []
        local_drop = [] #local drop path
        for i in range(len(inputs)):
            local_drop.append(1)
            if random.random() < self.lprob:
                local_drop[i] = 0
        if sum(local_drop) == 0:
            local_drop[random.randint(0, 1)] = 1
        for i in range(len(inputs)):
            if local_drop[i] == 1:
                selected.append(inputs[i]) # local drop path
        
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
        self.is_global = True
        if depth == 1:
            self.block = ConvLayer(input_channel, output_channel, drop_out_prob)
            fractal_list[depth-1].append(self.block)
        else:                
            self.short_path = ConvLayer(input_channel, output_channel, drop_out_prob)
            fractal_list[depth-1].append(self.short_path)
            self.fractal1 = FractalBlock(input_channel, output_channel, depth-1, lprob, drop_out_prob, True)
            self.fractal2 = FractalBlock(output_channel, output_channel, depth-1, lprob, drop_out_prob, False)
            if use_join:
                self.join = JoinLayer(depth, lprob)
        
    def set_is_global(self, is_global):
        self.is_global = is_global
        if self.depth != 1:
            self.fractal1.set_is_global(is_global)
            self.fractal2.set_is_global(is_global)
        
    def forward(self, x):
        if self.depth == 1:
            return self.block(x)
        else:
            x1 = self.short_path(x)
            x2 = self.fractal2(self.fractal1(x))
            
            if self.use_join:
                return self.join(x1, x2)
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
        self.is_global = True
        for i in range(C):
            fractal_list.append(nn.ModuleList())
        self.block_list = nn.ModuleList()
        self.block_list.append(FractalBlock(3, start_channel, C, 0.15, 0.0, True))
        for i in range(B-1):
            if start_channel*(2**i+1) <= end_channel:
                self.block_list.append(FractalBlock(start_channel*(2**i), start_channel*(2**(i+1)), C, 0.15, 0.1*(i+1), True))
            else:
                self.block_list.append(FractalBlock(end_channel, end_channel, C, 0.15, 0.1*(i+1), True))
                
        self.maxpool = nn.MaxPool2d((2, 2))
        self.fc = nn.Linear(start_channel*8, 10)
        
    def set_is_global(self, is_global):
        self.is_global = is_global
        
    def forward(self, x):
        for i in range(self.B):
            x = self.block_list[i](x)
            x = self.maxpool(x)
        
        x = x.flatten(1)
        x = self.fc(x)
        return x