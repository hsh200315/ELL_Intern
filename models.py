import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import random

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        print("Model: LeNet")
        self.conv1 = nn.Conv2d(3, 6, 5) #(입력 이미지 채널, 출력 채널, 정사각형 필터 크기) / CIFAR10, STL10 모두 컬러 이미지
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120) #(5*5 크기의 feature map이 16개 존재 -> faltten(1차원화) 이후 fcnn에 넘김)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #최종적으로 10개의 클래스 구분

    def forward_detail(self, input):
        c1 = F.relu(self.conv1(input))
        #32*32 크기의 이미지 입력 -> 5*5 필터링 -> 28*28크기의 6개의 feature map으로 출력 -> ReLU 통과
        s2 = F.max_pool2d(c1, (2, 2))
        #subsampling 진행, 2*2 max_pooling을 통해 28*28 -> 14*14
        c3 = F.relu(self.conv2(s2))
        #14*14 크기 6개의 feature map 입력 -> 5*5 필터링 -> 10*10 크기의 16개의 feature map으로 출력 -> ReLU 통과
        s4 = F.max_pool2d(c3, 2)
        #subsampling 진행, 2*2 max_pooling을 통해 10*10 -> 5*5 / (2,2) 와 2는 같은 크기를 나타냄
        s4 = torch.flatten(s4,1)
        #flatten 결과 (1,400)의 Tensor 생성
        f5 = F.relu(self.fc1(s4))
        #(400, 120)의 fcnn 통과 -> ReLU 통과
        f6 = F.relu(self.fc2(f5))
        #(120, 84)의 fcnn 통과 -> ReLU 통과
    
        output = self.fc3(f6)
        #(84, 10)의 fcnn 통과 -> 최종 10 class
        return output

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResBasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, downsampling=False):
        super(ResBasicBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        
        self.downsampling = downsampling
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1, stride=1)
        self.shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2)
        
        
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True) #inplace True하면 새로운 Tensor 만드는게 아니라 기존 Tensor 수정 -> 속도 향상/하지만 기존 값 사용하는 것에 영향
        
    def forward(self, x):
        save = x
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.bn2(self.conv2(c1))
        
        s = self.shortcut(save) if self.downsampling else save
        
        result = self.relu(c2+s)
        return result
    

class ResBottleNeckBlock(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, downsampling=False, projection=False):
        super(ResBottleNeckBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        
        self.projection = projection
        self.conv1 = nn.Conv2d(input_channel, middle_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(middle_channel, middle_channel, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(middle_channel, output_channel, kernel_size=1, padding=0, stride=1)
        self.shortcut =  nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride)
        
        
        self.bn1 = nn.BatchNorm2d(middle_channel)
        self.bn2 = nn.BatchNorm2d(middle_channel)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        save = x
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.relu(self.bn2(self.conv2(c1)))
        c3 = self.bn3(self.conv3(c2))
        
        s = self.shortcut(save) if self.projection else save
        
        result = self.relu(c3+s)
        return result

class ResNet(nn.Module):
    def __init__(self, start_channel, layer_num, block):
        #layer_num = [2,2,2,2]
        super(ResNet, self).__init__()
        self.conv = nn.ModuleList()
        #conv1
        self.conv.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        ))
        for i in range(4):
            self.conv.append(nn.Sequential())
            
        if block == "basic":
            for i in range(layer_num[0]): #conv2
                self.conv[1].append(ResBasicBlock(start_channel, start_channel))
                
            for i in range(3): #conv3,4,5
                self.conv[i+2].append(ResBasicBlock(start_channel*pow(2,i), start_channel*pow(2,i+1), True))
                for j in range(layer_num[i+1]-1):
                    self.conv[i+2].append(ResBasicBlock(start_channel*pow(2,i+1), start_channel*pow(2,i+1)))
                    
            self.fc1 = nn.Linear(start_channel*8, 10)
        elif block == "bottleneck":
            self.conv[1].append(ResBottleNeckBlock(start_channel, start_channel, start_channel*4, False, True)) #conv2
            for i in range(layer_num[0]-1):
                self.conv[1].append(ResBottleNeckBlock(start_channel*4, start_channel, start_channel*4))
            
            for i in range(3): #conv3,4,5
                self.conv[i+2].append(ResBottleNeckBlock(start_channel*pow(2,i+2), start_channel*pow(2,i+1), start_channel*pow(2,i+3), True, True))
                for j in range(layer_num[i+1]-1):
                    self.conv[i+2].append(ResBottleNeckBlock(start_channel*pow(2,i+3), start_channel*pow(2,i+1), start_channel*pow(2,i+3)))
            self.fc1 = nn.Linear(start_channel*32, 10)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        for i in range(5):
            for layer in self.conv[i]:
                x = layer(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x



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