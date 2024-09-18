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



class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsampling=False):
        super(BasicBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        padding = 3 if downsampling else 2 #downsampling 진행 시 이미지 크기 유지
        
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3)
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride)
        
        self.BN = nn.BatchNorm2d(output_dim)
        
    def forward(self, x):
        c1 = F.relu(self.BN(self.conv1(x)))
        c2 = self.BN(self.conv2(c1))
        s = self.BN(self.shortcut(x))
        
        result = F.relu(c2+s)
        return result

class BottleNeckBlock(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, downsampling=False):
        super(BottleNeckBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        
        self.conv1 = nn.Conv2d(input_dim, middle_dim, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(middle_dim, middle_dim, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(middle_dim, output_dim, kernel_size=1, padding=0, stride=1)
        self.shortcut =  nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride)
        
        self.BN1 = nn.BatchNorm2d(middle_dim)
        self.BN2 = nn.BatchNorm2d(output_dim)
        
    def forward(self, x):
        c1 = F.relu(self.BN1(self.conv1(x)))
        c2 = F.relu(self.BN1(self.conv2(c1)))
        c3 = self.BN2(self.conv3(c2))
        s = self.BN2(self.shortcut(x))
        
        result = F.relu(c3+s)
        return result

class ResNet(nn.Module):
    def __init__(self, num_layer):
        super(ResNet, self).__init__()
        layer_list = [18, 34, 110, 50, 101, 152]
        self.layer_2 = [2, 3, 3, 3, 3, 3]
        self.layer_3 = [2, 4, 4, 4, 4, 8]
        self.layer_4 = [2, 6, 44, 6, 23, 36]
        self.layer_5 = [2, 3, 3, 3, 3, 3]
        try:
            model_num = layer_list.index(num_layer)
            self.model_num = model_num
        except:
            print("ResNet layer 수를 [18, 34, 110, 50, 101, 152] 중 골라주세요")
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        if self.model_num <= 1:
            self.conv2_1 = BasicBlock(64, 64)
            self.conv2_2 = BasicBlock(64, 64)
            self.conv3_1 = BasicBlock(64, 128, True)
            self.conv3_2 = BasicBlock(128, 128)
            self.conv4_1 = BasicBlock(128, 256, True)
            self.conv4_2 = BasicBlock(256, 256)
            self.conv5_1 = BasicBlock(256, 512, True)
            self.conv5_2 = BasicBlock(512, 512)
            self.fc1 = nn.Linear(512, 10)
        elif self.model_num == 2:
            self.conv2_1 = BasicBlock(64, 256)
            self.conv2_2 = BasicBlock(256, 256)
            self.conv3_1 = BasicBlock(256, 512, True)
            self.conv3_2 = BasicBlock(512, 512)
            self.conv4_1 = BasicBlock(512, 1024, True)
            self.conv4_2 = BasicBlock(1024, 1024)
            self.conv5_1 = BasicBlock(1024, 2048, True)
            self.conv5_2 = BasicBlock(2048, 2048)
            self.fc1 = nn.Linear(2048, 10)
        else:
            self.conv2_1 = BottleNeckBlock(64, 64, 256)
            self.conv2_2 = BottleNeckBlock(256, 64, 256)
            self.conv3_1 = BottleNeckBlock(256, 128, 512, True)
            self.conv3_2 = BottleNeckBlock(512, 128, 512)
            self.conv4_1 = BottleNeckBlock(512, 256, 1024, True)
            self.conv4_2 = BottleNeckBlock(1024, 256, 1024)
            self.conv5_1 = BottleNeckBlock(1024, 512, 2048, True)
            self.conv5_2 = BottleNeckBlock(2048, 512, 2048)
            self.fc1 = nn.Linear(2048, 10)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2_1(x)
        for i in range(self.layer_2[self.model_num]-1):
            self.conv2_2(x)

        x = self.conv3_1(x)
        for i in range(self.layer_3[self.model_num]-1):
            x = self.conv3_2(x)
    
        x = self.conv4_1(x)
        for i in range(self.layer_4[self.model_num]-1):
            x = self.conv4_2(x)
            
        x = self.conv5_1(x)
        for i in range(self.layer_5[self.model_num]-1):
            x = self.conv5_2(x)
            
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x




class PreActBasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsampling=False):
        super(PreActBasicBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        padding = 3 if downsampling else 2 #downsampling 진행 시 이미지 크기 유지
        
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3)
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride)
        
        self.BN1 = nn.BatchNorm2d(input_dim)
        self.BN2 = nn.BatchNorm2d(output_dim)
        
    def forward(self, x):
        c1 = self.conv1(F.relu(self.BN1(x)))
        c2 = self.conv2(F.relu_(self.BN2(c1)))
        s = self.shortcut(x)
        
        result = c2+s
        return result

class PreActBottleNeckBlock(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, downsampling=False):
        super(PreActBottleNeckBlock, self).__init__()
        stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
        
        self.conv1 = nn.Conv2d(input_dim, middle_dim, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(middle_dim, middle_dim, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(middle_dim, output_dim, kernel_size=1, padding=0, stride=1)
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride)
        
        self.BN1 = nn.BatchNorm2d(input_dim)
        self.BN2 = nn.BatchNorm2d(middle_dim)
    
    def forward(self, x):
        c1 = self.conv1(F.relu(self.BN1(x)))
        c2 = self.conv2(F.relu(self.BN2(c1)))
        c3 = self.conv3(F.relu(self.BN2(c2)))
        s = self.shortcut(x)
        
        result = c3+s
        return result

class PreActResNet(nn.Module):
    def __init__(self, num_layer):
        super(PreActResNet, self).__init__()
        layer_list = [101, 110]
        self.layer_2 = [3, 3]
        self.layer_3 = [4, 4]
        self.layer_4 = [23, 44]
        self.layer_5 = [3, 3]
        try:
            model_num = layer_list.index(num_layer)
            self.model_num = model_num
            print(num_layer, model_num)
        except:
            print("PreActResNet layer 수를 [101, 110] 중 골라주세요")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        if self.model_num == 0:
            self.conv2_1 = PreActBottleNeckBlock(64, 64, 256)
            self.conv2_2 = PreActBottleNeckBlock(256, 64, 256)
            self.conv3_1 = PreActBottleNeckBlock(256, 128, 512, True)
            self.conv3_2 = PreActBottleNeckBlock(512, 128, 512)
            self.conv4_1 = PreActBottleNeckBlock(512, 256, 1024, True)
            self.conv4_2 = PreActBottleNeckBlock(1024, 256, 1024)
            self.conv5_1 = PreActBottleNeckBlock(1024, 512, 2048, True)
            self.conv5_2 = PreActBottleNeckBlock(2048, 512, 2048)
        elif self.model_num == 1:
            self.conv2_1 = PreActBasicBlock(64, 256)
            self.conv2_2 = PreActBasicBlock(256, 256)
            self.conv3_1 = PreActBasicBlock(256, 512, True)
            self.conv3_2 = PreActBasicBlock(512, 512)
            self.conv4_1 = PreActBasicBlock(512, 1024, True)
            self.conv4_2 = PreActBasicBlock(1024, 1024)
            self.conv5_1 = PreActBasicBlock(1024, 2048, True)
            self.conv5_2 = PreActBasicBlock(2048, 2048)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.BN = nn.BatchNorm2d(2048)
        self.fc1 = nn.Linear(2048, 10)
            
    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2_1(x)
        for i in range(self.layer_2[self.model_num]-1):
            x = self.conv2_2(x)

        x = self.conv3_1(x)
        for i in range(self.layer_3[self.model_num]-1):
            x = self.conv3_2(x)
    
        x = self.conv4_1(x)
        for i in range(self.layer_4[self.model_num]-1):
            x = self.conv4_2(x)
            
        x = self.conv5_1(x)
        for i in range(self.layer_5[self.model_num]-1):
            x = self.conv5_2(x)
            
        x = F.relu(self.BN(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    

    
class BasicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicLayer, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1)
        self.BN = nn.BatchNorm2d(output_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = F.relu(x)
        
        return x

class FractalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, depth):
        super(FractalBlock, self).__init__()
        if depth == 1:
            self.block = BasicLayer(input_dim, output_dim)
        else:
            self.fractal1 = FractalBlock(input_dim, output_dim, depth-1)
            self.fractal2 = FractalBlock(output_dim, output_dim, depth-1)
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