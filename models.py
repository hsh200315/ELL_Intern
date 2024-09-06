import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
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
