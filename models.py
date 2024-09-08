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




class BasicBlock(nn.Module):
	def add_block(input_dim, output_dim, downsampling=False):
		stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
		padding = 3 if downsampling else 2 #downsampling 진행 시 이미지 크기 유지
	
		block = nn.Sequential(
			nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=padding, stride=stride),
			nn.BatchNorm2d(output_dim), #매 convolution 이후, activation 전 BN 적용
			nn.ReLU(),
			nn.Conv2d(output_dim, output_dim, kernel_size=3),
			nn.BatchNorm2d(output_dim)
		)
		return block

	def add_projection(input_dim, output_dim):
		shortcut = nn.Sequential(
			nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=2), #shortcut projection 연산
			nn.BatchNorm2d(output_dim)
		)
		return shortcut

class BottleNeckBlock(nn.Module):
	def add_block(input_dim, middle_dim, output_dim, downsampling=False):
		stride = 2 if downsampling else 1 #conv3_1, conv4_1, conv5_1에서 downsampling
  
		block = nn.Sequential(
			nn.Conv2d(input_dim, middle_dim, kernel_size=1, padding=0, stride=1),
			nn.BatchNorm2d(middle_dim), #매 convolution 이후, activation 전 BN 적용
			nn.ReLU(),
			nn.Conv2d(middle_dim, middle_dim, kernel_size=3, padding=1, stride=stride),
			nn.BatchNorm2d(middle_dim),
			nn.ReLU(),
			nn.Conv2d(middle_dim, output_dim, kernel_size=1, padding=0, stride=1),
			nn.BatchNorm2d(output_dim)
		)
		return block

	def add_projection(input_dim, output_dim, donwsampling=False):
		stride = 2 if donwsampling else 1
		shortcut = nn.Sequential(
			nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride), #shortcut projection 연산
			nn.BatchNorm2d(output_dim)
		)
		return shortcut

class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
		self.conv2_1 = BasicBlock.add_block(64, 64)
		self.conv2_2 = BasicBlock.add_block(64, 64)
		self.conv3_1 = BasicBlock.add_block(64, 128, True)
		self.conv3_2 = BasicBlock.add_block(128, 128)
		self.conv4_1 = BasicBlock.add_block(128, 256, True)
		self.conv4_2 = BasicBlock.add_block(256, 256)
		self.conv5_1 = BasicBlock.add_block(256, 512, True)
		self.conv5_2 = BasicBlock.add_block(512, 512)
	
		self.fc1 = nn.Linear(512 * 7 * 7, 1000)

		self.p1 = BasicBlock.add_projection(64, 128) #projection 연산 (차원 일치)
		self.p2 = BasicBlock.add_projection(128, 256)
		self.p3 = BasicBlock.add_projection(256, 512)
	
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, padding=1)

		input = x; x = F.relu(self.conv2_1(x) + input)
		input = x; x = F.relu(self.conv2_2(x) + input)
		input = x; x = F.relu(self.conv3_1(x) + self.p1(input))
		input = x; x = F.relu(self.conv3_2(x) + input)
		input = x; x = F.relu(self.conv4_1(x) + self.p2(input))
		input = x; x = F.relu(self.conv4_2(x) + input)
		input = x; x = F.relu(self.conv5_1(x) + self.p3(input))
		input = x; x = F.relu(self.conv5_2(x) + input)
	
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		return x


class ResNet18_STL10(nn.Module):
	def __init__(self):
		super(ResNet18_STL10, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=8, stride=2)
		self.conv2_1 = BasicBlock.add_block(64, 64)
		self.conv2_2 = BasicBlock.add_block(64, 64)
		self.conv3_1 = BasicBlock.add_block(64, 128, True)
		self.conv3_2 = BasicBlock.add_block(128, 128)
		self.conv4_1 = BasicBlock.add_block(128, 256, True)
		self.conv4_2 = BasicBlock.add_block(256, 256)
		self.conv5_1 = BasicBlock.add_block(256, 512, True)
		self.conv5_2 = BasicBlock.add_block(512, 512)
	
		self.fc1 = nn.Linear(512 * 7 * 7, 1000)

		self.p1 = BasicBlock.add_projection(64, 128) #projection 연산 (차원 일치)
		self.p2 = BasicBlock.add_projection(128, 256)
		self.p3 = BasicBlock.add_projection(256, 512)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
	
		input = x; x = F.relu(self.conv2_1(x) + input)
		input = x; x = F.relu(self.conv2_2(x) + input)
		input = x; x = F.relu(self.conv3_1(x) + self.p1(input))
		input = x; x = F.relu(self.conv3_2(x) + input)
		input = x; x = F.relu(self.conv4_1(x) + self.p2(input))
		input = x; x = F.relu(self.conv4_2(x) + input)
		input = x; x = F.relu(self.conv5_1(x) + self.p3(input))
		input = x; x = F.relu(self.conv5_2(x) + input)

		x = torch.flatten(x, 1)
		x = self.fc1(x)
		return x

class ResNet50(nn.Module):
	def __init__(self):
		super(ResNet50, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
	
		self.conv2_1 = BottleNeckBlock.add_block(64, 64, 256)
		self.conv2_2 = BottleNeckBlock.add_block(256, 64, 256)
		self.conv2_3 = BottleNeckBlock.add_block(256, 64, 256)
	
		self.conv3_1 = BottleNeckBlock.add_block(256, 128, 512, True)
		self.conv3_2 = BottleNeckBlock.add_block(512, 128, 512)
		self.conv3_3 = BottleNeckBlock.add_block(512, 128, 512)
		self.conv3_4 = BottleNeckBlock.add_block(512, 128, 512)
	
		self.conv4_1 = BottleNeckBlock.add_block(512, 256, 1024, True)
		self.conv4_2 = BottleNeckBlock.add_block(1024, 256, 1024)
		self.conv4_3 = BottleNeckBlock.add_block(1024, 256, 1024)
		self.conv4_4 = BottleNeckBlock.add_block(1024, 256, 1024)
		self.conv4_5 = BottleNeckBlock.add_block(1024, 256, 1024)
		self.conv4_6 = BottleNeckBlock.add_block(1024, 256, 1024)

		self.conv5_1 = BottleNeckBlock.add_block(1024, 512, 2048, True)
		self.conv5_2 = BottleNeckBlock.add_block(2048, 512, 2048)
		self.conv5_3 = BottleNeckBlock.add_block(2048, 512, 2048)
	
		self.fc1 = nn.Linear(2048 * 7 * 7, 1000)

		self.p1 = BottleNeckBlock.add_projection(64, 256, False) #projection 연산 (차원 일치)
		self.p2 = BottleNeckBlock.add_projection(256, 512, True)
		self.p3 = BottleNeckBlock.add_projection(512, 1024, True)
		self.p4 = BottleNeckBlock.add_projection(1024, 2048, True)
	
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, padding=1)

		input = x; x = F.relu(self.conv2_1(x) + self.p1(input))
		input = x; x = F.relu(self.conv2_2(x) + input)
		input = x; x = F.relu(self.conv2_3(x) + input)
		input = x; x = F.relu(self.conv3_1(x) + self.p2(input))
		input = x; x = F.relu(self.conv3_2(x) + input)
		input = x; x = F.relu(self.conv3_3(x) + input)
		input = x; x = F.relu(self.conv3_4(x) + input)
		input = x; x = F.relu(self.conv4_1(x) + self.p3(input))
		input = x; x = F.relu(self.conv4_2(x) + input)
		input = x; x = F.relu(self.conv4_3(x) + input)
		input = x; x = F.relu(self.conv4_4(x) + input)
		input = x; x = F.relu(self.conv4_5(x) + input)
		input = x; x = F.relu(self.conv4_6(x) + input)
		input = x; x = F.relu(self.conv5_1(x) + self.p4(input))
		input = x; x = F.relu(self.conv5_2(x) + input)
		input = x; x = F.relu(self.conv5_3(x) + input)
	
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		return x