import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms

def load_data(dataset):
	if dataset == "cifar10":
		train_data = datasets.CIFAR10(
			root="./data",
			train=True,
			download=True,
			transform= transforms.Compose([
				transforms.RandomResizedCrop(32),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])
		)
		test_data = datasets.CIFAR10(
			root="./data",
			train=False,
			download=True,
			transform= transforms.Compose([
				transforms.Resize(32),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])
		)
		return train_data, test_data
  
	elif dataset == "stl10":
		train_data = datasets.STL10(
			root="./data",
			train=True,
			download=True,
			transform= transforms.Compose([
				transforms.RandomResizedCrop(32),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])
		)
		test_data = datasets.STL10(
			root="./data",
			train=False,
			download=True,
			transform= transforms.Compose([
				transforms.Resize(32),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])
		)
  
		return train_data, test_data