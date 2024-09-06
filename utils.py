import argparse

def add_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-m", "--model", dest="model", action="store")
	parser.add_argument("-d", "--dataset", dest="dataset",action="store")
	parser.add_argument("-b", "--batch_size", dest="batch_size",action="store")
	parser.add_argument("-l", "--lr", dest="lr",action="store")
	parser.add_argument("-e", "--epoch", dest="epoch", action="store")

	return parser.parse_args()