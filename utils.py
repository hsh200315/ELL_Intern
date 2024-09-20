import argparse
import random

def add_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", dest="model", action="store")
    parser.add_argument("-d", "--dataset", dest="dataset",action="store")
    parser.add_argument("-b", "--batch_size", dest="batch_size",action="store")
    parser.add_argument("-l", "--lr", dest="lr",action="store")
    parser.add_argument("-e", "--epoch", dest="epoch", action="store")
    parser.add_argument("-n", "--layer", dest="layer", nargs='+', type=int)
    parser.add_argument("-bl", "--block", dest="block", action="store")

    return parser.parse_args()

def is_true():
    p = 0.5
    if p >= random.random():
        return True
    else:
        return False