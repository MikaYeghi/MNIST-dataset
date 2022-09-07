import os

"""General parameters"""
ROOT_PATH = "/home/yemika/Mikael/Code/Python/Kaggle/mnist"
DATA_PATH = os.path.join(ROOT_PATH, "data/")
NUM_CLASSES = 10

"""Train parameters"""
N_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001