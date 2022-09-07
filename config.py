import os

"""General parameters"""
ROOT_PATH = "/home/yemika/Mikael/Code/Python/Kaggle/mnist"
DATA_PATH = os.path.join(ROOT_PATH, "data/")
NUM_CLASSES = 10
SAVE_PATH = "saved_models/"
MODEL_NAME = "model.pt"

"""Train parameters"""
N_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
LOAD_MODEL = True