import os

"""General parameters"""
ROOT_PATH = "/home/yemika/Mikael/Code/Python/Kaggle/mnist"
DATA_PATH = os.path.join(ROOT_PATH, "data/")
NUM_CLASSES = 10
SAVE_PATH = "saved_models/"
MODEL_NAME = "model.pt"

"""Ensemble training parameters"""
N_MODELS = 11            # number of models in the ensemble

"""Train parameters"""
N_EPOCHS = 30
BATCH_SIZE = 1024
LR = 0.001
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.5
LOAD_MODEL = False
VAL_FREQ = 5
PLOT_RESULTS = False

"""Test parameters"""
TEST_DATASET = True
PREDS_PATH = os.path.join(ROOT_PATH, "predictions/submission.csv")