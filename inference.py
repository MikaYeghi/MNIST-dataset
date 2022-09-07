import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MNISTDataset
from model import MNISTEfficientNet
from utils import evaluate, preprocess_batch, OH_encode

import pdb

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
DATA_PATH = config.DATA_PATH
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
N_EPOCHS = config.N_EPOCHS
NUM_CLASSES = config.NUM_CLASSES
LOAD_MODEL = config.LOAD_MODEL
SAVE_PATH = config.SAVE_PATH
MODEL_NAME = config.MODEL_NAME
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Load the data set"""
test_path = os.path.join(DATA_PATH, "train.csv")
test_data = MNISTDataset(test_path, device)

"""Create the dataloader"""
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

"""Load the model"""
model = MNISTEfficientNet(device=device)
model_path = os.path.join(SAVE_PATH, MODEL_NAME)
model.load(model_path)

"""Evaluation metric setup"""
total_correct = 0
total = len(test_data)

"""Run inference"""
for images_batch, labels_batch in tqdm(test_loader):
    images_batch = torch.unsqueeze(images_batch, 1)
    images_batch = preprocess_batch(images_batch)

    preds = evaluate(model, images_batch)
    
    for k in range(len(preds)):
        total_correct += (torch.argmax(preds[k]) == labels_batch[k]).item()
    
print(f"Accuracy: {round(total_correct / total * 100, 2)}%.")