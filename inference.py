import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse

import config
from dataset import MNISTDataset
from model import MNISTEfficientNet
from utils import evaluate, preprocess_batch, plot_layer, get_activation

import pdb
from matplotlib import pyplot as plt

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
TEST_DATASET = config.TEST_DATASET
PREDS_PATH = config.PREDS_PATH
PLOT_FILTERS = config.PLOT_FILTERS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Argument parsing (can replace arguments from config.py)"""
parser = argparse.ArgumentParser(description="MNIST model arguments")
parser.add_argument('--PREDS_PATH', type=str, default=PREDS_PATH, help='path to save the predictions')
parser.add_argument('--MODEL_NAME', type=str, default=MODEL_NAME, help='model name')
args = parser.parse_args()
PREDS_PATH = args.PREDS_PATH
MODEL_NAME = args.MODEL_NAME

"""Load the data set"""
test_path = os.path.join(DATA_PATH, "test.csv")
test_data = MNISTDataset(test_path, test_dataset=TEST_DATASET, device=device)

"""Create the dataloader"""
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

"""Load the model"""
model = MNISTEfficientNet(device=device)
model_path = os.path.join(SAVE_PATH, MODEL_NAME)
model.load(model_path)

"""Evaluation metric setup"""
total_correct = 0
total = len(test_data)

"""Set up layer visualization"""
if PLOT_FILTERS:
    activation = {}
    custom_handle = model.backbone.layers[0][0].proj.register_forward_hook(get_activation('conv', activation))

"""Run inference"""
if TEST_DATASET:    # data set without known labels
    pred_labels = torch.empty((0)).long().to(device)
    for images_batch in tqdm(test_loader):
        images_batch = torch.unsqueeze(images_batch, 1)
        images_batch = preprocess_batch(images_batch)

        preds = evaluate(model, images_batch)
        
        if PLOT_FILTERS:
            plot_layer(activation)

        pred_labels_ = torch.argmax(preds, dim=1).long()
        pred_labels = torch.cat((pred_labels, pred_labels_))
    
    # Convert pred_labels to a Pandas Dataframe
    pred_labels = pd.DataFrame(pred_labels.cpu().numpy(), columns=['Label'])
    pred_labels.index.names = ['ImageId']
    pred_labels.reset_index(inplace=True)
    pred_labels['ImageId'] += 1
    print(f"Saving predictions to {PREDS_PATH}...")
    pred_labels.to_csv(PREDS_PATH, index=False)
    print("Predictions successfully saved!")
else:               # data set with known labels
    for images_batch, labels_batch in tqdm(test_loader):
        images_batch = torch.unsqueeze(images_batch, 1)
        images_batch = preprocess_batch(images_batch)

        preds = evaluate(model, images_batch)
        
        for k in range(len(preds)):
            total_correct += (torch.argmax(preds[k]) == labels_batch[k]).item()
    
    print(f"Accuracy: {round(total_correct / total * 100, 2)}%.")