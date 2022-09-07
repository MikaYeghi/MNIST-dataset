import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
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
TEST_DATASET = config.TEST_DATASET
PREDS_PATH = config.PREDS_PATH
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

"""Run inference"""
if TEST_DATASET:    # data set without known labels
    pred_labels = torch.empty((0)).long().to(device)
    for images_batch in tqdm(test_loader):
        from matplotlib import pyplot as plt
        # for image in images_batch:
        #     plt.imshow(image.cpu().numpy())
        #     plt.show()
        images_batch = torch.unsqueeze(images_batch, 1)
        images_batch = preprocess_batch(images_batch)

        preds = evaluate(model, images_batch)
        pred_labels_ = torch.argmax(preds, dim=1).long()
        pred_labels = torch.cat((pred_labels, pred_labels_))
    # Convert pred_labels to a Pandas Dataframe
    pred_labels = pd.DataFrame(pred_labels.cpu().numpy())
    print(f"Saving predictions to {PREDS_PATH}...")
    pred_labels.to_csv(PREDS_PATH, index=True)
    print("Predictions successfully saved!")
else:               # data set with known labels
    for images_batch, labels_batch in tqdm(test_loader):
        images_batch = torch.unsqueeze(images_batch, 1)
        images_batch = preprocess_batch(images_batch)

        preds = evaluate(model, images_batch)
        
        for k in range(len(preds)):
            total_correct += (torch.argmax(preds[k]) == labels_batch[k]).item()
    
    print(f"Accuracy: {round(total_correct / total * 100, 2)}%.")