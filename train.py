import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import config
from dataset import MNISTDataset
from model import MNISTEfficientNet
from utils import create_loss_fn, create_optimizer, create_scheduler, make_train_step, preprocess_batch, OH_encode, plot_training_info

import pdb

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
DATA_PATH = config.DATA_PATH
BATCH_SIZE = config.BATCH_SIZE
VAL_FREQ = config.VAL_FREQ
LR = config.LR
N_EPOCHS = config.N_EPOCHS
NUM_CLASSES = config.NUM_CLASSES
LOAD_MODEL = config.LOAD_MODEL
SAVE_PATH = config.SAVE_PATH
MODEL_NAME = config.MODEL_NAME
SCHEDULER_STEP = config.SCHEDULER_STEP
SCHEDULER_GAMMA = config.SCHEDULER_GAMMA
PLOT_RESULTS = config.PLOT_RESULTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Argument parsing (can replace arguments from config.py)"""
parser = argparse.ArgumentParser(description="MNIST model arguments")
parser.add_argument('--MODEL_NAME', type=str, default=MODEL_NAME, help='model name')
args = parser.parse_args()
MODEL_NAME = args.MODEL_NAME

"""Load the data set"""
train_path = os.path.join(DATA_PATH, "train.csv")
val_path = os.path.join(DATA_PATH, "val.csv")
train_data = MNISTDataset(train_path, device=device)
val_data = MNISTDataset(val_path, device=device)

"""Create the dataloaders"""
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

"""Initialize the model"""
model = MNISTEfficientNet(device=device)
if LOAD_MODEL:
    model_path = os.path.join(SAVE_PATH, MODEL_NAME)
    model.load(model_path)

"""Define the loss function and the optimizer"""
loss_fn = create_loss_fn()
optimizer = create_optimizer(model, LR)
scheduler = create_scheduler(optimizer, SCHEDULER_GAMMA)

"""Training"""
train_step = make_train_step(model, loss_fn, optimizer)
train_losses = list()
val_losses = list()
LRs = list()

for epoch in range(N_EPOCHS):
    t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
    for images_batch, labels_batch in t:
        images_batch = torch.unsqueeze(images_batch, 1)
        images_batch = preprocess_batch(images_batch)
        labels_batch = OH_encode(labels_batch, NUM_CLASSES).to(device)

        loss = train_step(images_batch, labels_batch)
        
        train_losses.append(loss)
        LRs.append(optimizer.state_dict()['param_groups'][0]['lr'])

        t.set_description(f"Epoch: #{epoch + 1}. Loss: {round(loss, 4)}. LR: {optimizer.state_dict()['param_groups'][0]['lr']}")

    if (epoch + 1) % VAL_FREQ == 0:
        print("Running validation...")
        with torch.no_grad():
            t = tqdm(val_loader)
            for images_batch, labels_batch in t:
                images_batch = torch.unsqueeze(images_batch, 1)
                images_batch = preprocess_batch(images_batch)
                labels_batch = OH_encode(labels_batch, NUM_CLASSES).to(device)

                model.eval()
                preds = model(images_batch)
                val_loss = loss_fn(preds, labels_batch)
                val_losses.append(val_loss.item())
                t.set_description(f"Epoch: #{epoch + 1}. Validation loss: {round(val_loss.item(), 4)}.")

    if (epoch + 1) % SCHEDULER_STEP == 0 and epoch != 0:
        scheduler.step()
    
    model.save(MODEL_NAME)

if PLOT_RESULTS:
    plot_training_info(train_losses, val_losses, LRs)