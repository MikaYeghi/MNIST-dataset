import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MNISTDataset
from model import MNISTEfficientNet
from utils import create_loss_fn, create_optimizer, create_scheduler, make_train_step, preprocess_batch, OH_encode, plot_training_info

import pdb

"""Configurations"""
ROOT_PATH = config.ROOT_PATH
DATA_PATH = config.DATA_PATH
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
N_EPOCHS = config.N_EPOCHS
NUM_CLASSES = config.NUM_CLASSES
LOAD_MODEL = config.LOAD_MODEL
SAVE_PATH = config.SAVE_PATH
MODEL_NAME = config.MODEL_NAME
SCHEDULER_STEP = config.SCHEDULER_STEP
SCHEDULER_GAMMA = config.SCHEDULER_GAMMA
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Load the data set"""
train_path = os.path.join(DATA_PATH, "train.csv")
train_data = MNISTDataset(train_path, device)

"""Create the dataloaders"""
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

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
LRs = list()
print("-" * 80)

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
    
    if (epoch + 1) % SCHEDULER_STEP == 0 and epoch != 0:
        scheduler.step()
    
    model.save(MODEL_NAME)

plot_training_info(train_losses, LRs)