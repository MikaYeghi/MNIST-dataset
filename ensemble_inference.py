import os
import torch
import pandas as pd
import numpy as np

import config
from model import MNISTEfficientNet

import pdb

"""CONFIG"""
N_MODELS = config.N_MODELS
ROOT_PATH = config.ROOT_PATH
SAVE_PATH = config.SAVE_PATH
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Generate N models"""
model_names = [f"model_{i + 1}.pt" for i in range(N_MODELS)]
k = 0
print("Starting ensemble training...")
for model_name in model_names:
    # Model training
    os.system(f"python train.py --MODEL_NAME={model_name}")
    model_path = os.path.join(ROOT_PATH, SAVE_PATH, model_name)

    # Inference
    PREDS_PATH = os.path.join(ROOT_PATH, f"predictions/{model_name.split('.')[0]}.csv")
    os.system(f"python inference.py --PREDS_PATH={PREDS_PATH} --MODEL_NAME={model_name}")

    # Load the predictions from the csv file
    model_preds = pd.read_csv(PREDS_PATH)
    if k == 0:
        total_preds = model_preds.copy()
        k += 1
    else:
        total_preds = pd.concat([total_preds, model_preds.Label], axis=1)
print("Ensemble training finished!")

print("Generating predictions...")
ensemble_preds = pd.DataFrame(index=np.arange(len(total_preds)), columns=['ImageId', 'Label'])
ensemble_preds.ImageId = np.arange(len(total_preds)) + 1
labels = np.empty(len(total_preds))
for i in range(len(labels)):
    values, counts = np.unique(total_preds.Label.iloc[i].values, return_counts=True)
    labels[i] = values[np.argmax(counts)]
    i += 1
ensemble_preds.Label = labels
ensemble_preds = ensemble_preds.astype('int32')
print("Ensemble predictions have been generated!")

submission_path = os.path.join(ROOT_PATH, "predictions/submission.csv")
print(f"Saving the predictions to {submission_path}...")
ensemble_preds.to_csv(submission_path, index=False)