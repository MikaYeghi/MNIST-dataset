import pandas as pd
import os
from sklearn.model_selection import train_test_split

import config

"""CONFIG"""
DATA_PATH = config.DATA_PATH

"""Load the CSV file"""
original_train_path = os.path.join(DATA_PATH, "train_original.csv")
data = pd.read_csv(original_train_path)
train_path = os.path.join(DATA_PATH, "train.csv")
val_path = os.path.join(DATA_PATH, "val.csv")

# Randomly split the data (NOTE: might result in an imbalanced split)
train_data, val_data = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=True)

# Save the training and validation sets
train_data.to_csv(train_path, index=False)
val_data.to_csv(val_path, index=False)

print(f"Data successfully splitted and saved to:\ntrain data: {train_path}\nvalidation data: {val_path}.")