import os

import config

import pdb

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
filename = "digit-recognizer.zip"
raw_data_path = os.path.join(ROOT_PATH, "raw_data/")
data_path = os.path.join(raw_data_path, filename)

if os.path.exists(data_path):
    print("File already exists.")
else:
    os.system(f"kaggle competitions download -c digit-recognizer -p {raw_data_path} --force")
os.chdir("raw_data")
os.system(f"unzip {filename}")
os.system("mv train.csv train_original.csv")
os.system("cp train_original.csv test.csv ../data")
os.chdir("..")
os.system("python trainval_split.py")