import os

import config

print("BEFORE RUNNING THIS SCRIPT CHECK THE `ROOT_PATH` VARIABLE IN config.py -- IT MUST BE SET TO THE ABSOLUTE PATH OF THE ROOT DIRECTORY OF THE LOCAL COPY OF THIS REPOSITORY ON YOUR MACHINE.")

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
filename = "digit-recognizer.zip"
raw_data_path = os.path.join(ROOT_PATH, "raw_data/")
data_path = os.path.join(raw_data_path, filename)

if not os.path.exists(raw_data_path):
    os.system("mkdir raw_data")
if os.path.exists(data_path):
    print("File already exists.")
else:
    os.system(f"kaggle competitions download -c digit-recognizer -p {raw_data_path} --force")
os.chdir("raw_data")
os.system(f"unzip {filename}")
os.system("mv train.csv train_original.csv")
os.chdir("..")
if not os.path.exists(os.path.join(ROOT_PATH, "data")):
    os.system("mkdir data")
os.system("cp raw_data/train_original.csv raw_data/test.csv data")
os.system("python trainval_split.py")