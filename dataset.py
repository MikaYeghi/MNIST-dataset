from torch.utils.data import Dataset
import pandas as pd
import torch
import pdb

class MNISTDataset(Dataset):
    def __init__(self, data_path, test_dataset=False, device='cuda') -> None:
        super().__init__()

        self.data_path = data_path
        self.test_dataset = test_dataset
        self.device = device

        self.data = self.extract_data(self.data_path)
        print(f"Loaded a data set with {self.__len__()} images.")

    def extract_data(self, data_path):
        return pd.read_csv(data_path)

    def __getitem__(self, index):
        image_info = self.data.iloc[index]

        if self.test_dataset:
            image = torch.tensor(image_info.values).reshape(28, 28).to(self.device)
            return image
        else:
            label = image_info.label
            image = torch.tensor(image_info[1:].values).reshape(28, 28).to(self.device)
            return (image, label)
    
    def __len__(self):
        return len(self.data)