from torch.utils.data import Dataset
import pandas as pd
import torch
import pdb

class MNISTDataset(Dataset):
    def __init__(self, data_path, device='cuda') -> None:
        super().__init__()

        self.data_path = data_path
        self.device = device

        self.data = self.extract_data(self.data_path)

    def extract_data(self, data_path):
        return pd.read_csv(data_path)

    def __getitem__(self, index):
        image_info = self.data.iloc[index]
        
        label = image_info.label
        image = torch.tensor(image_info[1:].values).reshape(28, 28).to(self.device)

        return (image, label)
    
    def __len__(self):
        return len(self.data)