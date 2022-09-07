import torch
from torch import nn
import os

import pdb

class MNISTEfficientNet(nn.Module):
    def __init__(self, device='cuda', SAVE_PATH='saved_models/') -> None:
        super().__init__()

        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.backbone.classifier.fc = torch.nn.Linear(1280, 10)
        self.backbone.stem.conv = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.device = device
        self.SAVE_PATH = SAVE_PATH
        self.softmax = self.make_softmax()

        self.to(device)

    def forward(self, x):
        preds = self.backbone(x)
        preds = self.softmax(preds)
        return preds
    
    def to(self, device):
        self.backbone.to(device)

    def save(self, MODEL_NAME):
        """Saves the model to a file."""
        print("Saving the model...")
        path = os.path.join(self.SAVE_PATH, MODEL_NAME)
        torch.save(self.backbone.state_dict(), path)
    
    def load(self, path):
        """Loads the model from a file."""
        try:
            self.backbone.load_state_dict(torch.load(path))
            print(f"Model {path} loaded successfully.")
        except Exception as e:
            print(e)
    
    def make_softmax(self):
        return torch.nn.Softmax()