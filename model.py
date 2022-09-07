import torch
from torch import nn
import pdb

class MNISTEfficientNet(nn.Module):
    def __init__(self, device='cuda') -> None:
        super().__init__()

        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.backbone.classifier.fc = torch.nn.Linear(1280, 10)
        self.backbone.stem.conv = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.device = device

        self.to(device)

    def forward(self, x):
        preds = self.backbone(x)
        return preds
    
    def to(self, device):
        self.backbone.to(device)