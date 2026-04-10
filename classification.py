import torch.nn as nn
from models.layers import CustomDropout

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512*7*7,1024),
            nn.ReLU(),
            CustomDropout(0.5),
            nn.Linear(1024,37)
        )

    def forward(self,x):
        return self.fc(x)