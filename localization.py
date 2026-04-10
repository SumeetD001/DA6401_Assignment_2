import torch.nn as nn

class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512*7*7,256),
            nn.ReLU(),
            nn.Linear(256,4)
        )

    def forward(self,x):
        return self.fc(x)