import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,padding=1), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512,512,3,padding=1), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self,x):
        return self.features(x)
