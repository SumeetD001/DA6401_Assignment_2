import torch.nn as nn

class UNetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512,256,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1,2,2)
        )

    def forward(self,x):
        return self.net(x)