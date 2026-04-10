import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)