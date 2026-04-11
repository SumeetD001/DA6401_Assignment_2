"""
Classification model: a thin wrapper around VGG11.
Exists so the autograder can import it separately if needed and to
make the multi-task wiring explicit.
"""

import torch.nn as nn
from models.vgg11 import VGG11Encoder


class ClassificationModel(nn.Module):
    """37-class pet-breed classifier built on VGG11."""

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.model = VGG11(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x):
        return self.model(x)
