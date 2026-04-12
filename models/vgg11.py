import torch
import torch.nn as nn
from models.layers import CustomDropout

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """
    VGG-11 with BatchNorm and CustomDropout.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 37 for Oxford-IIIT Pet dataset).
    dropout_p   : float
        Dropout probability applied in the classifier head (default 0.5).
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 224×224 → 112×112
            _conv_bn_relu(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 — 112×112 → 56×56
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 — 56×56 → 28×28
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 — 28×28 → 14×14
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5 — 14×14 → 7×7
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


  
    def get_backbone(self) -> nn.Sequential:
        """Return only the convolutional feature extractor."""
        return self.features
